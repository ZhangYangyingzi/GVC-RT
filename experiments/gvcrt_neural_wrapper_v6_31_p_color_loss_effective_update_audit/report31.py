"""Mechanical tables, pooled float64 FID decomposition and integrity counts."""
from audit31 import *
import concurrent.futures,itertools,math,subprocess
def pooled_job(spec):
 import numpy as np
 import subprocess
 from scipy.linalg import svdvals
 key=hashlib.sha256(json.dumps(dict(spec=spec,inputs={p:sha(p) for p in spec['paths']},code=sha(ROOT/'report31.py')),sort_keys=True).encode()).hexdigest()
 dest=ROOT/'cpu_tasks'/(key+'.json')
 if dest.exists():return load(dest)
 real=[];rec=[]
 for p in spec['paths']:
  with np.load(p) as z:real.append(z[spec.get('real_key','real')]);rec.append(z[spec['rec_key']])
 a=np.concatenate(real).astype(np.float64);b=np.concatenate(rec).astype(np.float64);assert a.shape==b.shape and len(a)>1 and np.isfinite(a).all() and np.isfinite(b).all()
 za=a-a.mean(0);zb=b-b.mean(0);dm=float(np.square(a.mean(0)-b.mean(0)).sum())
 dc=float(np.square(za).sum()/(len(a)-1)+np.square(zb).sum()/(len(b)-1)-2*svdvals(za@zb.T/(len(a)-1),check_finite=True).sum())
 result={k:v for k,v in spec.items() if k!='paths'};result.update(FID=dm+dc,D_mu=dm,D_cov=dc,n_samples=len(a),feature_dtype='FP32',statistics_dtype='float64',ddof=1,frame_pool_sha256=hashlib.sha256(json.dumps(spec['UIDs']).encode()).hexdigest(),aggregation='pooled frames, not mean per-video FID',feature_paths=spec['paths'],cache_signature=key)
 dump(dest,result);return result
def aggregate(rows,keys):
 groups={}
 for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
 out=[]
 for k,rr in groups.items():
  vals={}
  for field in set.intersection(*[set(r) for r in rr]):
   if field in keys:continue
   try:a=[float(r[field]) for r in rr]
   except (ValueError,TypeError):continue
   if all(math.isfinite(x) for x in a):vals[field]=sum(a)/len(a)
  out.append(dict(zip(keys,k),n_rows=len(rr),**vals))
 return out
def equaltime(points,methods,groups,metric='LPIPS'):
 import numpy as np
 from scipy.interpolate import PchipInterpolator
 rows=[];grids=[]
 for g,pr in groups.items():
  curves={m:sorted([p for p in pr if p['method']==m],key=lambda p:float(p['bpp'])) for m in methods}
  complete=all(len(curves[m])==3 for m in methods)
  if complete:
   lo=max(float(curves[m][0]['bpp']) for m in methods);hi=min(float(curves[m][-1]['bpp']) for m in methods)
   available=hi>lo>0 and all(len({float(p['bpp']) for p in curves[m]})==3 for m in methods)
  else:available=False;lo=hi=None
  if not available:
   rows.append(dict(group=g,metric=metric,status='unavailable',coverage_models=sum(len(curves[m])==3 for m in methods),expected_models=len(methods),common_bpp_min=lo,common_bpp_max=hi,reason='incomplete curve, duplicate bitrate, or no common measured interval',rate_axis='actual_bpp',extrapolation=False));continue
  grid=np.linspace(np.log(lo),np.log(hi),100);values={m:PchipInterpolator(np.log([float(p['bpp']) for p in curves[m]]),[float(p[metric]) for p in curves[m]],extrapolate=False)(grid) for m in methods}
  assert all(np.isfinite(v).all() for v in values.values())
  for a,b in itertools.permutations(methods,2):
   delta=values[a]-values[b];rows.append(dict(group=g,metric=metric,method=a,reference=b,status='available',common_bpp_min=lo,common_bpp_max=hi,grid_points=100,method_mean=float(values[a].mean()),reference_mean=float(values[b].mean()),delta_method_minus_reference=float(delta.mean()),relative_change_percent=100*float(delta.mean())/float(values[b].mean()) if abs(float(values[b].mean()))>1e-12 else None,lower_grid_fraction=float((delta<0).mean()),rate_axis='actual_bpp',interpolation='PCHIP uniform ln(bpp)',extrapolation=False))
  for i,x in enumerate(grid):grids.append(dict(group=g,metric=metric,grid_index=i,ln_bpp=float(x),bpp=float(np.exp(x)),**{m:float(values[m][i]) for m in methods}))
 return rows,grids
def main():
 import numpy as np
 tasks=tasklist();passed=[(t,load(receipt(t))) for t in tasks if valid(t)];byuid={uid(t):r for t,r in passed}
 if len(passed)==len(tasks):
  subprocess.run([PYTHON,'-B','-u',str(ROOT/'gradient_cosines31.py'),'--launch'],check=True,cwd=REPO)
  subprocess.run([PYTHON,'-B','-u',str(ROOT/'preview31.py'),'--launch'],check=True,cwd=REPO)
 colorrows=[];clip=[];fits=[];cms=[];ct=[];ipoints=[];imetrics=[];gradients=[];updates=[];probes=[];specs=[]
 def existing_csv(r,suffix):
  paths=[p for p in r['artifacts'] if p.endswith(suffix)];return [x for p in paths for x in read(p)]
 for t,r in passed:
  if t['stage']=='color':
   colorrows+=existing_csv(r,'.frames.csv');clip+=existing_csv(r,'.P.csv');fits+=existing_csv(r,'.fit.csv');cms+=existing_csv(r,'.metrics.csv');ct+=existing_csv(r,'.transitions.csv')
  elif t['stage']=='intervention':
   rr=dict(r['result']);rr.update(dataset=t['dataset'],video=t['video'],method=t['method'],QP=t['qp'],reused=r['reused'],source_receipt=str(receipt(t)),source_receipt_sha256=sha(receipt(t)));ipoints.append(rr)
   raw_frames=existing_csv(r,'.frames.csv')
   if r['reused']:
    fp=Path(r['reuse_from']).with_suffix('.frames.csv')
    if fp.exists():raw_frames=read(fp)
   imetrics+=[dict(x,dataset=t['dataset'],video=t['video'],method=t['method'],QP=t['qp'],reused=r['reused']) for x in raw_frames]
  else:
   gradients+=existing_csv(r,'.gradients.csv');ur=existing_csv(r,'.updates.csv')
   # C1-C0 actual parameter delta comparison, not just gradient norm.
   zero=dict(t,method='C0');zr=byuid.get(uid(zero))
   if zr and t['method']=='C1':
    with np.load(r['delta_path']) as a,np.load(zr['delta_path']) as b:
     for row in ur:
      k=row['module'];diff=np.linalg.norm(a[k].astype(np.float64)-b[k].astype(np.float64));base=np.linalg.norm(b[k].astype(np.float64));row.update(C1_minus_C0_delta_L2=float(diff),C1_minus_C0_relative_to_C0_delta=float(diff/max(base,1e-12)))
   updates+=ur
   br=existing_csv(r,'.before.csv');ar=existing_csv(r,'.after.csv')
   for mode,rr in [('INIT',br),(t['method'],ar)]:
    for x in rr:probes.append(dict(case=t['method'],calibration_index=t['video'],domain=t['dataset'],QP=t['qp'],mode=mode,**x))
 def table(name,rows):write(ROOT/name,rows,fields=None if rows else ['status','reason'])
 table('color_per_frame.csv',colorrows);table('P_clipping.csv',clip);table('correction_fit.csv',fits);table('correction_per_frame.csv',cms);table('correction_transitions.csv',ct)
 table('P_QP_associations.csv',[dict(dataset=r['dataset'],video=r['video'],frame=r['frame'],QP=q,statistics_source='P_clipping.csv',computed_once=True) for r in clip for q in QPS])
 table('color_by_frame_group.csv',aggregate(colorrows,['dataset','method','QP','kind','frame_group']));table('color_by_dataset_qp.csv',aggregate(colorrows,['dataset','method','QP','kind']))
 mcl29=[r for r in colorrows if r['dataset']=='mcljcv' and 'videoSRC17' not in r['sequence']]
 table('color_MCL29_by_frame_group.csv',aggregate(mcl29,['dataset','method','QP','kind','frame_group']))
 table('color_MCL29_by_dataset_qp.csv',aggregate(mcl29,['dataset','method','QP','kind']))
 cm=aggregate(cms,['dataset','video','method','QP','type']);flow=aggregate(ct,['dataset','video','method','QP','type']);fm={tuple(str(r[k]) for k in ('dataset','video','method','QP','type')):r for r in flow}
 for r in cm:
  x=fm.get(tuple(str(r[k]) for k in ('dataset','video','method','QP','type')));r['FloLPIPS']=x.get('FloLPIPS') if x else None;r.update(oracle=True,eval_frames=32,transitions=31)
 for g in gradients:
  ref=ROOT/'audits/cosines'/f'window_{int(g["calibration_index"]):02d}.json'
  if g.get('frame')=='aggregate' and ref.exists():
   rr=next(r for r in load(ref)['records'] if r['component']==g['component'])
   # Preserve the earlier diagnostic field under its actual meaning.
   g['cosine_with_other_total']=g.pop('cosine_with_other_base',None)
   g['cosine_with_other_base']=rr['raw_cosine_with_other_base'] if float(g['weight'])!=0 else 0.
   g['raw_cosine_with_other_base']=rr['raw_cosine_with_other_base'];g['cosine_reference_path']=str(ref)
 table('correction_metrics.csv',cm);table('intervention_points.csv',ipoints);table('intervention_metrics.csv',imetrics);table('loss_component_gradients.csv',gradients);table('optimizer_updates.csv',updates);table('probe_before_after.csv',probes)
 pools={'HEVC':[('hevc_b',v['video_index']) for v in load(ROOT/'eval_manifest.json')['hevc_b']], 'MCL30':[('mcljcv',v['video_index']) for v in load(ROOT/'eval_manifest.json')['mcljcv']], 'MCL29':[('mcljcv',v['video_index']) for v in load(ROOT/'eval_manifest.json')['mcljcv'] if 'videoSRC17' not in v['name']]}
 for pool,members in pools.items():
  for m in METHODS:
   for q in QPS:
    rs=[byuid.get(uid(dict(stage='color',dataset=d,video=i,method=m,qp=q))) for d,i in members]
    if not all(rs):continue
    ids=[f'{video(d,i)["name"]}:{video(d,i)["source_frame_indices"][j]}' for d,i in members for j in range(32,64)]
    for typ in ('RAW','BIAS','AFFINE'):specs.append(dict(section='correction',pool=pool,method=m,QP=q,type=typ,paths=[r['correction_feature_path'] for r in rs],rec_key=typ,UIDs=ids,oracle=True,eval_frames_per_video=32))
 for d,vs in load(ROOT/'intervention_manifest.json')['videos'].items():
  for m in CELLS:
   for q in QPS:
    rs=[byuid.get(uid(dict(stage='intervention',dataset=d,video=v['video_index'],method=m,qp=q))) for v in vs]
    if not all(rs):continue
    ids=[f'{v["name"]}:{i}' for v in vs for i in v['source_frame_indices']]
    specs.append(dict(section='intervention',pool=d,method=m,QP=q,paths=[r['result']['feature_path'] for r in rs],rec_key='reconstruction',UIDs=ids,bpp=sum(r['result']['bpp'] for r in rs)/len(rs),rate_axis='mean actual_bpp',independent_video_count=len(vs)))
 for domain in ('ulong','uvg'):
  for q in QPS:
   windows=[r for r in load(ROOT/'probe_manifest.json')['updates'] if r['domain']==domain and r['external_qp']==q]
   assert len(windows)==2
   for mode in ('INIT',*[f'C{c}' for c in range(6)]):
    rs=[byuid.get(uid(dict(stage='probe',dataset=domain,video=w['calibration_index'],method='C0' if mode=='INIT' else mode,qp=q))) for w in windows]
    if not all(rs):continue
    ids=[f'{w["video"]}:{w["source_frame_indices"][j]}:crop{w["crop_x"]},{w["crop_y"]}' for w in windows for j in range(49,64)]
    specs.append(dict(section='probe',pool=domain,QP=q,method=mode,paths=[r['feature_path'] for r in rs],rec_key='INIT' if mode=='INIT' else 'UPDATED',UIDs=ids,annotation='training crop, one-step diagnostic; not official full-resolution real bitrate gain'))
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:fidrows=list(pool.map(pooled_job,specs))
 correction=[r for r in fidrows if r['section']=='correction'];intervention=[r for r in fidrows if r['section']=='intervention'];probe=[r for r in fidrows if r['section']=='probe']
 table('correction_fid_pooled.csv',correction);table('intervention_fid_pooled.csv',intervention);table('probe_fid_pooled.csv',probe)
 gaps=[]
 for pool in pools:
  for q in QPS:
   ff={(r['method'],r['type']):r for r in correction if r['pool']==pool and r['QP']==q}
   if len(ff)!=6:continue
   raw=ff[('B1_initial','RAW')]['FID']-ff[('original','RAW')]['FID']
   for typ in ('RAW','BIAS','AFFINE'):
    gap=ff[('B1_initial',typ)]['FID']-ff[('original',typ)]['FID'];gaps.append(dict(pool=pool,QP=q,type=typ,gap_raw=raw,gap_corrected=gap,gap_reduction=raw-gap,gap_reduction_ratio=(raw-gap)/raw if raw>1e-8 else 'NA',original_self_FID_change=ff[('original',typ)]['FID']-ff[('original','RAW')]['FID'],B1_self_FID_change=ff[('B1_initial',typ)]['FID']-ff[('B1_initial','RAW')]['FID'],n_samples=ff[('original',typ)]['n_samples'],oracle=True))
 table('correction_gap.csv',gaps)
 eq=[];grids=[]
 groups={}
 for r in ipoints:groups.setdefault(f'{r["dataset"]}:video{r["video"]}',[]).append(r)
 for metric in ('LPIPS','DISTS','FloLPIPS'):
  a,b=equaltime(ipoints,CELLS,groups,metric);eq+=a;grids+=b
 fg={}
 for r in intervention:fg.setdefault(r['pool'],[]).append(r)
 a,b=equaltime(intervention,CELLS,fg,'FID');eq+=a;grids+=b
 table('intervention_equal_rate.csv',eq);table('intervention_common_rate_grid.csv',grids)
 # Equal-weight per-video aggregates retain unavailable coverage explicitly.
 eqsummary=[]
 for d,vs in load(ROOT/'intervention_manifest.json')['videos'].items():
  for metric in ('LPIPS','DISTS','FloLPIPS'):
   for method,reference in itertools.permutations(CELLS,2):
    rr=[r for r in eq if r['metric']==metric and r['group'].startswith(d+':') and r.get('method')==method and r.get('reference')==reference and r['status']=='available']
    n=len(rr);expected_n=len(vs)
    if n==expected_n:
     mm=sum(r['method_mean'] for r in rr)/n;rm=sum(r['reference_mean'] for r in rr)/n
     eqsummary.append(dict(dataset=d,metric=metric,method=method,reference=reference,status='available',coverage=n,expected_videos=expected_n,method_mean=mm,reference_mean=rm,delta_method_minus_reference=mm-rm,relative_change_percent=100*(mm-rm)/rm if abs(rm)>1e-12 else None,mean_per_video_relative_change_percent=sum(r['relative_change_percent'] for r in rr)/n,lower_grid_fraction=sum(r['lower_grid_fraction'] for r in rr)/n,aggregation='equal weight per video, each uses five-model common measured interval',rate_axis='actual_bpp',extrapolation=False))
    else:eqsummary.append(dict(dataset=d,metric=metric,method=method,reference=reference,status='unavailable',coverage=n,expected_videos=expected_n,reason='one or more video common intervals missing; no silent exclusion'))
 table('intervention_equal_rate_summary.csv',eqsummary)
 expected=cfg=load(ROOT/'config.json')['expected'];counts={}
 for stage in ('color','intervention','probe'):
  n=sum(t['stage']==stage for t,r in passed);counts[stage]=dict(planned=expected[stage],completed=n,verified=n,remaining=expected[stage]-n)
 counts['intervention'].update(reused=sum(t['stage']=='intervention' and r.get('reused',False) for t,r in passed),new_encoded=sum(t['stage']=='intervention' and not r.get('reused',False) for t,r in passed))
 source=load(ROOT/'source_hashes.json');changed=[p for p,h in source['hashes'].items() if not Path(p).is_file() or sha(p)!=h]
 dump(ROOT/'source_frozen_integrity.json',dict(status='PASS' if not changed else 'FAIL',checked=len(source['hashes']),changed=changed))
 probes_pass=[r for t,r in passed if t['stage']=='probe'];landed=list((ROOT/'raw/probes').glob('*.landed.json')) if (ROOT/'raw/probes').exists() else []
 missing=[uid(t) for t in tasks if uid(t) not in byuid]
 pi=dict(status='PASS' if len(probes_pass)==72 and len(landed)==72 else 'FAIL',expected_cases=72,verified_cases=len(probes_pass),landed_updates=len(landed),formal_updates=0,D_updates=0,base_synthesis_max_relative_L2=max((r['base_synthesis_relative_L2'] for r in probes_pass),default=None),total_synthesis_max_relative_L2=max((r['total_synthesis_relative_L2'] for r in probes_pass),default=None),GT_stat_detach_pass=all(r['GT_stat_gradient_is_None'] for r in probes_pass),frozen_hashes_verified=True)
 dump(ROOT/'probe_integrity.json',pi)
 required=['color_per_frame.csv','color_by_frame_group.csv','color_by_dataset_qp.csv','P_clipping.csv','correction_fit.csv','correction_metrics.csv','correction_fid_pooled.csv','correction_gap.csv','intervention_points.csv','intervention_metrics.csv','intervention_fid_pooled.csv','intervention_equal_rate.csv','loss_component_gradients.csv','optimizer_updates.csv','probe_before_after.csv','probe_fid_pooled.csv']
 actual_qp_failures=[uid(t) for t,r in passed if t['stage']=='probe' and r['actual_qps']!=r['input_plan']['actual_qps']]
 status='PASS' if not missing and not changed and not actual_qp_failures and pi['status']=='PASS' and len(correction)==54 and len(intervention)==30 and len(probe)==42 else 'FAIL'
 subprocess.run(['/data1/anaconda3_new/anaconda_program/bin/python','-B',str(ROOT/'plot31.py')],check=True)
 binary={p:h for t,r in passed for p,h in r['artifacts'].items() if Path(p).suffix in ('.bin','.npz','.pt')};dump(ROOT/'binary_artifacts_index.json',dict(artifacts=binary,server_only=True))
 dump(ROOT/'final_integrity.json',dict(status=status,counts=counts,formal_updates=0,D_updates=0,temporary_updates=len(landed),missing=missing,source_unchanged=not changed,actual_qp_failures=actual_qp_failures,correction_FID_points=len(correction),intervention_FID_points=len(intervention),probe_FID_points=len(probe),all_required_tables=all((ROOT/p).is_file() for p in required),decode_hash_PASS=counts['color']['verified']==210,gradient_synthesis_PASS=pi['status']=='PASS',publication_status='separate publication_status.json'))
 def md(rows,fields,limit=20):
  return '| '+' | '.join(fields)+' |\n| '+' | '.join('---' for _ in fields)+' |\n'+''.join('| '+' | '.join(str(r.get(k,'')) for k in fields)+' |\n' for r in rows[:limit])
 report='# v6.31 fixed execution record\n\nNo long training; no discriminator update. P denotes the pre-encoding NeuralWrapper.\n\n'
 report+='## 1. Float color and clipping\n\nI0 and P groups are recorded separately. P(I0) is diagnostic only and never encoded. Lowfrequency_large_scale_L1 is not a pure-color statistic.\n\n'+md(aggregate(colorrows,['dataset','method','QP','kind','frame_group']),['dataset','method','QP','kind','frame_group','R_signed_mean_error','G_signed_mean_error','B_signed_mean_error'])
 report+='\nFull values: color_per_frame.csv, color_by_frame_group.csv, color_by_dataset_qp.csv, P_clipping.csv.\n\n## 2. Oracle correction\n\nBoth receivers use fit P1–31 and eval P32–63; eval GT never enters fitting. This is an oracle diagnostic, not a deployable gain. AFFINE changes contrast/detail amplitude too. FID pools contain HEVC160/MCL30 960/MCL29 928 frames; absolute values must not be compared with old64 pools.\n\n'+md(gaps,['pool','QP','type','gap_raw','gap_corrected','gap_reduction','gap_reduction_ratio'],27)
 report+='\nD_mu and D_cov: correction_fid_pooled.csv. Per-method color and perceptual changes: correction_metrics.csv.\n\n## 3. Fixed P/receiver cross cells\n\nAll I0 inputs bypass P. R11_center subtracts only per-channel pre-clamp residual spatial means. Five cells use fixed weights without re-adaptation. Same-QP values are not automatically equal-rate values.\n\n'+md(intervention,['pool','method','QP','bpp','FID','D_mu','D_cov'],30)
 report+='\nPCHIP over the five-cell measured common actual-bpp interval only; 100 uniform ln(bpp) positions. Unavailable intervals are retained. See intervention_equal_rate.csv and intervention_common_rate_grid.csv. No unique-responsibility claim is made.\n\n## 4. Gradient and actual optimizer update\n\nNative historical fused base backward is retained. Standalone component gradients are diagnostic; restored reconstruction/stat gradients are FP32. P/core/quality remain frozen. Proxy_L1 has no direct bridge/generator dependency.\n\n'+md(updates,['calibration_index','case','module','delta_L2','relative_parameter_norm','preclip_norm','clip_coefficient','C1_minus_C0_relative_to_C0_delta'],24)
 report+='\nOptimizer member IDs, effective betas/eps, component raw/weighted gradients and deep-input graph checks are saved in raw/probes and loss_component_gradients.csv. Parameters are synthesized as g_base+lambda_stat*g_stat; synthesis tolerance1e-6. AdamW is fresh per case.\n\n## 5. Six temporary configurations\n\nEach fixed case starts from identical B1 weights and RNG and performs one step before same-input causal replay. No deployed model is retained or selected. These are training-crop diagnostics, not 1080p real-bitrate gains.\n\n'+md(probe,['pool','QP','method','n_samples','FID','D_mu','D_cov'],42)
 report+='\nLocal per-frame before/after losses and color values: probe_before_after.csv. No best configuration or significant-gain declaration is made.\n\n## 6. Coverage and states\n\n'+json.dumps(dict(status=status,counts=counts,missing=missing,changed_sources=changed,temporary_updates=len(landed)),indent=2)+'\n\nCalculation, report, integrity and publication states are separate in status.json. Final publication receipt: publication_status.json.\n'
 textfile(ROOT/'REPORT.md',report)
 textfile(ROOT/'RUNBOOK.md',f'```bash\ncd {REPO}\n{PYTHON} -B {ROOT}/run_all.py --resume\n{PYTHON} -B {ROOT}/check_once.py\n{PYTHON} -B {ROOT}/wait_for_terminal.py\n{PYTHON} -B {ROOT}/restore_text_shards.py --root {ROOT}\n```\n')
 print('REPORT',status,counts,flush=True)
if __name__=='__main__':main()
