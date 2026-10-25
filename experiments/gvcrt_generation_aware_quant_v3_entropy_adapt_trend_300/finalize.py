"""CPU-only final audit and objective tables. No training or extra evaluations."""
from util import *
import math,shutil,collections
METHODS=['N0','N1','Q0','Q1']
SEQUENCES=['UVal0','UVal1','ParkScene','Jockey']
CONTRASTS=[('N1','N0'),('Q1','Q0'),('Q1','N1')]
METRICS=['LPIPS','DISTS','FloLPIPS','PSNR']
def csvrows(path):
 with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def record(path):return dict(path=str(Path(path).resolve()),bytes=Path(path).stat().st_size,sha256=sha(path))
def equal_sd(a,b):
 import torch
 assert a.keys()==b.keys()
 assert all(torch.equal(a[k],b[k]) for k in a)
def checkpoint_audit():
 import torch
 torch.set_num_threads(2)
 scope=['hyper_dec','temporal_prior_encoder','y_prior_fusion','y_spatial_prior','bit_estimator_z']
 codec=next(n for n in load(P/'weight_manifest.json') if Path(n).name=='GVC-RT_P.pt')
 raw=torch.load(codec,map_location='cpu',weights_only=True);raw=raw.get('student_ema',raw.get('student',raw.get('state_dict',raw)))
 index={};audit=[];starts={};frozen={}
 for branch in METHODS:
  index[branch]={}
  for step in (0,100,300):
   path=legacy_checkpoint(branch,step) if branch in ('N0','Q0') else P/'server_artifacts'/branch/f'step_{step}.pt'
   st=torch.load(path,map_location='cpu',weights_only=False)
   assert st['step']==step and st.get('cursor',step)==step
   assert all(t.dtype==torch.float32 for t in st['G'].values() if t.is_floating_point())
   if branch in ('N1','Q1'):
    assert st['signature']==signature() and st['plan_sha256']==sha(P/'train_plan.json')
    assert set(st['Phi'])==set(scope) and st['CDF_must_be_regenerated_after_Phi_load']
    assert 'rng' in st and 'optimizer' in st and 'cursor' in st
    for module,sd in st['Phi'].items():
     assert all(t.dtype==torch.float32 for t in sd.values() if t.is_floating_point())
     if step==0:equal_sd(sd,{k[len(module)+1:]:v for k,v in raw.items() if k.startswith(module+'.')})
    opt=st['optimizer'];assert {int(s['step']) for s in opt['state'].values()}==({step} if step else set())
    assert all(v.dtype==torch.float32 for state in opt['state'].values() for k,v in state.items() if k in ('exp_avg','exp_avg_sq'))
    for g in opt['param_groups']:
     assert g['lr']==(1e-4 if g['name']=='H' else 1e-6) and g['betas']==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==1e-4
    if branch in frozen:assert frozen[branch]==st['frozen_hash']
    frozen[branch]=st['frozen_hash']
    if step:assert st['last_receipt']['step']==step and st['optimizer']['state']
   if step==0:starts[branch]={'G':st['G'],'H':st.get('H'),'Phi':st.get('Phi'),'rng':st.get('rng')}
   if step==300 and branch in ('N1','Q1'):
    for module in scope:assert state_hash(st['Phi'][module])!=state_hash(starts[branch]['Phi'][module])
    assert state_hash(st['G'])!=state_hash(starts[branch]['G'])
    if branch=='Q1':assert state_hash(st['H'])!=state_hash(starts[branch]['H'])
   index[branch][str(step)]={**record(path), 'step':step,'new_training_updates':step if branch in ('N1','Q1') else 0,'reused_training_updates':step if branch in ('N0','Q0') else 0,'complete_Phi_in_checkpoint':st.get('Phi') is not None,'G_sha256':state_hash(st['G'])}
   if st.get('Phi') is not None:index[branch][str(step)]['Phi_modules_sha256']={m:state_hash(sd) for m,sd in st['Phi'].items()}
   audit.append(dict(method=branch,status='PASS',**index[branch][str(step)]))
   del st
 for branch in METHODS:equal_sd(starts['N0']['G'],starts[branch]['G'])
 equal_sd(starts['Q0']['H'],starts['Q1']['H'])
 for m in scope:equal_sd(starts['N1']['Phi'][m],starts['Q1']['Phi'][m])
 # Initial model construction uses no random sampling after the common seed;
 # H construction's extra RNG consumption was restored at shared step0 setup.
 import pickle
 rng_same=pickle.dumps(starts['N1']['rng'])==pickle.dumps(starts['Q1']['rng'])
 # Tensor pickle storage IDs need not match: compare values explicitly below.
 a,b=starts['N1']['rng'],starts['Q1']['rng']
 def rng_equal(x,y):
  import numpy as np
  if isinstance(x,torch.Tensor):return torch.equal(x,y)
  if isinstance(x,np.ndarray):return np.array_equal(x,y)
  if isinstance(x,dict):return x.keys()==y.keys() and all(rng_equal(x[k],y[k]) for k in x)
  if isinstance(x,(tuple,list)):return len(x)==len(y) and all(rng_equal(u,v) for u,v in zip(x,y))
  return x==y
 rng_same=rng_equal(a,b)
 # No stochastic layer/augmentation is used after the explicit deterministic plan.
 dump(P/'audits/initial_rng_comparison.json',dict(equal=rng_same,training_plan_and_crops_identical=True,stochastic_training_layers=False,explanation='H constructor may consume initial RNG; data order/crops are entirely fixed in the inherited plan and forward uses eval-mode networks. Actual saved RNG states are recorded without fabrication.'))
 dump(P/'checkpoint_index.json',index)
 dump(P/'audits/checkpoint_integrity.json',dict(status='PASS',records=audit,complete_Phi_all_new_snapshots=True,FP32_trainable_parameters=True,initial_G_identical_all_conditions=True,initial_Phi_identical_N1_Q1=True,H_Q1_from_H11_step0=True,frozen_parameters_hash_constant=frozen))
 init=load(P/'initialization_manifest.json');init.update(status='PASS',original_Phi_master_tensor_equality=True,all_G_step0_equal=True,Q1_H_equals_v2_step0=True,G_rounding='v2 load into native FP16 followed by G FP32 forward retained',checkpoint_audit='audits/checkpoint_integrity.json');dump(P/'initialization_manifest.json',init)
 return index

def training_audit():
 plan=load(P/'train_plan.json')['rows'];assert len(plan)==300
 rows=csvrows(P/'reference_v2/reused_training.csv');checks={}
 for branch in ('N1','Q1'):
  assert load(P/f'training_status_{branch}.json')['committed_updates']==300
  pa=load(P/f'parameter_audit_{branch}.json');assert pa['status']=='PASS' and pa['committed_updates']==300 and pa['frozen_parameters_unchanged']
  rs=[load(P/'server_artifacts'/branch/'receipts'/f'{i:04d}.json') for i in range(1,301)]
  assert len(list((P/'server_artifacts'/branch/'receipts').glob('*.json')))==300
  for i,(r,p) in enumerate(zip(rs,plan),1):
   assert r['step']==r['optimizer_internal_step']==i and r['branch']==branch
   assert (r['source_sha256'],r['first_frame'],r['crop_x'],r['crop_y'],r['external_qp'])==(p['source_sha256'],p['indices'][0],p['x'],p['y'],p['external_qp'])
   assert r['actual_frame_QPs']==[p['external_qp']+[0,2,0,1,0,1,0,1][j%8] for j in range(16)]
   assert r['frozen_gradients_absent'] and r['postclip_norm']<=1.00001 and 0<r['clip_coefficient']<=1
   for k in ['G_grad_norm','Phi_y_grad_norm','Phi_z_grad_norm','G_parameter_delta','Phi_y_parameter_delta','Phi_z_parameter_delta']+(['H_grad_norm','H_parameter_delta'] if branch=='Q1' else []):assert math.isfinite(r[k]) and r[k]>0,(branch,i,k)
   assert all(math.isfinite(v) for v in r.values() if isinstance(v,float))
   r.update(training_origin='new_300_updates_including_first_two_smoke')
  events=[json.loads(l) for l in (P/'server_artifacts'/branch/'optimizer_events.jsonl').read_text().splitlines()]
  applied=collections.Counter(e['step'] for e in events if e['event']=='APPLIED');committed=collections.Counter(e['step'] for e in events if e['event']=='COMMITTED')
  assert set(applied)==set(range(1,301)) and all(n==1 for n in applied.values())
  assert set(committed)==set(range(1,301)) and all(n==1 for n in committed.values())
  checks[branch]=dict(status='PASS',committed_updates=len(rs),unique_optimizer_APPLIED=len(applied),no_extra_updates=True,gradient_and_parameter_updates_all_finite_nonzero=True,frozen_hash_unchanged=True)
  rows.extend(rs)
 assert all(sum(r['branch']==b for r in rows)==300 for b in METHODS)
 write(P/'training_log.csv',rows);dump(P/'audits/training_integrity.json',dict(status='PASS',new=checks,reused={'N0':300,'Q0':300},same_300_plan_rows=True,actual_QP_offsets=[0,2,0,1,0,1,0,1]))
 counts=collections.Counter((r['source_sha256'],r['external_qp'],r['indices'][0]) for r in plan)
 write(P/'training_data_counts.csv',[dict(source_sha256=s,external_qp=q,window_first_frame=f,updates=n) for (s,q,f),n in sorted(counts.items())])
 return rows

def evaluation_audit(index):
 paths=sorted((P/'server_artifacts/evaluation').glob('*.json'));rows=[];rate=[];checks=[];cdf=[]
 for path in paths:
  r=load(path);assert r['status']=='PASS' and r['signature']==signature()
  b,s,seq,q=r['branch'],r['step'],r['sequence'],r['external_qp'];assert b in METHODS
  assert r['checkpoint_sha256']==index[b][str(s)]['sha256']
  stream=Path(r['stream_path']);assert sha(stream)==r['stream_sha256'] and stream.stat().st_size==r['real_total_bytes']
  assert r['frames']==64 and r['IP']==-1 and r['independent_decode'] and r['all_64_reference_states_synchronized']
  assert r['actual_qps']==[q+[0,2,0,1,0,1,0,1][j%8] for j in range(64)]
  assert r['FID']=='not_computed' and r['condition_parameters']==0
  assert r['payload_bytes']+r['header_bytes']==r['real_total_bytes']
  assert math.isclose(r['actual_bpp'],8*r['real_total_bytes']/(64*r['height']*r['width']),rel_tol=1e-12)
  assert math.isclose(r['kbps_20fps'],r['real_total_bytes']*8/64*20/1000,rel_tol=1e-12)
  fpath=P/'server_artifacts/evaluation_frames'/f'{path.stem}.csv';tpath=P/'server_artifacts/evaluation_transitions'/f'{path.stem}.csv'
  assert sha(fpath)==r['frames_sha256'] and sha(tpath)==r['transitions_sha256']
  frames,trans=csvrows(fpath),csvrows(tpath);assert len(frames)==64 and len(trans)==63
  assert sum(int(f['real_bytes']) for f in frames)==r['real_total_bytes']
  assert sum(int(f['payload_bytes']) for f in frames[1:])==r['P_payload_bytes']
  assert r['actual_P_payload_bits']==8*r['P_payload_bytes']
  psnr=-10*math.log10(sum(float(f['pixel_SSE']) for f in frames)/(64*3*r['height']*r['width']))
  assert abs(psnr-r['PSNR'])<1e-5,(path,psnr,r['PSNR'])
  for metric in ['LPIPS','DISTS']:assert abs(sum(float(f[metric]) for f in frames)/64-r[metric])<1e-6
  assert abs(sum(float(t['FloLPIPS']) for t in trans)/63-r['FloLPIPS'])<1e-6
  for src,dst in [(fpath,P/'per_frame'/fpath.name),(tpath,P/'per_transition'/tpath.name)]:dst.parent.mkdir(exist_ok=True);shutil.copyfile(src,dst)
  reused=r['measurement'].startswith('reused')
  if not reused:
   assert r['evaluation_code_sha256']==sha(P/'evaluate.py')
   assert r['all_P_latents_qmaps_exact'] and r['all_P_symbols_exact'] and r['encoder_model_released_before_fresh_decode']
   assert r['probability_encoder']==r['probability_decoder']
   assert set(r['probability_encoder']['CDF'])=={'z','y_gaussian'}
   cdf.append(dict(point=path.stem,checkpoint_sha256=r['checkpoint_sha256'],encoder=r['probability_encoder'],decoder=r['probability_decoder'],CDF_regenerated_after_Phi_load=True,encoder_released_before_decode=True,full64_states_symbols_latents_qmaps_exact=True))
  else:assert sha(r['reuse_source'])==r['reuse_source_sha256']
  r.update(evaluation_origin='reused' if reused else 'new',evaluation_json_path=str(path),PSNR_mean_of_frame_PSNR=sum(float(f['PSNR']) for f in frames)/64)
  if 'model_parameters' not in r:
   size=load(P/'server_artifacts/initial_baselines'/f'{b}_s0_UVal0_q2.json')
   for k in ('model_parameters','model_parameter_bytes','probability_parameters'):r[k]=size[k]
   r['model_size_provenance']='same architecture/dtype measured by new step0 baseline parity check; legacy timing unchanged'
  rows.append(r);checks.append(dict(point=path.stem,status='PASS',origin=r['evaluation_origin'],stream_sha256=r['stream_sha256'],frames=64,transitions=63,global_RGB_PSNR_audited=True))
  est=r['estimated_P_main_latent_bits']+r['estimated_P_hyper_latent_bits']
  rate.append(dict(method=b,step=s,sequence=seq,QP=q,evaluation_origin=r['evaluation_origin'],estimated_y_bits=r['estimated_P_main_latent_bits'],estimated_z_bits=r['estimated_P_hyper_latent_bits'],estimated_P_y_z_bits=est,actual_P_payload_bytes=r['P_payload_bytes'],actual_P_payload_bits=r['actual_P_payload_bits'],actual_minus_estimated_bits=r['actual_P_payload_bits']-est,relative_gap_percent=100*(r['actual_P_payload_bits']/est-1),scope='P1-P63 only; excludes I and headers; actual y/z combined',real_total_bytes=r['real_total_bytes'],headers_bytes=r['header_bytes']))
 required={(b,s,seq,q) for b in METHODS for s in (0,100,300) for seq in (SEQUENCES if s==300 else SEQUENCES[:2]) for q in (2,7)}
 got={(r['branch'],r['step'],r['sequence'],r['external_qp']) for r in rows};assert required<=got and len(got)==len(rows)
 initial_checks=[]
 for method,ref in [('N1','N0'),('Q1','Q0')]:
  for seq in SEQUENCES[:2]:
   for q in (2,7):
    a=next(r for r in rows if (r['branch'],r['step'],r['sequence'],r['external_qp'])==(method,0,seq,q))
    b=next(r for r in rows if (r['branch'],r['step'],r['sequence'],r['external_qp'])==(ref,0,seq,q))
    assert a['stream_sha256']==b['stream_sha256'] and a['real_total_bytes']==b['real_total_bytes']
    assert a['source_frame_indices']==b['source_frame_indices']
    assert all(abs(a[k]-b[k])<1e-8 for k in METRICS)
    initial_checks.append(dict(method=method,reference=ref,sequence=seq,QP=q,stream_sha256=a['stream_sha256'],metrics_exact_within_1e_8=True))
 dump(P/'audits/all_initial_evaluation_identity.json',dict(status='PASS',records=initial_checks))
 for b in METHODS:
  for seq in SEQUENCES:assert len({r['external_qp'] for r in rows if r['branch']==b and r['step']==300 and r['sequence']==seq}-{2,7})<=3
 extra=load(P/'audits/reused_N0_extended_decode.json');assert extra['status']=='PASS' and extra['complete']==extra['planned']==12
 c=load(P/'cdf_consistency_audit.json');c.update(status='PASS',new_formal_evaluations=cdf,legacy_probability_network_frozen=True,legacy_extended_decode_audit='audits/reused_N0_extended_decode.json',note='Identical integer CDF hashes across checkpoints are allowed when quantized CDF entries do not change; fresh rebuilding still required.');dump(P/'cdf_consistency_audit.json',c)
 write(P/'evaluation.csv',rows);write(P/'estimated_vs_actual_P_rate.csv',rate)
 dump(P/'audits/evaluation_integrity.json',dict(status='PASS',required_points=len(required),required_complete=len(required & got),total_points=len(rows),new_points=sum(r['evaluation_origin']=='new' for r in rows),reused_points=sum(r['evaluation_origin']=='reused' for r in rows),records=checks,FID='not_computed'))
 return rows

def interpolate(rows,x,k):
 a=sorted(rows,key=lambda r:r['actual_bpp'])
 for r in a:
  if abs(x-r['actual_bpp'])<1e-15:return r[k]
 for l,r in zip(a,a[1:]):
  if l['actual_bpp']<=x<=r['actual_bpp']:
   return l[k]+(r[k]-l[k])*(x-l['actual_bpp'])/(r['actual_bpp']-l['actual_bpp'])
 raise ValueError('extrapolation prohibited')
def comparisons(rows):
 matched=[];dom=[];paired=[]
 for method,ref in CONTRASTS:
  label=f'{method}-{ref}'
  for seq in SEQUENCES:
   a=[r for r in rows if r['branch']==method and r['step']==300 and r['sequence']==seq];b=[r for r in rows if r['branch']==ref and r['step']==300 and r['sequence']==seq]
   for x in a:
    for y in b:
     flags={}
     for k in METRICS:
      good=x[k]>=y[k] if k=='PSNR' else x[k]<=y[k]
      flags[k]=x['actual_bpp']<=y['actual_bpp'] and good and (x['actual_bpp']<y['actual_bpp'] or x[k]!=y[k])
     joint=x['actual_bpp']<=y['actual_bpp'] and all(x[k]>=y[k] if k=='PSNR' else x[k]<=y[k] for k in METRICS) and (x['actual_bpp']<y['actual_bpp'] or any(x[k]!=y[k] for k in METRICS))
     dom.append(dict(comparison=label,sequence=seq,method_QP=x['external_qp'],reference_QP=y['external_qp'],method_actual_bpp=x['actual_bpp'],reference_actual_bpp=y['actual_bpp'],**{k+'_delta':x[k]-y[k] for k in METRICS},**{k+'_dominates':v for k,v in flags.items()},all_metrics_jointly_dominates=joint))
   lo=max(min(r['actual_bpp'] for r in a),min(r['actual_bpp'] for r in b));hi=min(max(r['actual_bpp'] for r in a),max(r['actual_bpp'] for r in b))
   if lo>hi:matched.append(dict(comparison=label,sequence=seq,status='no_observed_rate_overlap',overlap_low_bpp=lo,overlap_high_bpp=hi));continue
   for name,x in [('low',lo),('mid',(lo+hi)/2),('high',hi)]:
    out=dict(comparison=label,sequence=seq,status='interpolated_within_observed_overlap',position=name,actual_bpp=x,overlap_low_bpp=lo,overlap_high_bpp=hi,interpolation='linear_quality_in_actual_bpp; no extrapolation; not BD-rate')
    for k in METRICS:
     u,v=interpolate(a,x,k),interpolate(b,x,k);out.update({k+'_method':u,k+'_reference':v,k+'_delta':u-v})
     if k!='PSNR':out[k+'_relative_percent']=100*(u/v-1)
    matched.append(out)
  for x in [r for r in rows if r['branch']==method]:
   y=next((r for r in rows if r['branch']==ref and (r['sequence'],r['step'],r['external_qp'])==(x['sequence'],x['step'],x['external_qp'])),None)
   if y:paired.append(dict(comparison=label,step=x['step'],sequence=x['sequence'],QP=x['external_qp'],actual_bpp_method=x['actual_bpp'],actual_bpp_reference=y['actual_bpp'],actual_bpp_delta=x['actual_bpp']-y['actual_bpp'],**{k+'_delta':x[k]-y[k] for k in METRICS},comparison_type='same_external_QP; not matched_rate'))
 write(P/'matched_rate_comparisons.csv',matched);write(P/'observed_dominance.csv',dom);write(P/'paired_differences.csv',paired)
 summary=[]
 for method,ref in CONTRASTS:
  for scope,seqs in [('UVal',SEQUENCES[:2]),('descriptive',SEQUENCES[2:])]:
   rs=[r for r in matched if r['comparison']==f'{method}-{ref}' and r['sequence'] in seqs and r.get('position')=='mid']
   if rs:summary.append(dict(comparison=f'{method}-{ref}',scope=scope,sequences=len(rs),statistic='equal_sequence_mean_of_overlap_midpoint_differences',**{k+'_delta':sum(r[k+'_delta'] for r in rs)/len(rs) for k in METRICS}))
 write(P/'summary.csv',summary)
 return matched,dom,summary

def markdown_table(rows,keys):
 return '| '+' | '.join(keys)+' |\n| '+' | '.join('---' for k in keys)+' |\n'+'\n'.join('| '+' | '.join(f'{r.get(k, ""):.7g}' if isinstance(r.get(k),float) else str(r.get(k,'')) for k in keys)+' |' for r in rows)+'\n'
def report(rows,matched,dom,summary):
 lines=['# Quantization and probability adaptation: v3 300-update trend experiment','', 'Four logical conditions, one seed (20261023), 300 updates per condition. N0 reuses v1 A training and v2-verified evaluations; Q0 reuses v2 H11. N1 and Q1 are newly trained from the common original G/Bridge start, with original Phi and identity H for Q1. No spatial visual condition V is installed.','', '## Actual observed points and dominance','', 'All values below are actual persistent 64-frame RANS encodings with independent decoding. Arrows: LPIPS/DISTS/FloLPIPS lower is better; PSNR higher is better. QP is not a matched-rate comparison. Full step0/100/300 records, time, memory, bytes and provenance are in [evaluation.csv](evaluation.csv).','']
 step300=sorted([r for r in rows if r['step']==300],key=lambda r:(SEQUENCES.index(r['sequence']),METHODS.index(r['branch']),r['external_qp']))
 lines.append(markdown_table(step300,['branch','sequence','external_qp','actual_bpp','kbps_20fps','LPIPS','DISTS','FloLPIPS','PSNR','evaluation_origin']))
 lines+=['','Observed dominance compares only measured points: method rate ≤ reference rate, quality no worse, and at least one strict inequality. Counts are cross-QP pairs, not independent samples or statistical confidence. [All pairwise measured comparisons](observed_dominance.csv) include unfavorable differences.','']
 counts=[]
 for m,r in CONTRASTS:
  for seq in SEQUENCES:
   ds=[x for x in dom if x['comparison']==f'{m}-{r}' and x['sequence']==seq]
   counts.append(dict(comparison=f'{m}-{r}',sequence=seq,measured_pairs=len(ds),**{k:sum(x[k+'_dominates'] for x in ds) for k in METRICS},joint=sum(x['all_metrics_jointly_dominates'] for x in ds)))
 lines.append(markdown_table(counts,['comparison','sequence','measured_pairs',*METRICS,'joint']))
 lines+=['','## Fixed comparisons within actual rate overlap','', 'Linear interpolation uses quality against actual bpp, separately for each sequence and each fixed contrast. Only the observed overlap is used; there is no extrapolation, formal BD-rate, checkpoint selection by test quality, or interpretation of perceptual percentage changes as bitrate savings. Low/mid/high overlap coordinates are in [matched_rate_comparisons.csv](matched_rate_comparisons.csv). The following values are method minus reference at each overlap midpoint. PSNR differences are dB.','']
 for m,r in CONTRASTS:
  label=f'{m}-{r}';lines+=['### '+label,'']
  rs=[x for x in matched if x['comparison']==label and (x.get('position')=='mid' or x['status']=='no_observed_rate_overlap')]
  lines.append(markdown_table(rs,['sequence','status','actual_bpp','LPIPS_delta','DISTS_delta','FloLPIPS_delta','PSNR_delta']))
  validation=[x for x in rs if x['sequence'] in SEQUENCES[:2] and x.get('position')=='mid']
  if validation:
   directions=[]
   for metric in METRICS:
    favorable=sum(x[metric+'_delta']>0 if metric=='PSNR' else x[metric+'_delta']<0 for x in validation)
    unfavorable=sum(x[metric+'_delta']<0 if metric=='PSNR' else x[metric+'_delta']>0 for x in validation)
    directions.append(f'{metric}: improves on {favorable}/{len(validation)} validation sequences, worsens on {unfavorable}/{len(validation)}')
   lines.append('At the observed overlap midpoints, '+ '; '.join(directions)+'. These are descriptive signed differences at step300, without statistical significance or convergence claims.\n')
  if label=='N1-N0':lines.append('This comparison opens the same original probability modules without adding H. Original y_prior_fusion outputs, including its local quant_step, may adapt.\n')
  elif label=='Q1-Q0':lines.append('This comparison opens probability adaptation while retaining the corrected H11 quantization definition.\n')
  else:lines.append('This comparison matches G/Phi adaptation scope and tests the incremental presence of H. Benefits from probability adaptation alone do not establish a benefit from H.\n')
 lines+=['### Equal-sequence descriptive summaries','',markdown_table(summary,['comparison','scope','sequences','LPIPS_delta','DISTS_delta','FloLPIPS_delta','PSNR_delta']),'','## Engineering checks','', '- Initial N1/N0 and Q1/Q0 full 64-frame files, symbols, reconstruction and references match, including H identity initialization. Initial shared trainable tensors are verified from the indexed original sources.','- Both new branches commit exactly 300 optimizer updates, including the initial two smoke updates. Receipts, optimizer state and parameter hashes verify nonzero finite G/Phi-y/Phi-z updates and Q1 H updates; frozen parameters remain unchanged. No validation or test samples enter training.','- Every new evaluation reloads full Phi/G/H, regenerates integer CDFs and clears entropy caches. The encoder is released before a fresh decoder instance loads the same checkpoint. All 64 reference states and 63 P symbol/latent/quantization-map checks pass. [CDF audit](cdf_consistency_audit.json) records both sides. Equal CDF hashes after small parameter changes can occur due to integer CDF discretization; CDFs are rebuilt regardless.','- Twelve legacy N0 points lacked an explicit latent/qmap flag. Their original values are preserved, and a separate read-only old/new decoder audit verifies all 64 reconstructions/references and P latent/quantization maps. This does not count as new training or replace legacy metric measurements.','- [P-only rate accounting](estimated_vs_actual_P_rate.csv) excludes native I and headers from both P comparisons. Actual RANS y/z payload is combined; analytic estimated y/z bits are listed separately. Evaluation z likelihood retains the inherited native +1e-5 convention; differentiable training uses the inherited 1e-9 probability floor.','- PSNR is calculated from aggregate RGB MSE; the frame-PSNR mean is separately named. FloLPIPS uses 63 real adjacent transitions. FID: not_computed.','', '## Plots and fixed previews','', '![Training](figures/training.png)','![Multiplier statistics](figures/multiplier_statistics.png)']
 for seq in SEQUENCES:lines+=['',f'![Actual rate–quality {seq}](figures/actual_RD_{seq}.png)',f'![Fixed preview {seq}](figures/fixed_preview_{seq}.png)']
 lines+=['','## Limits and provenance','', 'This is a single-seed, fixed-300-update trend experiment, not a convergence study or a reproduction of a complete official training objective. Training uses the unchanged v2 U-Long plan (eight sources, two fixed 16-frame windows per source); the inherited window/QP coupling remains. The plan is not resampled. Inherited G/Bridge weights have historical UVG exposure, including Beauty from v6.40; ParkScene/Jockey checks are descriptive and cannot establish strict unseen-content generalization. UVal0/UVal1 are held out from this round’s optimizer updates. Multiple QPs of one sequence are not independent video samples.','', 'N0/Q0 evaluation time and memory figures are historical; new records are labeled new. Timing includes inherited instrumentation and rate estimation, and peak memory includes metric models, so these are descriptive measurements rather than a controlled throughput benchmark. Master Phi parameters and accumulated gradients are FP32 while differentiable casts preserve the v2 FP16 codec path. G forward precision is unchanged. Fresh original Phi/G starts are used for both new branches; Q1 is not initialized from H11 step300.','', 'Reused extra QP endpoints are explicitly labeled; any new supplement is selected using only actual rate bracketing, with at most three extra legal QPs per method/sequence. The fixed frame15, QP2, crop(y=200,x=400,size256) previews are not selected by quality. Raw videos, large weights, complete streams and caches remain on the server with SHA256 indices.','', '[Protocol](PROTOCOL.md) · [Exact Phi scope](PROBABILITY_PARAMETER_SCOPE.md) · [Reuse provenance](reuse_provenance.json) · [Integrity](final_integrity.json) · [Artifact index](artifact_index.json)']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n')

def artifact_index(index):
 files={}
 for bystep in index.values():
  for rec in bystep.values():files[rec['path']]={k:rec[k] for k in ('path','bytes','sha256')}
 for root in [P/'server_artifacts']:
  for f in root.rglob('*'):
   if f.is_file() and f.suffix in ('.pt','.bin','.npy','.npz'):files[str(f)]=record(f)
 for r in [load(f) for f in (P/'server_artifacts/evaluation').glob('*.json')]:files[r['stream_path']]=record(r['stream_path'])
 for path,h in load(P/'weight_manifest.json').items():files[path]=record(path);assert files[path]['sha256']==h['sha256']
 dump(P/'artifact_index.json',dict(status='PASS',large_files_server_only=True,records=list(files.values())))
 dump(P/'source_index.json',dict(dependency_sources=[dict(path=p,sha256=h,bytes=Path(p).stat().st_size) for p,h in load(P/'dependency_manifest.json').items()],source_v2_commit='3f9fae83a8be672c4c4999e09b7c4b08bf4a8fb4',v2_directory=str(V2)))

def main():
 assert load(P/'run_status.json')['status'] in ('COMPUTE_COMPLETE','COMPLETE','PUBLISHED')
 assert load(P/'preflight.json')['status']=='PASS' and load(P/'preflight.json')['signature']==signature()
 assert load(P/'audits/behavior_symbols.json')['status']=='PASS'
 for path,h in load(P/'dependency_manifest.json').items():assert sha(path)==h,('historical dependency changed',path)
 for name in ('data_manifest.json','train_plan.json','evaluation_sources.json'):assert sha(P/name)==sha(V2/name)
 index=checkpoint_audit();training_audit();rows=evaluation_audit(index);matched,dom,summary=comparisons(rows);report(rows,matched,dom,summary);artifact_index(index)
 dump(P/'effective_runtime_audit.json',dict(status='PASS',signature=signature(),INDEX_MAP=[0,1,0,2,0,2,0,2],executed_offsets=[0,2,0,1,0,1,0,1],legacy_four_element_description_in_effective_config='inherited documentary field; not used as a cyclic schedule',Phi_scope=['hyper_dec','temporal_prior_encoder','y_prior_fusion','y_spatial_prior','bit_estimator_z'],trainable_parameters_and_gradients='FP32',G_precision_unchanged=True,shared_train_compress_decompress_runtime=True))
 dump(P/'final_integrity.json',dict(status='PASS',signature=signature(),new_updates={'N0':0,'N1':300,'Q0':0,'Q1':300},reused_updates={'N0':300,'N1':0,'Q0':300,'Q1':0},fixed_evaluation_points='64/64',new_fixed_points='32/32',total_evaluation_points=len(rows),new_measurements=sum(r['evaluation_origin']=='new' for r in rows),reused_measurements=sum(r['evaluation_origin']=='reused' for r in rows),historical_dependencies_unchanged=True,all_new_Phi_CDF_independent_decode=True,training_integrity='audits/training_integrity.json',evaluation_integrity='audits/evaluation_integrity.json',checkpoint_integrity='audits/checkpoint_integrity.json',legacy_extended_decode='audits/reused_N0_extended_decode.json',publication_verified=False,remaining_compute_blockers=[]))
 print('FINALIZE_PASS',len(rows),flush=True)
if __name__=='__main__':main()
