"""Mechanical terminal checks and numeric reporting for the fixed continuation."""
from io32 import *
import collections,math,subprocess,traceback

def main():
 from evaluate32 import valid,receipt
 from report_math33 import pooled
 from compare33 import run,MAIN,DATASETS
 from migration33 import state_digest
 import torch
 assert load(ROOT/'audits/frozen_dependencies.json')['status']=='PASS' and load(ROOT/'audits/numerical_compatibility.json')['status']=='PASS'
 for n,h in load(ROOT/'audits/analysis_freeze.json')['hashes'].items():assert sha(ROOT/n)==h
 assert load(ROOT/'audits/migration_and_first_step.json')['status']=='PASS' and load(ROOT/'audits/comparison_arithmetic.json')['status']=='PASS'
 tasks=load(ROOT/'task_manifest.json')['tasks'];assert len(tasks)==4094 and all(valid(t) for t in tasks)
 receipts=[load(receipt(t)) for t in tasks];records=[];frame_rows=[];transition_rows=[];binary={};checked={}
 def checkpath(p,expected=None):
  if p not in checked:checked[p]=sha(p)
  if expected is not None:assert checked[p]==expected,(p,checked[p],expected)
  if Path(p).suffix in ('.pt','.pth','.ckpt','.npz','.npy','.bin'):binary[p]=dict(path=p,bytes=Path(p).stat().st_size,sha256=checked[p])
 for t,rr in zip(tasks,receipts):
  r=dict(rr['result'],dataset=t['dataset'],video=t['video'],method=t['method'],QP=t['qp'],adaptation_step=t.get('step',0));v=next(v for v in load(ROOT/'eval_manifest.json')[t['dataset']] if v['video_index']==t['video'])
  r.update(native_fps=v['rate_accounting_fps'],source_fps_metadata=v.get('source_fps_metadata'),kbps_20fps=r['bits_per_frame']*.02)
  assert r['frames']==64 and r['independent_decode_pass'] and r['state_sync_pass'] and r['bytes_consumed']==r['real_bytes']==Path(r['bitstream_path']).stat().st_size
  assert r['source_rgb_sha256']==v['rgb_sha256'] and r['source_frame_indices']==v['source_frame_indices'];assert abs(r['bpp']-r['real_bytes']*8/(64*1920*1080))<1e-15
  for k in ('LPIPS','DISTS','FloLPIPS','PSNR','bpp'):assert math.isfinite(float(r[k]))
  records.append(r);meta=dict(dataset=t['dataset'],video=t['video'],method=t['method'],QP=t['qp'],adaptation_step=t.get('step',0),sequence_actual_bpp=r['bpp'])
  f=read(r['frame_metrics_path']);tr=read(r['transitions_path']);assert len(f)==64 and len(tr)==63
  frame_rows.extend(dict(x,**meta) for x in f);transition_rows.extend(dict(x,**meta) for x in tr)
  for p,md in rr['artifact_metadata'].items():checkpath(p,md['sha256'])
 write(ROOT/'evaluation_points.csv',records);write(ROOT/'per_sequence_all_qp.csv',records);write(ROOT/'per_frame.csv',frame_rows);write(ROOT/'transitions.csv',transition_rows)
 write(ROOT/'evaluation/provenance.csv',[dict(task=r['task'],read_only_reuse=r.get('read_only_reuse',False),new_encode=r.get('fresh_encode',False),source_receipt=r.get('reuse_receipt'),source_receipt_sha256=r.get('reuse_receipt_sha256'),signature=r['signature']) for r in receipts])
 pools=pooled(records)
 for r in pools:
  expected=r['QP'] in (0,5,9) if r['method'].endswith('_1500') else True
  assert (r['status']=='available')==expected
  if expected:assert all(math.isfinite(float(r[k])) for k in ('FID','D_mu','D_cov')) and abs(r['FID']-r['D_mu']-r['D_cov'])<=1e-10
 for d in DATASETS:assert len({r['GT_pool_sha256'] for r in pools if r['dataset']==d and r['status']=='available'})==1
 write(ROOT/'pooled_fid.csv',pools);write(ROOT/'fid_decomposition_raw.csv',pools)
 summary,intervals=run(records,pools)
 # Same-QP raw quality alongside pooled components; explicitly not same-rate gains.
 allqp=[]
 for p in pools:
  if p['status']!='available':continue
  d=p['dataset'];base='mcljcv' if d=='mcljcv29' else d;vs=load(ROOT/'eval_manifest.json')[base];ids={v['video_index'] for v in vs if not(d=='mcljcv29' and 'videoSRC17' in v['name'])}
  rs=[r for r in records if r['dataset']==base and r['video'] in ids and r['method']==p['method'] and r['QP']==p['QP']];assert len(rs)==len(ids)
  allqp.append(dict(p,LPIPS=sum(r['LPIPS'] for r in rs)/len(rs),DISTS=sum(r['DISTS'] for r in rs)/len(rs),FloLPIPS=sum(r['FloLPIPS'] for r in rs)/len(rs),PSNR=sum(r['PSNR'] for r in rs)/len(rs),comparison_scope='same QP raw measured points, not same rate'))
 write(ROOT/'all_qp_summary.csv',allqp)
 high=[]
 for d in ('hevc_b','mcljcv','mcljcv29'):
  common=next(x for x in intervals if x['phase']=='main_2000' and x['dataset']==d and x['video']=='pooled' and x['metric']=='FID')
  for r in allqp:
   if r['dataset']!=d or r['method'] not in MAIN:continue
   lo=common['bpp_min'];hi=common['bpp_max'];threshold=math.exp(math.log(lo)+(math.log(hi)-math.log(lo))*2/3) if common['status']=='available' else None
   high.append(dict(r,common_interval_status=common['status'],common_bpp_min=lo,common_bpp_max=hi,upper_third_ln_bpp_threshold=threshold,in_common_high_rate_band=threshold is not None and threshold<=r['bpp']<=hi))
 write(ROOT/'high_rate_actual_points.csv',high)
 accounting={};training={};ema={};history=[];freeze_audits=[]
 for b in BRANCHES:
  bp=ROOT/'branches'/b;ti=load(bp/'training_integrity.json');assert ti['status']=='PASS' and ti['additional_updates']==1000 and ti['cumulative_adaptation_updates']==2000
  mig=load(bp/'migration_receipt.json');assert mig['status']=='PASS' and not mig['fresh_optimizer'] and mig['source_optimizer_internal_steps']==[1000]
  eq=load(bp/'migration_equivalence.json');assert eq['status']=='PASS' and eq['zero_optimizer_updates']
  calls=[json.loads(x) for x in (bp/'optimizer_calls.jsonl').read_text().splitlines()];physical=sum(x['event']=='STARTED' for x in calls);assert physical>=1000
  accounting[b]=dict(physical_optimizer_calls=physical,committed_additional_updates=1000,cumulative_adaptation_updates=2000,discarded_or_replayed_calls=physical-1000,parent_training_checkpoint=load(ROOT/'config.json')['branch_sources'][b]['path'])
  training[b]=dict(additional=1000,cumulative=2000)
  rows=[load(bp/'receipts'/f'{i}.json') for i in range(1001,2001)];counts=collections.Counter((r['domain'],r['external_qp']) for r in rows);assert len(counts)==20 and set(counts.values())=={50}
  for i,r in enumerate(rows,1001):
   assert r['adaptation_step']==i and r['additional_step']==i-1000 and r['source_plan_step']==i-1000 and r['optimizer_internal_step']==i
   assert r['finite'] and r['gradient_formula_error']<=1e-6 and r['postclip_scaling_error']<=1e-6 and r['global_postclip_norm']<=1.00001
   assert r['actual_qps']==load(ROOT/'plans/shared.json')['updates'][i-1001]['actual_qps']
   if b!=BRANCHES[0]:assert r['total_EMA_commits']==i and r['additional_EMA_commits']==i-1000 and all(x['finite'] and x['norm']>0 for x in r['additional_gradient_modules'].values())
   else:assert r['total_EMA_commits'] is None and r['additional_EMA_commits']==0
   history.append(dict(branch=b,step=i,additional_step=i-1000,stratum=r['stratum'],EMA_counter=r['EMA_commit_count'],total_EMA_commits=r['total_EMA_commits'],additional_EMA_commits=r['additional_EMA_commits'],C1_loss=r['total_loss'],additional_loss=r['additional_loss'],complete_objective_loss=r['complete_objective_loss'],gradient_merge_error=r['gradient_formula_error'],postclip_scaling_error=r['postclip_scaling_error']))
  s=torch.load(bp/'checkpoints/latest.pt',map_location='cpu',weights_only=True);assert s['signature']==signature('train') and s['adaptation_step']==2000 and s['additional_step']==1000 and s['replay_cursor']==1000
  assert {int(x['step']) for x in s['optimizer']['state'].values()}=={2000};assert set(s['visits'].values())=={100}
  for g,lr in zip(s['optimizer']['param_groups'],(6e-6,1e-6)):assert g['lr']==lr and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==1e-4
  if b==BRANCHES[0]:assert s['EMA'] is None;ema[b]=dict(status='N/A',additional_commits=0,total_commits=None)
  else:assert set(v['counter'] for v in s['EMA'].values())=={100};ema[b]=dict(status='PASS',additional_commits=1000,total_commits=2000,counters={k:v['counter'] for k,v in s['EMA'].items()})
  freeze_audits.append(dict(branch=b,frozen=s['frozen_hashes'],source_frozen=load(ROOT/'config.json')['branch_sources'][b]['frozen_hashes']));assert freeze_audits[-1]['frozen']==freeze_audits[-1]['source_frozen'];del s
  ix=load(bp/'checkpoint_index.json');assert set(ix)=={'1000','1001','1250','1500','1750','2000'}
  for e in ix.values():checkpath(e['path'],e['sha256']);checkpath(e['inference']['path'],e['inference']['sha256'])
 geometry=[]
 for b in BRANCHES:
  for ix in (0,165,253,477,720):
   rr=load(ROOT/'branches'/b/'receipts'/f'{1001+ix}.json');assert rr['full_frame_status']=='PASS_NATIVE_FULL_CHAIN'
   geometry.append(dict(branch=b,source_plan_index=ix,global_step=1001+ix,source_RGB_hashes=rr['full_RGB_sha256'],crop_RGB_hashes=rr['cropped_RGB_sha256'],replay=rr['replay'],new_updates_already_in_budget=True))
 dump(ROOT/'geometry_audit.json',dict(status='PASS',inherited_geometry=load(ROOT/'audits/inherited_geometry_audit.json'),new_geometry_checks=geometry,native_sizes_not_resized=True))
 dump(ROOT/'data_isolation_audit.json',load(ROOT/'audits/data_isolation.json'))
 write(ROOT/'training_EMA_history.csv',history)
 # Read-only training reference, calibrations and all historical numerical dependencies.
 for p,md in load(ROOT/'audits/historical_before.json').items():checkpath(p,md['sha256'])
 for n,h in load(ROOT/'audits/implementation_freeze.json')['files'].items():assert sha(ROOT/n)==h
 dump(ROOT/'restore_integrity.json',dict(status='PASS',optimizer_RNG_buffers_and_lineage_verified=True,accounting=accounting))
 dump(ROOT/'gradient_integrity.json',dict(status='PASS',committed_steps=3000,preclip_merge_error_correctly_recorded=True,postclip_scaling_verified=True,first_steps={b:load(ROOT/'audits'/f'first_committed_step_{b}.json') for b in BRANCHES},frozen=freeze_audits))
 dump(ROOT/'EMA_integrity.json',dict(status='PASS',branches=ema,GT_reference_unchanged=True,additional_prefill=0,additional_calibration=0))
 dump(ROOT/'replay_integrity.json',dict(status='PASS',C_D_all_new_updates_streaming_full_chain_verified=True,normal_and_special_native_geometries_preserved=True,reference_tolerances=dict(relative_L2=1e-4,maxabs='1e-4+1e-4*maxabs(reference)'),B_auxiliary_is_diagnostic_only=True))
 for p in ROOT.rglob('*'):
  if p.is_file() and 'publication' not in p.relative_to(ROOT).parts and p.suffix in ('.pt','.pth','.ckpt','.npz','.npy','.bin'):checkpath(str(p))
 dump(ROOT/'binary_artifacts_index.json',dict(server_only=True,artifacts=list(binary.values())))
 reused=sum(bool(r.get('read_only_reuse')) for r in receipts);plannednew=sum(t['stage']=='new' and bool(r.get('fresh_encode')) for t,r in zip(tasks,receipts));fallback=sum(t['stage']!='new' and bool(r.get('fresh_encode')) for t,r in zip(tasks,receipts));assert plannednew==1794 and reused+fallback==2300
 integ=dict(status='PASS',calculation='COMPLETE',publication='PENDING',training=training,optimizer_call_accounting=accounting,historical_reference_prefill=dict(completed=1000,expected=1000,reused_read_only=True),historical_calibration=dict(completed=40,expected=40,reused_read_only=True),new_prefill=0,new_calibration=0,evaluation=dict(completed=4094,expected=4094,historical_reused=reused,planned_new_encodes=1794,historical_extra_reencoded=fallback,diagnostic1500=414,main2000=1380),restore='PASS',gradients='PASS',frozen='PASS',EMA=ema,historical_files_unchanged=True,real_RANS=True,independent_decode=True,all_headers_and_I_bytes_included=True,formal_FID='fixed dataset pooled official Inception2048,float64,ddof1',no_test_selection=True,discriminator_optimizer_updates=0,unresolved=[])
 report=['# v6.33 fixed paired continuation','', 'Training/evaluation counters and numeric tables; no model selection or research judgement.','',f'Additional/cumulative updates: {training}.',f'Physical/replay accounting: {accounting}.',f'Historical read-only points{reused}; historical fallback encodes{fallback}; planned new encodes1794; logical4094/4094.','', 'Historical GT reference1000/1000 and calibration40/40 are verified read-only aliases; additional prefill0 and calibration0. Exact source checkpoints,full optimizer moments/steps,RNG,buffers and C/D EMA states were migrated. The same frozen1000 windows are exposed a second time; zero new source videos/windows. B EMA is N/A. C/D each add1000 EMA commits and finish2000 with100/stratum. Missing historical B auxiliary views are neither retrofilled nor relabelled.','', '## Fixed protocol','', 'See config.json,effective_protocol.json,plans/continuation.json,source_hashes.json,migration receipts and data isolation audit. Native auxiliary geometry,causal DPB,actual QPs,FP32 training/FP16 deployment and4096 restoration are inherited. The composition diagnostic is recorded before in-place clipping; clipped scaling is separately verified. All primary endpoints are fixed step2000; step1500 is a separate three-QP diagnostic.','', '## Common-rate numerical comparisons','', 'Main: eight-model common measured intervals,100 uniform ln(bpp) points,PCHIP,no extrapolation. Paired metrics use video-equal weights; pooled FID uses dataset-mean actual bpp. Relative%=100*(dataset method mean-dataset reference mean)/dataset reference mean,not BD-rate or mean of video percentages. D_mu/D_cov component diagnostics use the same linear ln(bpp) weights for closure and are labelled separately from primary PCHIP FID. Unavailable common intervals are retained; fallback tables are explicitly pairwise. Difference-of-differences is a raw metric contrast on the same eight-model grid,without a percentage denominator.','', '|Dataset|Band|Method|Reference|Metric|Method mean|Reference mean|Delta|Relative %|','|---|---|---|---|---|---:|---:|---:|---:|']
 for r in summary:
  if r['phase']!='main_2000' or r['metric'] not in ('LPIPS','DISTS','FloLPIPS','FID') or r['contrast']!='paired_difference':continue
  vs=[f'{r[k]:.9g}' if r.get(k) is not None else 'unavailable' for k in ('method_mean','reference_mean','delta','relative_change_percent')];report.append('|'+ '|'.join([r['dataset'],r['rate_band'],r['method'],r['reference'],r['metric'],*vs])+'|')
 report+=['','## Files','', 'evaluation_points.csv/per_sequence_all_qp.csv and evaluation/provenance.csv contain every logical point and source. per_frame.csv/transitions.csv retain64frames/63transitions. pooled_fid.csv/fid_decomposition_raw.csv contain fixed-pool FID,mean/covariance terms and GT hashes. equal_rate_summary.csv/equal_rate_per_sequence.csv/common_rate_grid.csv/common_intervals.csv retain means,intervals,inputs and interpolation labels. difference_of_differences.csv and trajectory_3QP.csv separate contrasts and matched three-QP trajectories. high_rate_actual_points.csv lists HEVC-B/MCL measured points with the predefined upper-third common ln(bpp) threshold; same QP is not claimed to be same rate. rate_sign_ranges.csv records interpolated-grid sign runs,not extra measured encodes.','', 'Binary artifacts remain server-only,with path/size/SHA in binary_artifacts_index.json. final_integrity.json and publication_status.json separate execution and remote publication.']
 textfile(ROOT/'REPORT.md','\n'.join(report)+'\n');textfile(ROOT/'RESULTS_SUMMARY.md','\n'.join(report[:11])+'\n\nSee REPORT.md and equal_rate_summary.csv for the prescribed numeric comparisons.\n')
 subprocess.run(['/data1/anaconda3_new/anaconda_program/bin/python','-B',str(ROOT/'plot33.py')],cwd=REPO,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),check=True)
 for d in DATASETS:
  for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
   assert (ROOT/'results'/d/f'Rate_{metric}.png').is_file()
 dump(ROOT/'computation_integrity.json',integ);dump(ROOT/'final_integrity.json',dict(integ,status='PENDING_PUBLICATION'))
 print('COMPUTATION AND REPORT PASS',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:error(dict(stage='completion_report33',error=traceback.format_exc()));raise
