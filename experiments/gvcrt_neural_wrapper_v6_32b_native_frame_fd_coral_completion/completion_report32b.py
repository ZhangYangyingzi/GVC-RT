"""Complete six-method raw aggregation and terminal audits; no model selection."""
from io32 import *
import collections,math,subprocess,traceback

def pairwise_fallback(records,pools):
 import numpy as np
 from scipy.interpolate import PchipInterpolator
 out=[]
 methods=['original','B1_initial']+[f'{b}_1000' for b in BRANCHES]
 pairs=[(methods[4],methods[3]),(methods[5],methods[3]),(methods[4],methods[5]),(methods[4],methods[2]),(methods[5],methods[2])]+[(m,r) for m in methods[2:] for r in methods[:2]]
 def curve(curves,d,v,metric):
  xs={m:np.array([r['bpp'] for r in curves[m]],float) for m in methods}
  common_lo=max(x.min() for x in xs.values());common_hi=min(x.max() for x in xs.values())
  if common_hi>common_lo:return
  for m,r in pairs:
   lo=max(xs[m].min(),xs[r].min());hi=min(xs[m].max(),xs[r].max());row=dict(dataset=d,video=v,metric=metric,method=m,reference=r,interval_scope='pairwise fallback; NOT six-method common mean',bpp_min=float(lo),bpp_max=float(hi))
   if hi<=lo:row.update(status='unavailable',reason='no positive pairwise common interval')
   else:
    grid=np.linspace(np.log(lo),np.log(hi),100);ys={}
    for k in (m,r):
     x=xs[k];y=np.array([z[metric] for z in curves[k]],float);o=np.argsort(x);assert len(set(x))==len(x);ys[k]=PchipInterpolator(np.log(x[o]),y[o],extrapolate=False)(grid)
    a=float(ys[m].mean());b=float(ys[r].mean());row.update(status='available',method_mean=a,reference_mean=b,delta=a-b,relative_change_percent=100*(a-b)/b if b else None)
   out.append(row)
 for d,vs in load(ROOT/'eval_manifest.json').items():
  for v in vs:
   curves={m:[r for r in records if r['dataset']==d and r['video']==v['video_index'] and r['method']==m] for m in methods}
   for metric in ('LPIPS','DISTS','FloLPIPS'):curve(curves,d,v['name'],metric)
 for d in ('ulong','uvg_holdout','uvg_validation','hevc_b','mcljcv','mcljcv29'):
  curves={m:[r for r in pools if r['dataset']==d and r['method']==m and r['status']=='available'] for m in methods}
  for metric in ('FID','D_mu','D_cov'):curve(curves,d,'pooled',metric)
 return out

def main():
 from evaluate32 import valid,task_id,receipt
 from report32 import pooled,equal_rate,fields
 recording=load(ROOT/'audits/gradient_recording_integrity.json');assert recording['status']=='PASS' and recording['steps_checked']==2000
 corrected={(r['branch'],int(r['step'])):r for r in read(ROOT/'audits/gradient_recording_corrected_checks.csv')};assert len(corrected)==2000
 tasks=load(ROOT/'task_manifest.json')['tasks'];assert len(tasks)==3312 and all(valid(t) for t in tasks)
 good=[load(receipt(t)) for t in tasks];records=[];perframe=[];trans=[];binary={};checked={}
 for t,rr in zip(tasks,good):
  v=next(v for v in load(ROOT/'eval_manifest.json')[t['dataset']] if v['video_index']==t['video'])
  r=dict(rr['result'],video=t['video'],QP=t['qp'],method=t['method'],adaptation_step=t.get('step',0),source_fps_metadata=v.get('source_fps_metadata',dict(native_rate_accounting_fps=v['rate_accounting_fps'],source_format=v.get('pixel_format'))));r.update(kbps_20fps=r['bits_per_frame']*.02);records.append(r)
  meta=dict(dataset=t['dataset'],video=t['video'],method=t['method'],QP=t['qp'],bpp=r['bpp'],adaptation_step=t.get('step',0))
  for row in read(r['frame_metrics_path']):perframe.append(dict(row,**meta))
  for row in read(r['transitions_path']):trans.append(dict(row,**meta))
  for path,md in rr['artifact_metadata'].items():
   if path not in checked:checked[path]=sha(path)
   assert checked[path]==md['sha256']
   if Path(path).suffix in ('.pt','.bin','.npz','.npy'):binary[path]=dict(path=path,bytes=Path(path).stat().st_size,sha256=checked[path])
 keys=['dataset','video','sequence','method','QP','adaptation_step','real_bytes','bpp','bits_per_frame','native_fps','source_fps_metadata','rate_accounting_fps','kbps','kbps_20fps','LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM','source_rgb_sha256','source_frame_indices','actual_qps','reconstruction_sha256','bitstream_path','bitstream_sha256','feature_path','feature_sha256']
 write(ROOT/'evaluation_points.csv',fields(records,keys));write(ROOT/'per_sequence_all_qp.csv',fields(records,keys));write(ROOT/'per_frame.csv',perframe);write(ROOT/'transitions.csv',trans)
 write(ROOT/'evaluation/provenance.csv',[dict(task=rr['task'],read_only_reuse=rr.get('read_only_reuse',False),source_receipt=rr.get('reuse_receipt'),source_sha256=rr.get('reuse_receipt_sha256'),signature=rr['signature'],historical_signature=rr.get('historical_signature'),new_encode=not rr.get('read_only_reuse',False) and rr.get('fresh_encode',False)) for rr in good])
 pools=pooled(records)
 for r in pools:
  expected=r['QP'] in (0,5,9) if r['method'].endswith('_250') else True
  assert (r['status']=='available')==expected,(r['dataset'],r['method'],r['QP'],r['status'])
  if expected:assert all(math.isfinite(float(r[k])) for k in ('FID','D_mu','D_cov')) and abs(r['FID']-r['D_mu']-r['D_cov'])<=1e-10
 feature_pools=[]
 for d in ('ulong','uvg_holdout','uvg_validation','hevc_b','mcljcv','mcljcv29'):
  pr=[r for r in pools if r['dataset']==d and r['status']=='available'];hashes={r['GT_pool_sha256'] for r in pr};assert len(hashes)==1,(d,hashes)
  feature_pools.append(dict(dataset=d,GT_pool_sha256=next(iter(hashes)),sample_count=pr[0]['sample_count'],video_count=pr[0]['video_count'],ddof=1,all_methods_and_QPs_same_GT_pool=True,status='PASS'))
 dump(ROOT/'feature_pools_integrity.json',dict(status='PASS',datasets=feature_pools,official_feature_weight_sha256=load(ROOT/'protocol.json')['FID']['weight_sha256']))
 write(ROOT/'pooled_fid.csv',pools);write(ROOT/'fid_decomposition_raw.csv',pools)
 mainrows,grid=equal_rate(records,pools);early,eg=equal_rate(records,pools,True);write(ROOT/'equal_rate_main.csv',mainrows);write(ROOT/'equal_rate_early_diagnostic.csv',early);write(ROOT/'common_rate_grid.csv',grid or [dict(status='unavailable')]);write(ROOT/'early_common_rate_grid.csv',eg or [dict(status='unavailable')]);fallback=pairwise_fallback(records,pools);write(ROOT/'pairwise_fallback.csv',fallback or [dict(status='not_needed',reason='all six-model intervals exist')])
 summaries=[]
 for phase,rs in [('1000 main',mainrows),('250 diagnostic',early)]:
  groups=collections.defaultdict(list)
  for r in rs:groups[(r['dataset'],r['metric'],r['method'],r['reference'])].append(r)
  # MCL29 derives from the identical encoded points; paired metrics are reweighted over29.
  for key,rs0 in list(groups.items()):
   if key[0]=='mcljcv' and rs0[0]['video']!='pooled':groups[('mcljcv29',*key[1:])]=[r for r in rs0 if 'videoSRC17' not in r['video']]
  for (d,metric,m,ref),rr in groups.items():
   av=[r for r in rr if r['status']=='available'];row=dict(phase=phase,dataset=d,metric=metric,method=m,reference=ref,expected_videos=len(rr),completed_videos=len(av),interval_scope='six-model common interval',delta_definition='method-reference',relative_definition='100*(dataset_method_mean-dataset_reference_mean)/dataset_reference_mean',aggregation='pooled dataset FID' if rr[0]['video']=='pooled' else 'equal video weights')
   if len(av)!=len(rr):row.update(status='unavailable',reason='six-method common interval unavailable for one or more required videos')
   else:
    a=sum(r['method_mean'] for r in rr)/len(rr);b=sum(r['reference_mean'] for r in rr)/len(rr);row.update(status='available',method_mean=a,reference_mean=b,delta=a-b,relative_change_percent=100*(a-b)/b if b else None)
   summaries.append(row)
 write(ROOT/'equal_rate_summary.csv',summaries)
 training={};accounting={};ema={};ema_history=[];loss_history=[]
 for branch in BRANCHES:
  bp=ROOT/'branches'/branch;hist=OLD32/'branches'/branch if branch in BRANCHES[:2] else bp;integ=load(bp/'training_integrity.json');training[branch]=integ['updates'];assert training[branch]==1000
  calls=[json.loads(x) for x in (hist/'optimizer_calls.jsonl').read_text().splitlines()];started=sum(r['event']=='STARTED' for r in calls);accounting[branch]=dict(physical_optimizer_calls=started,committed_updates=1000,discarded_or_replayed_calls=started-1000,execution_origin=str(hist));assert started>=1000
  if branch in BRANCHES[2:]:
   native_checks=[]
   for idx in (165,253,477,720):
    nr=load(bp/'receipts'/f'{idx+1:04d}.json');assert nr['finite'] and all(v['finite'] and v['norm']>0 for v in nr['additional_gradient_modules'].values());native_checks.append(dict(plan_index=idx,receipt=str(bp/'receipts'/f'{idx+1:04d}.json'),gradient_modules=nr['additional_gradient_modules'],replay=nr['replay']))
   dump(ROOT/'audits'/f'native_geometry_gradients_{branch}.json',dict(status='PASS',checks=native_checks,committed_steps_already_in_budget=True))
   assert integ['EMA_commits']==1000 and set(integ['EMA_counters'].values())=={50};ema[branch]=dict(status='PASS',commits=1000,counters=integ['EMA_counters'])
   for i in range(1,1001):
    r=load(bp/'receipts'/f'{i:04d}.json');assert r['total_EMA_commits']==i and corrected[(branch,i)]['receipt_sha256']==sha(bp/'receipts'/f'{i:04d}.json') and all(float(corrected[(branch,i)][k])<=1e-6 for k in ('preclip_norm_composition_relative_error','clip_scaling_relative_error','legacy_alias_formula_residual')) and r['finite'] and r['global_postclip_norm']<=1.00001;assert all(v['finite'] and v['norm']>0 for v in r['additional_gradient_modules'].values())
   # Recovery checkpoint is parsed and optimizer/EMA/cursor agreement checked independently.
   import torch
   s=torch.load(bp/'checkpoints/latest.pt',map_location='cpu',weights_only=True);assert s['signature']==signature('train') and s['calibration_sha256']==sha(ROOT/'calibration/final.json') and s['seed_index_sha256']==sha(ROOT/'reference_stats/index.json');assert s['adaptation_step']==1000 and s['optimizer']['state'] and sum(v['counter'] for v in s['EMA'].values())==1000 and s['replay_cursor']==1000;del s
  for i in range(1,1001):
   r=load(hist/'receipts'/f'{i:04d}.json');ema_history.append(dict(branch=branch,step=i,execution_origin=str(hist),stratum=r['stratum'],stratum_visits=r['stratum_visits'],EMA_counter=r['EMA_commit_count'],total_EMA_commits=r.get('total_EMA_commits',0),full_frame_status=r['full_frame_status'],feature_path=r.get('feature_path'),feature_sha256=r.get('feature_sha256'),historical_read_only=branch in BRANCHES[:2]));loss_history.append(dict(branch=branch,step=i,domain=r['domain'],QP=r['external_qp'],C1_loss=r['total_loss'],additional_loss=r.get('additional_loss'),complete_objective_loss=r.get('complete_objective_loss',r['total_loss']),C1_gradient_norm=r.get('C1_full_gradient_norm',r.get('C1_gradient_norm')),additional_gradient_norm=r.get('additional_gradient_norm'),preclip=r['global_preclip_norm'],clip=r['clip_coefficient'],learning_rates=r['learning_rates'],seconds=r['seconds'],peak_memory_MiB=r['peak_memory_MiB']))
  for item in load(bp/'checkpoint_index.json').values():
   for k in ('path',):
    path=item[k];assert sha(path)==item['sha256'];binary[path]=dict(path=path,bytes=Path(path).stat().st_size,sha256=item['sha256'])
   e=item['inference'];assert sha(e['path'])==e['sha256'];binary[e['path']]=dict(path=e['path'],bytes=Path(e['path']).stat().st_size,sha256=e['sha256'])
 write(ROOT/'training_EMA_history.csv',ema_history);write(ROOT/'loss_update_summary.csv',loss_history)
 before=load(ROOT/'audits/historical_before.json');bad=[]
 for path,md in before.items():
  if sha(path)!=md['sha256']:bad.append(path)
 assert not bad,bad
 for path,h in load(ROOT/'source_hashes.json')['hashes'].items():assert sha(path)==h
 pre=load(ROOT/'prefill_integrity.json');cal=load(ROOT/'calibration/final.json');assert pre['completed']==1000 and cal['completed_windows']==40
 reused=sum(bool(r.get('read_only_reuse')) for r in good);fresh=sum(not r.get('read_only_reuse',False) and bool(r.get('fresh_encode')) for r in good)
 assert fresh+reused==3312
 grad=dict(status='PASS',gradient_recording_audit=recording,calibration_real_network_windows=40,calibration_no_optimizer_updates=True,C_D_first_committed_steps=[str(ROOT/'audits'/f'first_committed_step_{b}.json') for b in BRANCHES[2:]],RGB_bridge_generator_gradients_finite_nonzero=True,distribution_weights_fixed=cal,normalization='moments contain15 mean; feature VJP4096 once; restoreFP32 once; weighted sum then single clip and AdamW',EMA=ema,full_chain_replay_tolerance=dict(relative_L2=1e-4,maxabs='1e-4+1e-4*maxabs(reference)'),historical_C1_crop_AST_identical=True,frozen_verified=True)
 dump(ROOT/'gradient_integrity.json',grad);dump(ROOT/'forward_replay_integrity.json',dict(status='PASS',calibration_windows=40,geometry=load(ROOT/'geometry_audit.json'),training_each_update_verified=True,toy=load(ROOT/'audits/distribution_math.json')));dump(ROOT/'restore_integrity.json',dict(status='PASS',final_model_optimizer_EMA_cursor_consistent=True,physical_call_accounting=accounting))
 
 for p in ROOT.rglob('*'):
  if p.is_file() and 'publication' not in p.relative_to(ROOT).parts and p.suffix in ('.pt','.npz','.npy','.bin'):
   if str(p) not in binary:binary[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
 dump(ROOT/'binary_artifacts_index.json',dict(artifacts=list(binary.values()),server_only=True));dump(ROOT/'source_frozen_integrity.json',dict(status='PASS',historical_files_unchanged=True,dependencies_checked=len(load(ROOT/'source_hashes.json')['hashes']),evaluation_artifact_hashes_verified=len(checked)))
 integrity=dict(status='PASS',calculation='COMPLETE',publication='PENDING',training=training,optimizer_call_accounting=accounting,prefill=dict(completed=1000,expected=1000,strata=20),calibration=dict(completed=40,expected=40),evaluation=dict(completed=3312,expected=3312,historical_reused=reused,newly_encoded=fresh,C_D_adaptation_encodes=1196,all_adaptation_logical_points=2392),geometry='PASS',gradients='PASS',EMA=ema,frozen='PASS',real_RANS=True,independent_decode=True,all_headers_and_I_bytes_included=True,formal_evaluation_ddof=1,training_statistics_ddof=0,common_rate='six models100 ln(bpp) PCHIP no extrapolation; explicit pairwise fallback when unavailable',known_historical_AB_auxiliary_missing_per_branch=4,known_historical_AB_EMA_commits=0,known_auxiliary_missing_does_not_block_main_experiment=True,discriminator_optimizer_updates=0,unresolved=[])
 # Terminal computation PASS is committed only after all reports and required plots succeed.
 report=['# v6.32b execution and numerical record','', 'Computation, training, real evaluation and numerical audits: PASS. Publication is tracked independently in publication_status.json and final_integrity.json.','',f'Committed updates: {training}. Physical/replay accounting: {accounting}.',f'Logical evaluation3312/3312; historical read-only reuse{reused}; newly encoded{fresh}. Shared B1 prefill1000/1000, calibration40/40, twenty complete strata.','', '## Protocol revision and historical provenance','', 'See PROTOCOL_REVISION.md, protocol_revision.json and effective_protocol.json for explicit scoping of inherited, unused metadata fields; frozen runtime hashes and actual raw source bytes remain preserved.', 'Only the auxiliary training full-frame view changes to each source native valid H×W. The four plan indices165/253/477/720 retain original source, hashes, frame indices and crop; no resize/substitution. Formal testing remains1920×1080/64frames. Official alignment helper pads each native view and unpads before Inception. Ordinary1080p forward equivalence is recorded in geometry_audit.json.','', 'A/B are historical read-only1000-step C1 runs, not v6.32b reruns. Each retains four missing auxiliary full views and zero diagnostic EMA commits. These known historical auxiliary gaps are not counted as completed diagnostics and do not remove formal test points. C/D each start from the same original v6.22 B1, not B_1000, and receive1000 new AdamW updates.','',f'Fixed training-only coefficients: lambda_FD={cal["lambda_FD"]:.17g}; alpha_mu={cal["alpha_mu"]:.17g}; alpha_cov={cal["alpha_cov"]:.17g}. No test-based coefficient or checkpoint selection.','', 'Training statistics: native full-frame official Inception2048, float64/ddof0, twenty independent EMA strata, beta.95, fixed GT-only epsilon. Formal FID: fixed dataset pooled features, float64/ddof1, no training regularizer or EMA. MCL29 derives from the same encoded MCL30 sources. Correlated frames and repeated plan occurrences are not independent samples.','', '## Gradient recording correction', '', 'The original gradient_formula_error diagnostic was evaluated after clip_grad_norm_ mutated the aliased gr/p.grad tensors. It measures clipping, not preclip gradient composition. All raw logs, training code and checkpoints are preserved. audits/gradient_recording_integrity.json validates all2000 norm-composition/clip identities and includes read-only real-network first-step vector replays for C and D with zero optimizer/EMA updates. The actual merge, single clip and optimizer operations are unchanged.', '', '## Main1000-step common-rate differences','', 'All deltas are method−reference; relative changes are100×(dataset method mean−dataset reference mean)/dataset reference mean, not BD-rate. Six-method common measured bpp intervals are used. Negative perceptual-metric deltas are lower numerical values; no branch was selected using these results.','', '|Dataset|Method|Reference|Metric|Method mean|Reference mean|Delta|Relative %|','|---|---|---|---|---:|---:|---:|---:|']
 for r in summaries:
  if r['phase']!='1000 main' or r['metric'] not in ('LPIPS','DISTS','FloLPIPS','FID'):continue
  vals=[f'{r[k]:.9g}' if k in r and r[k] is not None else 'unavailable' for k in ('method_mean','reference_mean','delta','relative_change_percent')];report.append('|'+ '|'.join([r['dataset'],r['method'],r['reference'],r['metric'],*vals])+'|')
 report+=['','## Files and checks','', 'evaluation_points.csv/per_sequence_all_qp.csv: actual encoded point records and provenance. per_frame.csv/transitions.csv: all64frame/63transition values. pooled_fid.csv and fid_decomposition_raw.csv: fixed-pool FID and mean/covariance terms. equal_rate_main.csv and equal_rate_summary.csv: fixed1000-step comparisons. equal_rate_early_diagnostic.csv: separate250-step QP0/5/9 diagnostic. common_rate_grid.csv records actual intervals/interpolated values. pairwise_fallback.csv is explicitly labelled and never presented as a six-model mean.','', 'gradient_integrity.json, forward_replay_integrity.json, restore_integrity.json, source_frozen_integrity.json and final_integrity.json record real-chain, freezing, transaction and publication checks. Large server-only artifact paths and SHA256 values are in binary_artifacts_index.json.']
 textfile(ROOT/'REPORT.md','\n'.join(report)+'\n');textfile(ROOT/'RESULTS_SUMMARY.md','\n'.join(report[:16])+'\n\nSee equal_rate_summary.csv and REPORT.md for all fixed comparisons.\n')
 subprocess.run(['/data1/anaconda3_new/anaconda_program/bin/python','-B',str(ROOT/'plot32.py')],cwd=REPO,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),check=True)

 for d in ('ulong','uvg_holdout','uvg_validation','hevc_b','mcljcv','mcljcv29'):
  for phase in ('main','early_diagnostic'):
   for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
    p=ROOT/'results'/d/phase/f'Rate_{metric}.png';assert p.is_file() and p.stat().st_size>0
 dump(ROOT/'computation_integrity.json',integrity);dump(ROOT/'final_integrity.json',dict(integrity,status='PENDING_PUBLICATION'))
 print('COMPUTATION AND REPORT PASS',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:error(dict(stage='completion_report',error=traceback.format_exc()));raise
