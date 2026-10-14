"""Mechanical raw tables and explicit unavailable comparisons; no research interpretation."""
from io32 import *
import math,collections,subprocess
def fields(rows,keys):return [{k:r.get(k) for k in keys} for r in rows]
def same_qp_summary(records,pools):
 """Fixed QP5 raw diagnostics, with measured rate differences stated explicitly."""
 rows=[]
 for step in (250,1000):
  methods=['original','B1_initial']+[f'{b}_{step}' for b in BRANCHES]
  for d in ('ulong','uvg_holdout','uvg_validation','hevc_b','mcljcv','mcljcv29'):
   base='mcljcv' if d=='mcljcv29' else d
   vs=load(ROOT/'eval_manifest.json')[base]
   if d=='mcljcv29':vs=[v for v in vs if 'videoSRC17' not in v['name']]
   ids={v['video_index'] for v in vs}
   for m in methods:
    rr=[r for r in records if r['dataset']==base and r['method']==m and r['QP']==5 and r['video'] in ids]
    row=dict(phase='250 diagnostic' if step==250 else '1000 main',dataset=d,method=m,QP=5,expected_videos=len(ids),completed_videos=len(rr),comparison='same QP; measured bpp can differ')
    if len(rr)!=len(ids):row.update(status='unavailable',reason='incomplete fixed video pool')
    else:
     row.update(status='available',**{k:sum(float(r[k]) for r in rr)/len(rr) for k in ('bpp','LPIPS','DISTS','FloLPIPS')})
     pool=next((r for r in pools if r['dataset']==d and r['method']==m and r['QP']==5 and r['status']=='available'),None)
     if pool:row.update(**{k:pool[k] for k in ('FID','D_mu','D_cov')})
    rows.append(row)
 write(ROOT/'same_qp_summary.csv',rows)
 return rows
def pooled(records):
 import numpy as np
 from scipy.linalg import svdvals
 out=[]
 for d in ('ulong','uvg_holdout','uvg_validation','hevc_b','mcljcv','mcljcv29'):
  base='mcljcv' if d=='mcljcv29' else d;vs=load(ROOT/'eval_manifest.json')[base]
  if d=='mcljcv29':vs=[v for v in vs if 'videoSRC17' not in v['name']]
  for m in sorted({r['method'] for r in records}):
   for q in range(10):
    points=[next((r for r in records if r['dataset']==base and r['video']==v['video_index'] and r['method']==m and r['QP']==q),None) for v in vs]
    if any(p is None for p in points):
     out.append(dict(dataset=d,method=m,QP=q,status='unavailable',reason='incomplete fixed video pool',completed_videos=sum(p is not None for p in points),expected_videos=len(vs)));continue
    real=[];rec=[]
    for r in points:
     with np.load(r['feature_path']) as z:
      a=z['real'].astype(np.float64);b=z['reconstruction'].astype(np.float64);assert a.shape==b.shape==(64,2048) and np.isfinite(a).all() and np.isfinite(b).all();real.append(a);rec.append(b)
    a=np.concatenate(real);b=np.concatenate(rec);ma=a.mean(0);mb=b.mean(0);za=a-ma;zb=b-mb;dm=float(np.square(ma-mb).sum());dc=float(np.square(za).sum()/(len(a)-1)+np.square(zb).sum()/(len(b)-1)-2*svdvals(za@zb.T/(len(a)-1),check_finite=True).sum());fid=dm+dc
    assert all(math.isfinite(x) for x in (fid,dm,dc))
    out.append(dict(dataset=d,method=m,QP=q,status='available',FID=fid,D_mu=dm,D_cov=dc,bpp=sum(r['bpp'] for r in points)/len(points),sample_count=len(a),video_count=len(points),ddof=1,aggregation='fixed dataset UID pool; covariance not averaged per video',GT_pool_sha256=hashlib.sha256(a.tobytes()).hexdigest(),reconstruction_pool_sha256=hashlib.sha256(b.tobytes()).hexdigest()))
 return out
def equal_rate(records,pools,early=False):
 import numpy as np
 from scipy.interpolate import PchipInterpolator
 methods=['original','B1_initial']+[f'{b}_{250 if early else 1000}' for b in BRANCHES]
 comparisons=[(methods[3],methods[2]),(methods[4],methods[3]),(methods[5],methods[3]),(methods[4],methods[5])]+[(m,r) for m in methods[2:] for r in methods[:2]]
 rows=[];gridrows=[]
 def compare(curves,d,v,metric):
  counts={m:len(curves.get(m,[])) for m in methods};expect=3 if early else 10
  # Baselines diagnostic view uses same QP0/5/9, never compare 3-QP means to 10-QP main.
  if any(n!=expect for n in counts.values()):
   reason='incomplete six-method common-rate matrix';lo=hi=None;values=None
  else:
   xs={m:np.array([p['bpp'] for p in curves[m]],float) for m in methods}
   if any(len(set(x))!=len(x) or np.any(x<=0) for x in xs.values()):reason='duplicate/nonpositive bpp';lo=hi=None;values=None
   else:
    lo=max(float(x.min()) for x in xs.values());hi=min(float(x.max()) for x in xs.values());values=None;reason='no positive common rate interval'
    if hi>lo:
     grid=np.exp(np.linspace(np.log(lo),np.log(hi),100));values={}
     for m,x in xs.items():
      order=np.argsort(x);y=np.array([p[metric] for p in curves[m]],float);values[m]=PchipInterpolator(np.log(x[order]),y[order],extrapolate=False)(np.log(grid));assert np.isfinite(values[m]).all()
     for k,bpp in enumerate(grid):gridrows.append(dict(dataset=d,video=v,metric=metric,index=k,bpp=float(bpp),**{m:float(values[m][k]) for m in methods}))
  for m,r in comparisons:
   row=dict(dataset=d,video=v,metric=metric,method=m,reference=r,rate_axis='actual bpp',grid='100 uniform ln(bpp)',shared_bpp_min=lo,shared_bpp_max=hi,delta_definition='method-reference',covered_points=counts,expected_points=expect,aggregation='video-equal-weight' if v!='pooled' else 'fixed pooled features; mean dataset bpp',phase='250 diagnostic' if early else '1000 main')
   if values is None:row.update(status='unavailable',reason=reason,method_mean=None,reference_mean=None,delta=None,relative_change_percent=None,better_grid_fraction=None)
   else:
    a=float(values[m].mean());b=float(values[r].mean());row.update(status='available',method_mean=a,reference_mean=b,delta=a-b,relative_change_percent=100*(a-b)/b if b else None,better_grid_fraction=float((values[m]<values[r]).mean()))
   rows.append(row)
 for d,vs in load(ROOT/'eval_manifest.json').items():
  for v in vs:
   curves={m:[r for r in records if r['dataset']==d and r['video']==v['video_index'] and r['method']==m and (not early or r['QP'] in (0,5,9))] for m in methods}
   for metric in ('LPIPS','DISTS','FloLPIPS'):compare(curves,d,v['name'],metric)
 for d in ('ulong','uvg_holdout','uvg_validation','hevc_b','mcljcv','mcljcv29'):
  curves={m:[r for r in pools if r['dataset']==d and r['method']==m and r['status']=='available' and (not early or r['QP'] in (0,5,9))] for m in methods}
  for metric in ('FID','D_mu','D_cov'):compare(curves,d,'pooled',metric)
 return rows,gridrows
def finalize(failed):
 from evaluate32 import valid,task_id,receipt
 free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
 devices={int(x.split(',')[0]):float(x.split(',')[1]) for x in free if int(x.split(',')[0]) in (4,5,6,7)}
 gpu=max(devices,key=devices.get)
 assert devices[gpu]>14000,'Insufficient memory for the fixed independent-decode previews; no resolution change permitted'
 subprocess.run([PYTHON,'-B',str(ROOT/'preview32.py'),'--gpu',str(gpu)],cwd=REPO,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),check=True)
 ts=load(ROOT/'task_manifest.json')['tasks'];good=[];missing=[];records=[];perframe=[];pervideo=[];colors=[]
 for t in ts:
  if not valid(t):missing.append(dict(task=t,reason='C/D blocked full-resolution reference' if t.get('branch') in BRANCHES[2:] else 'missing or failed receipt'));continue
  rr=load(receipt(t));good.append(rr);r=rr['result']
  v=next(v for v in load(ROOT/'eval_manifest.json')[t['dataset']] if v['video_index']==t['video'])
  r=dict(r,native_fps=v['rate_accounting_fps'],source_fps_metadata=v.get('source_fps_metadata'),rate_accounting_fps=v['rate_accounting_fps'],kbps=r['bits_per_frame']*v['rate_accounting_fps']/1000,kbps_20fps=r['bits_per_frame']*20/1000,video=t['video'],method=t['method'],QP=t['qp'],adaptation_step=t.get('step',0))
  records.append(r)
  meta=dict(dataset=t['dataset'],video=t['video'],method=t['method'],QP=t['qp'],bpp=r['bpp'],adaptation_step=t.get('step',0))
  for a in read(r['frame_metrics_path']):perframe.append(dict(a,**meta))
  for a in read(r['transitions_path']):pervideo.append(dict(a,**meta,kind='transition'))
  c=r.get('color_path')
  if not c:
   c=V31/'raw/color'/f'color_{t["dataset"]}_{t["video"]}_{t["method"]}_{t["qp"]}.frames.csv'
  if Path(c).is_file():
   for a in read(c):
    if a.get('kind','reconstruction')=='reconstruction':colors.append(dict(a,**meta))
 keys=['dataset','video','sequence','method','QP','adaptation_step','real_bytes','bpp','bits_per_frame','native_fps','source_fps_metadata','rate_accounting_fps','kbps','kbps_20fps','LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM','source_rgb_sha256','source_frame_indices','actual_qps','checkpoint_sha256','reconstruction_sha256','bitstream_path','bitstream_sha256','feature_path','feature_sha256']
 write(ROOT/'evaluation_points.csv',fields(records,keys));write(ROOT/'per_frame.csv',perframe);write(ROOT/'per_video.csv',fields(records,keys));write(ROOT/'transitions.csv',pervideo)
 write(ROOT/'color_summary.csv',colors or [dict(status='unavailable',reason='no verified color records')])
 pools=pooled(records);write(ROOT/'pooled_fid.csv',pools)
 same_qp=same_qp_summary(records,pools)
 main,grid=equal_rate(records,pools);early,eg=equal_rate(records,pools,True);write(ROOT/'equal_rate_main.csv',main);write(ROOT/'equal_rate_early_diagnostic.csv',early);write(ROOT/'common_rate_grid.csv',grid or [dict(status='unavailable',reason='C/D missing from six-method common interval')]);write(ROOT/'early_common_rate_grid.csv',eg or [dict(status='unavailable',reason='C/D missing from six-method common interval')])
 summaries=[]
 for phase,rr in [('1000 main',main),('250 diagnostic',early)]:
  groups=collections.defaultdict(list)
  for r in rr:groups[(r['dataset'],r['metric'],r['method'],r['reference'])].append(r)
  for (d,metric,m,ref),rs in groups.items():
   available=[r for r in rs if r['status']=='available'];row=dict(phase=phase,dataset=d,metric=metric,method=m,reference=ref,expected_videos=len(rs),completed_videos=len(available),aggregation='equal video weights' if rs[0]['video']!='pooled' else 'dataset pooled FID',rate_axis='actual bpp',delta_definition='method-reference')
   if len(available)!=len(rs):row.update(status='unavailable',reason='incomplete required common-rate coverage',method_mean=None,reference_mean=None,delta=None,relative_change_percent=None)
   else:
    a=sum(r['method_mean'] for r in rs)/len(rs);b=sum(r['reference_mean'] for r in rs)/len(rs);row.update(status='available',method_mean=a,reference_mean=b,delta=a-b,relative_change_percent=100*(a-b)/b if b else None)
   summaries.append(row)
 write(ROOT/'equal_rate_summary.csv',summaries)
 update=[];feature=[];training={};calls={}
 for b in BRANCHES:
  bp=ROOT/'branches'/b;st=load(bp/'training_status.json');training[b]=st.get('adaptation_step',0)
  log=bp/'optimizer_calls.jsonl';physical=[json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
  count=sum(x['event']=='STARTED' for x in physical);calls[b]=dict(physical_optimizer_calls=count,committed_updates=training[b],replayed_or_discarded_calls=max(0,count-training[b]))
  for i in range(1,training[b]+1):
   r=load(bp/'receipts'/f'{i:04d}.json');update.append({k:v for k,v in r.items() if k not in ('frame_trace','full_RGB_sha256','cropped_RGB_sha256','source_frame_indices','canonical_indices')});feature.append(dict(branch=b,step=i,stratum=r['stratum'],visits=r['stratum_visits'],EMA_commit_count=r['EMA_commit_count'],status=r['full_frame_status'],feature_path=r['feature_path'],feature_sha256=r['feature_sha256']))
 write(ROOT/'loss_update_summary.csv',update or [dict(status='unavailable')]);write(ROOT/'feature_statistics_history.csv',feature or [dict(status='unavailable')])
 source_bad=[]
 for p,h in load(ROOT/'source_hashes.json')['hashes'].items():
  if sha(p)!=h:source_bad.append(p)
 # Verify every reused and new artifact content again, not just fast metadata admission.
 for rr in good:
  for p,v in rr['artifact_metadata'].items():
   if sha(p)!=v['sha256']:source_bad.append(p)
 dump(ROOT/'source_frozen_integrity.json',dict(status='PASS' if not source_bad else 'FAIL',changed_files=source_bad,dependencies_checked=len(load(ROOT/'source_hashes.json')['hashes']),evaluation_artifacts_checked=sum(len(r['artifact_metadata']) for r in good),historical_files_written=False))
 pre=load(ROOT/'prefill_integrity.json') if (ROOT/'prefill_integrity.json').exists() else dict(completed=0,expected=1000,status='FAIL')
 integrity=dict(status='BLOCKED',training=training,optimizer_call_accounting=calls,expected=dict(logical=3312,new_encodes=2392,baseline=920,prefill=1000,calibration=40),completed=dict(logical=len(good),new_encodes=sum(r.get('fresh_encode',False) for r in good),baseline=sum(r['task']['stage']=='baseline' for r in good),baseline_feature_reextractions=sum(r.get('feature_reextraction',False) for r in good),prefill=pre['completed'],calibration=0),missing=missing,failures=failed,common_check=load(ROOT/'audits/common.json')['status'],distribution_math=load(ROOT/'audits/distribution_math.json')['status'],complete_distribution_reference='BLOCKED_FULL_RESOLUTION_MISMATCH',source_frozen='PASS' if not source_bad else 'FAIL',per_point_frames=64,per_point_transitions=63,comparison_complete=False,unresolved=load(ROOT/'audits/full_resolution_conflicts.json')['conflicts'])
 dump(ROOT/'final_integrity.json',integrity)
 dump(ROOT/'BLOCKED.json',dict(status='BLOCKED',reason='Four fixed training windows not 1920x1080; complete seeded statistics and C/D dependencies blocked',training=training,evaluation=integrity['completed'],publication='PENDING'))
 dump(ROOT/'gradient_integrity.json',dict(common_C1='PASS',math_toy='PASS',C_D_full_chain='BLOCKED_REFERENCE_INCOMPLETE',details=load(ROOT/'audits/common.json')))
 dump(ROOT/'forward_replay_integrity.json',dict(status='BLOCKED',reason='C/D complete reference dependency not met; no full-chain FD calibration/VJP training executed',toy=load(ROOT/'audits/distribution_math.json')['two_pass_toy']))
 binary=[]
 for p in ROOT.rglob('*'):
  if p.is_file() and p.suffix in ('.pt','.bin','.npz','.npy'):binary.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 dump(ROOT/'binary_artifacts_index.json',dict(artifacts=binary))
 report=['# v6.32 execution record','',f'Status: BLOCKED. Actual committed updates: {training}.',f'Physical optimizer calls and replay/discard accounting: {calls}.',f'Logical evaluation: {len(good)}/3312; new encodes: {integrity["completed"]["new_encodes"]}/2392; baseline aliases: {integrity["completed"]["baseline"]}/920.',f'B1 full-frame prefill: {pre["completed"]}/1000. Calibration: 0/40 (blocked).','', 'The four fixed-plan source windows listed in audits/full_resolution_conflicts.json have native heights 960, 1038, 810, 816 instead of the required 1080. No resize, padding substitution, frame or plan replacement was performed. A/B crop C1 optimization continues independently; C/D, complete seeded EMA and their comparisons remain unavailable.','',f'Source: {load(ROOT/"config.json")["source_checkpoint"]}.','Objective: fused LPIPS + .5 DISTS + beta(q)*rate + .01 proxy; separately restored 4096 reconstruction/stat gradients, fixed weights .1383589248954108 and .038636700918173825; supervised P49–63 mean /15.','A B/G LR: 3e-6,5e-7. B: 6e-6,1e-6. Fresh AdamW (.9,.999), eps1e-8, decay1e-4, global clip1. P/core/I/quality frozen. D updates zero.','', '250-step 3-QP diagnostic tables and 1000-step 10-QP main tables are separate. Common six-method intervals and unavailable rows are preserved; no missing branch is dropped from a claimed full average. FID uses each fixed dataset pool, float64 ddof1; MCL29 is a subset of MCL30 with no new encoding.','', 'A/B full-frame auxiliary pre-update features are preserved when native dimensions match, with independent global RNG and restored training buffers. No incomplete reference is substituted for a complete EMA seed. Crop updates do not depend on auxiliary statistics. Correlated frames and repeated UIDs are not independent samples. FP32 training and FP16 deployment hashes remain separate.','', 'See evaluation_points.csv, pooled_fid.csv, equal_rate_main.csv, equal_rate_early_diagnostic.csv, loss_update_summary.csv, feature_statistics_history.csv and final_integrity.json for mechanical numeric records and exact coverage. No branch/checkpoint selection was performed.']
 report+=['','## Fixed QP5 raw diagnostics','',
  'Video metrics use equal video weights. FID and its components use the fixed pooled frame set. These are same-QP observations; differing measured bpp prevents claiming equal-rate gains. The required six-method equal-rate comparisons remain unavailable because C/D are blocked.','',
  '| Phase | Dataset | Method | bpp | LPIPS | DISTS | FloLPIPS | pooled FID | D_mu | D_cov |',
  '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in same_qp:
  vals=[format(r[k],'.8g') if r.get(k) is not None else 'unavailable' for k in ('bpp','LPIPS','DISTS','FloLPIPS','FID','D_mu','D_cov')]
  report.append('| '+' | '.join([r['phase'],r['dataset'],r['method'],*vals])+' |')
 report+=['','## Verification and incomplete dependencies','',
  f'Common C1 checks: {integrity["common_check"]}; FP64 distribution mathematics/toy VJP: {integrity["distribution_math"]}; source/frozen artifact verification: {integrity["source_frozen"]}.',
  'Full-chain C/D replay, calibration coefficients, EMA commits and optimizer/EMA restoration are unexecuted and BLOCKED. Their numeric coefficients and gradients are unavailable, rather than zero or passing. A/B EMA commit count is zero because the required shared seed is incomplete.',
  'The source heights above conflict with the full-frame requirement. Eligible prefill has 996 plan occurrences and 14940 target feature occurrences; 16/20 strata have complete planned coverage. Unique UID/window/video counts and low-rank bounds are in prefill_integrity.json. Repeated or correlated frames are not independent samples.',
  'Fixed display: HEVC and MCL first video, QP5, frames16/32/63; GT/original/B1/A/B are shown with measured bpp, and missing C/D are labeled unavailable. See previews/hevc_b/fixed_comparison.png and previews/mcljcv/fixed_comparison.png.']
 write(ROOT/'calibration/raw_norms.csv',[dict(status='BLOCKED',completed_windows=0,expected_windows=40,reason='complete shared full-resolution GT and B1 EMA seed unavailable')])
 textfile(ROOT/'REPORT.md','\n'.join(report)+'\n')
 subprocess.run(['/data1/anaconda3_new/anaconda_program/bin/python','-B',str(ROOT/'plot32.py')],cwd=REPO,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),check=True)
 print('REPORT COMPLETE WITH EXPLICIT BLOCKERS',flush=True)
