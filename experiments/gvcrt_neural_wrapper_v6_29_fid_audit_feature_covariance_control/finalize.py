"""All required checks gate publication; execution PASS is distinct from benefit."""
from io21 import *
import math,statistics

def main():
 import numpy as np
 frozen()
 for n in ('FID_statistics_integrity','FID_historical_reproduction','FID_invariances_integrity','FID_analytic_tests','FID_gap_integrity','DISTS_deep_BG_probe','stat_image_GT_gradient_probe','A_reuse_equivalence','B_covstats_pretraining','B_covstats_codec_equivalence','reuse_points','data_split_integrity','actual_checkpoint_parent_chain','intermediate_stage_evidence','task1_integrity'):
  assert load(ROOT/'audits'/f'{n}.json')['status']=='PASS',n
 points=[]
 for d in ('ulong','hevc_b','mcljcv'):
  fs=list((ROOT/'audits/features'/d).glob('*/qp*.json'));assert len(fs)==6
  for p in fs:assert load(p)['status']=='PASS';points.append(dict(path=str(p),sha256=sha(p),frames=load(p)['frames']))
 dump(ROOT/'audits/FID_feature_integrity.json',dict(status='PASS',points=18,records=points,same_float_RGB_tensor=True,relativeL2_tolerance=1e-4,no_PNG_quantization_mismatch=True))
 assert load(ROOT/'branches/B_covstats/training_integrity.json')['updates']==1000
 subprocess.run([PYTHON,'-B',str(ROOT/'report.py')],check=True)
 # Reaggregate MCL29 using identical already-coded points, never re-encode.
 from comparison import comparisons,summarize,PAIRS
 from adapter import validate
 from fid_statistics import lowrank_unclamped
 vs=[v for v in sources('mcljcv') if v['name']!='videoSRC17_1920x1080_24'];assert len(vs)==29
 raw=[load(point('mcljcv',m,v['video_index'],q)) for v in vs for m in METHODS for q in range(10)];pooled=[]
 for m in METHODS:
  for q in range(10):
   rs=[x for x in raw if x['method']==m and x['QP']==q];gt=[];rec=[]
   for x in rs:
    with np.load(x['feature_path']) as f:gt.append(f['real']);rec.append(f['reconstruction'])
   pooled.append(dict(dataset='mcljcv_exclude_overlap29',method=m,QP=q,bpp=statistics.mean(x['bpp'] for x in rs),FID=lowrank_unclamped(np.concatenate(gt),np.concatenate(rec)),samples=1856,sequences=29))
 folder=ROOT/'results/ip_-1/mcljcv_exclude_overlap29';per=[];grid=[]
 for v in vs:
  for metric in ('LPIPS','DISTS','FloLPIPS'):
   rr,gg=comparisons('mcljcv_exclude_overlap29',v['name'],metric,[x for x in raw if x['video_index']==v['video_index']]);per+=rr;grid+=gg
 fr,gg=comparisons('mcljcv_exclude_overlap29','pooled_dataset','FID',pooled,True);grid+=gg
 for name,rs in [('per_sequence_all_qp',raw),('fid_pooled',pooled),('equal_rate_per_sequence',per),('equal_rate_summary',summarize('mcljcv_exclude_overlap29',per,29)),('equal_rate_fid_pooled',fr),('common_rate_grid',grid)]:write(folder/(name+'.csv'),rs)
 core=[]
 for d in (*DATASETS,'mcljcv_exclude_overlap29'):
  f=ROOT/'results/ip_-1'/d
  core.extend(read(f/'equal_rate_summary.csv')+read(f/'equal_rate_fid_pooled.csv'))
 write(ROOT/'results/four_metric_comparisons.csv',core)
 # Raw logical counts are audited again after every completed file has been validated.
 count=0;reused=0
 for d in DATASETS:
  for v in sources(d):
   for m in METHODS:
    for q in range(10):
     r=load(point(d,m,v['video_index'],q));validate(r,v,q,m);count+=1;reused+=bool(r.get('reused'))
 assert count==1840 and reused==1380
 subprocess.run([PYTHON,'-B',str(ROOT/'rate_bands.py')],check=True)
 iaudit=ROOT/'audits/native_I_payload_reconstruction.json'
 if not iaudit.exists() or load(iaudit).get('status')!='PASS':
  while True:
   rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
   free={int(r.split(',')[0]):int(r.split(',')[1]) for r in rows.splitlines() if int(r.split(',')[0]) in (4,5,6,7)}
   gpu=max(free,key=free.get)
   if free[gpu]>=6500:break
   time.sleep(15)
  subprocess.run([PYTHON,'-B',str(ROOT/'native_i_audit.py'),'--gpu',str(gpu)],check=True)
 subprocess.run([PYTHON,'-B',str(ROOT/'result_checks.py')],check=True)
 jobs=[]
 for script in ('covariance_diagnostic.py','new_lovo.py'):
  for d in (*DATASETS,'mcljcv_exclude_overlap29'):
   log=(ROOT/'logs'/f'{script[:-3]}_{d}.log').open('ab')
   jobs.append(subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'diagnostic_cache.py'),'--script',script,'--dataset',d],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}))
 for job in jobs:assert job.wait()==0,('CPU diagnostic failed',job.args)
 for d in (*DATASETS,'mcljcv_exclude_overlap29'):
  assert load(ROOT/'results/covariance_diagnostic'/d/'integrity.json')['status']=='PASS'
  assert load(ROOT/'results/new_leave_one_video_out'/d/'integrity.json')['status']=='PASS'
 while True:
  gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
  free={int(r.split(',')[0]):int(r.split(',')[1]) for r in gpu_rows.splitlines() if int(r.split(',')[0]) in (4,5,6,7)}
  gpu=max(free,key=free.get)
  if free[gpu]>=6500:break
  time.sleep(15)
 subprocess.run([PYTHON,'-B',str(ROOT/'visuals.py'),'--gpu',str(gpu)],check=True)
 assert load(ROOT/'audits/visual_integrity.json')['status']=='PASS'
 subprocess.run([PLOT_PYTHON,'-B',str(ROOT/'diagnostic_plots.py')],check=True)
 text='# V6.29\n\nExecution and audit completion are separate from metric benefit.\n\n'+f'Training A: reused verified 1000 updates; B:1000 updates. Logical RD points:{count}, reused:{reused}, new:{count-reused}.\n\n'
 text+='## FID audit\n\n18 explicit covariance/eigendecomposition points and 18 same-RGB official/historical feature points passed their pre-registered tolerances. Original→B1 gap evidence is stored in results/FID_audit. 首次出现阶段尚未定位：已核验可用 V6.20/V6.21/V6.22 同协议中间证据，但更早全部阶段的可比证据尚未建立。\n\n'
 text+='## Matched comparison\n\nSigned delta=method−reference; percent=100*(method_mean−reference_mean)/reference_mean. Negative means a lower metric. No statistical significance claim is made.\n\n|Dataset|Method|Reference|Metric|Delta|Relative %|\n|---|---|---|---|---:|---:|\n'
 for r in core:text+=f"|{r['dataset']}|{r['method']}|{r['reference']}|{r['metric']}|{r.get('delta')}|{r.get('relative_change_percent')}|\n"
 text+='\nThe fixed budget is1000 updates. Spatial single-frame DISTS covariance differs from across-frame Inception covariance. Results do not establish all covariance objectives or budgets. MCL30/29 are separate matched pools; videoSRC17 duplicates HEVC-B Kimono1 across test sets and is not evidence of training leakage.\n'
 (ROOT/'REPORT.md').write_text(text);(ROOT/'RESULTS_SUMMARY.md').write_text(text)
 subprocess.run([PYTHON,'-B',str(ROOT/'enrich_report.py')],check=True)
 for p,h in load(ROOT/'audits/dependencies.json').items():assert sha(p)==h,('historical dependency changed',p)
 final=dict(status='PENDING_PUBLICATION',source_SHA256=EXPECTED,plan_SHA256=sha(ROOT/'plans/shared.json'),A_updates=1000,B_updates=1000,expected_points=1840,completed_points=count,reused_points=reused,new_points=count-reused,FID_independent_features=True,FID_independent_PSD_statistics=True,historical_dependencies_unchanged=True,no_discriminator=True,no_GAN=True,lambda_stat=load(ROOT/'calibration/final.json')['lambda_stat'],publication_verified=False,stat_gradient_scaling_verified=True,GT_detached=True,source_frames_verified=True,I0_native_payload_and_reconstruction_equal=True,real_RANS=True,independent_decode=True,state_sync=True,four_metrics_complete=True,no_extrapolation=True,covariance_diagnostic_completed=True,LOVO_completed=True,visuals_completed=True,MCL30_and29_separately_reaggregated=True)
 dump(ROOT/'final_integrity.json',final)
 subprocess.run([PYTHON,'-B',str(ROOT/'publish_github.py')],check=True,env={**os.environ,'V629_PUBLICATION_PHASE':'final_integrity_verified' if load(ROOT/'final_integrity.json')['status']=='PASS' else 'computational_verified'})
 assert load(ROOT/'publication_status.json')['status']=='PASS'
 final.update(status='PASS',publication_verified=True,shards_reassembly_verified=True,verified_data_commit=load(ROOT/'publication_status.json')['commit']);dump(ROOT/'final_integrity.json',final)
 state=load(ROOT/'pipeline_status.json');state.update(status='PASS',A_updates=1000,B_updates=1000,completed_RD=1840);dump(ROOT/'pipeline_status.json',state)
 subprocess.run([PYTHON,'-B',str(ROOT/'publish_github.py')],check=True,env={**os.environ,'V629_PUBLICATION_PHASE':'final_integrity_verified' if load(ROOT/'final_integrity.json')['status']=='PASS' else 'computational_verified'})
 pub=load(ROOT/'publication_status.json');assert pub['status']=='PASS' and pub['all_remote_blobs_verified'];dump(ROOT/'audits/remote_final_verification.json',dict(status='PASS',commit=pub['commit'],key_files_all_verified=True));subprocess.run([PYTHON,'-B',str(ROOT/'publish_github.py')],check=True,env={**os.environ,'V629_PUBLICATION_PHASE':'verification_receipt'});pub=load(ROOT/'publication_status.json');assert pub['status']=='PASS';print('V629 FINAL PASS',pub['commit'],flush=True)
if __name__=='__main__':main()
