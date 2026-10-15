"""Detached resource-admitted evaluation -> aggregation -> integrity -> normal publication."""
import subprocess,fcntl,time,traceback,shutil,gc
from audit_io import *
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
ANALYSIS_PYTHON='/data1/anaconda3_new/anaconda_program/bin/python'

def resources():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines();free={int(x.split(',')[0]):float(x.split(',')[1]) for x in rows if int(x.split(',')[0]) in (4,5,6,7)}
 info={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()};return free,info['MemAvailable']/1024

def main():
 from worker import ORDER,valid,key,code_hashes
 lock=(ROOT/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert load(ROOT/'preflight.json')['status']==load(ROOT/'sanity/sanity.json')['status']==load(ROOT/'selftest.json')['status']==load(ROOT/'analysis_selftest.json')['status']=='PASS'
 assert all(sha(ROOT/n)==h for n,h in load(ROOT/'implementation_hashes.json').items())
 proto=load(ROOT/'protocol.json');videos=load(ROOT/'dataset_manifest.json')['datasets'];pending=[]
 for d,vs in videos.items():
  for v in vs:
   for q in range(10):
    t=dict(dataset=d,video=v['video_index'],qp=q)
    if not all(valid(dict(t,method=m)) for m in ORDER):pending.append(t)
 running=[];failed=[];attempts={};last_hour=None;started=time.time();measured=load(ROOT/'sanity/sanity.json');peak=measured['peak_memory_MiB'];rss=measured['peak_RSS_MiB'];pilot=ROOT/'parts/ulong/00/q0/B1_alpha0/point.json';pilot_ready=pilot.exists();retired=[]
 while pending or running:
  for w in list(running):
   rc=w['p'].poll()
   if rc is None:continue
   w['log'].close();running.remove(w);job=w['job'];tail=Path(w['logpath']).read_text(errors='replace')[-8000:]
   if rc:
    tag=json.dumps(job,sort_keys=True);attempts[tag]=attempts.get(tag,0)+1
    if 'out of memory' in tail.lower() and attempts[tag]<=4:
     # Retry only this evaluation group after other workers drain; never alter protocol.
     pending.insert(0,job);peak=max(peak*1.15,peak+1024);retired.append(dict(job=job,kind='OOM_backoff',attempt=attempts[tag]));time.sleep(10)
    else:failed.append(dict(job=job,returncode=rc,tail=tail))
   else:
    st=load(ROOT/f"worker_status/{job['dataset']}_{job['video']}_{job['qp']}.json");assert st['status']=='PASS';peak=max(peak,st['peak_memory_MiB']);rss=max(rss,st['peak_RSS_MiB'])
  if pilot.exists():
   pr=load(pilot);pilot_ready=True;peak=max(peak,pr['peak_memory_MiB']);rss=max(rss,pr['peak_RSS_MiB'])
  free,host=resources();occupied={g:sum(w['gpu']==g for w in running) for g in (4,5,6,7)}
  allow=1 if not pilot_ready else 8
  for job in list(pending):
   if len(running)>=allow:break
   if all(valid(dict(job,method=m)) for m in ORDER):pending.remove(job);continue
   lockpath=ROOT/'cache/locks'/f"{job['dataset']}_{job['video']}_{job['qp']}.lock";lockpath.parent.mkdir(parents=True,exist_ok=True)
   with lockpath.open('a') as pointlock:
    try:fcntl.flock(pointlock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(pointlock,fcntl.LOCK_UN)
    except BlockingIOError:continue  # An orphan worker from the same protocol still owns this group.
   if shutil.disk_usage(ROOT).free<20*2**30:break
   # Free memory already excludes allocations. Reserve 25% headroom for resident jobs too.
   eligible=[g for g in (4,5,6,7) if occupied[g]<2 and free[g]>=peak*1.25+2048+occupied[g]*peak*.25 and host>=rss*1.25+8192]
   if not eligible:continue
   pref=proto['gpu_preference'][job['dataset']];g=pref if pref in eligible else max(eligible,key=free.get)
   tag=f"{job['dataset']}_{job['video']}_{job['qp']}";lp=ROOT/'logs'/f'{tag}.log';log=lp.open('a');cmd=[PYTHON,'-B','-u',str(ROOT/'worker.py'),'--dataset',job['dataset'],'--video',str(job['video']),'--qp',str(job['qp']),'--gpu',str(g)]
   env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   p=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT);running.append(dict(p=p,log=log,logpath=str(lp),gpu=g,job=job));pending.remove(job);occupied[g]+=1;free[g]-=peak*1.25+2048;host-=rss*1.25
  # Receipts are small; do not scan streams or tensors during heartbeat.
  completed=0
  for d,vs in videos.items():
   for v in vs:
    for q in range(10):
     for m in ORDER:
      if (ROOT/'parts'/key(dict(dataset=d,video=v['video_index'],qp=q,method=m))/'point.json').exists():completed+=1
  status=dict(stage='EVALUATING' if pilot_ready else 'FULL64_CANARY',terminal=False,driver_PID=os.getpid(),heartbeat=time.time(),completed_points=completed,expected_points=3220,pending_groups=len(pending),failed_groups=len(failed),running=[dict(PID=w['p'].pid,gpu=w['gpu'],job=w['job']) for w in running],measured_peak_MiB=peak,measured_RSS_MiB=rss,report='PENDING',publication='PENDING')
  dump('status.json',status)
  hour=int(time.time()//3600)
  if hour!=last_hour:
   last_hour=hour
   with (ROOT/'hourly_summary.txt').open('a') as f:f.write(json.dumps({k:status[k] for k in ('heartbeat','stage','completed_points','pending_groups','failed_groups')})+'\n')
  if pending and not running and shutil.disk_usage(ROOT).free<20*2**30:raise RuntimeError('BLOCKED_DISK_ADMISSION: less than 20GiB free; existing results preserved')
  time.sleep(60)
 if failed:dump('failed_groups.json',failed);raise RuntimeError('Evaluation groups failed; no incomplete report publication')
 dump('status.json',dict(stage='AGGREGATING',terminal=False,driver_PID=os.getpid(),heartbeat=time.time(),completed_points=3220,report='RUNNING',publication='PENDING'))
 with (ROOT/'logs/analysis.log').open('a') as out:
  subprocess.run([ANALYSIS_PYTHON,'-B','-u',str(ROOT/'analysis.py')],cwd=REPO,stdout=out,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),check=True)
 changed=[p for p,h in load(ROOT/'source_hashes.json').items() if sha(p)!=h];assert not changed,changed
 assert all(sha(ROOT/n)==h for n,h in load(ROOT/'implementation_hashes.json').items())
 required=['README.md','PROTOCOL.md','protocol.json','all_points.csv.gz','same_qp_rate_deltas.csv','alpha_summary.csv','latent_fidelity_summary.csv','coding_diagnostics.csv','quantization_diagnostics.csv','entropy_diagnostics.csv','bootstrap_summary.csv','trend_summary.csv','mechanism_decision.json','REPORT.md','point_receipts.jsonl.gz']
 assert all((ROOT/n).exists() for n in required)
 figs=['alpha_vs_actual_bpp','alpha_vs_teacher_cosine','alpha_vs_lpips','alpha_vs_dists','alpha_vs_flolpips','delta_teacher_vs_delta_bpp','delta_teacher_vs_delta_lpips','quantization_error_vs_alpha','symbol_change_fraction_vs_alpha','entropy_bits_vs_alpha','rate_change_by_dataset','mechanism_summary']
 assert all((ROOT/'figures'/f'{n}.{ext}').exists() for n in figs for ext in ('png','pdf'))
 dump('integrity.json',dict(status='PASS',completed_points=3220,expected_points=3220,source_files_unchanged=True,checkpoint_files_unchanged=True,no_training=True,optimizer_updates=0,real_persisted_RANS=True,fresh_independent_decode=True,causal_DPB=True,native_I_bypass=True,teacher='paper_compatible_teacher_proxy',exact_original_teacher_verified=False,worker_OOM_backoffs=retired,artifacts={n:sha(ROOT/n) for n in required}))
 dump('status.json',dict(stage='PUBLISHING',terminal=False,driver_PID=os.getpid(),heartbeat=time.time(),completed_points=3220,report='COMPLETE',integrity='PASS',publication='RUNNING'))
 from publish import main as publish
 published=publish();final=dict(stage='DONE' if published else 'COMPLETE_LOCAL_PUSH_FAILED',terminal=True,completed_points=3220,report='COMPLETE',integrity='PASS',publication=load(ROOT/'publication_status.json')['status'],driver_PID=os.getpid(),finished=time.time())
 dump('status.json',final);dump('DONE.json' if published else 'PUBLICATION_FAILED.json',final)
 print('PIPELINE_TERMINAL',final,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  msg=traceback.format_exc();dump('errors.json',dict(error=msg,time=time.time()))
  if (ROOT/'integrity.json').exists() and load(ROOT/'integrity.json')['status']=='PASS':
   old=load(ROOT/'publication_status.json') if (ROOT/'publication_status.json').exists() else {};old.update(status='FAILED',error=msg);dump('publication_status.json',old)
  final=dict(stage='FAILED',terminal=True,driver_PID=os.getpid(),error=msg,results_preserved=True,finished=time.time());dump('status.json',final);dump('FAILED.json',final);raise
