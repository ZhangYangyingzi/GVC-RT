"""Additional formal workers admitted by measured memory, without altering numerical workers.
The primary driver remains the sole aggregation/integrity/publication owner.
A 64-group frontier keeps new claims disjoint from its forward-order submissions.
"""
import subprocess,fcntl,time,shutil,traceback
from audit_io import *
from worker import ORDER,valid
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
def lock_available(job):
 p=ROOT/'cache/locks'/f"{job['dataset']}_{job['video']}_{job['qp']}.lock";p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN);return True
  except BlockingIOError:return False

def main():
 lock=(ROOT/'parallel_resources.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 tasks=[]
 for d,vs in load(ROOT/'dataset_manifest.json')['datasets'].items():
  for v in vs:
   for q in range(10):tasks.append(dict(dataset=d,video=v['video_index'],qp=q))
 running=[];failed={};attempts={};completed=set();peak_nvml=5800.;rss=load(ROOT/'sanity/sanity.json')['peak_RSS_MiB'];last_hour=None;cooldown=0
 def tag(j):return f"{j['dataset']}_{j['video']}_{j['qp']}"
 def done(j):
  k=tag(j)
  if k in completed:return True
  if all(valid(dict(j,method=m)) for m in ORDER):completed.add(k);return True
  return False
 while True:
  for w in list(running):
   rc=w['p'].poll()
   if rc is None:continue
   w['log'].close();running.remove(w);k=tag(w['job'])
   if rc:
    tail=Path(w['logpath']).read_text(errors='replace')[-6000:];attempts[k]=attempts.get(k,0)+1
    if 'out of memory' in tail.lower() and attempts[k]<=4:peak_nvml=max(peak_nvml*1.15,peak_nvml+1024);cooldown=time.time()+120
    else:failed[k]=dict(job=w['job'],returncode=rc,error=tail)
   else:
    assert done(w['job']);st=load(ROOT/f'worker_status/{k}.json');assert st['status']=='PASS';rss=max(rss,st['peak_RSS_MiB']);peak_nvml=max(peak_nvml,st['peak_memory_MiB']+512)
  base=load(ROOT/'status.json')
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
  cards={int(x.split(',')[0]):dict(uuid=x.split(',')[1].strip(),free=float(x.split(',')[2])) for x in raw if int(x.split(',')[0]) in (4,5,6,7)}
  applications=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines();usage={int(x.split(',')[0]):float(x.split(',')[2]) for x in applications if x.split(',')[2].strip().isdigit()}
  primary=base.get('running',[]);residents=[dict(PID=w['PID'],gpu=w['gpu']) for w in primary]+[dict(PID=w['p'].pid,gpu=w['gpu']) for w in running]
  for w in residents:peak_nvml=max(peak_nvml,usage.get(w['PID'],0))
  reserve=peak_nvml*1.25
  info={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()};host=info['MemAvailable']/1024
  # Reserve the FULL remaining peak of every resident v6.33 process, rather than
  # mistaking its low allocation during loading/CPU IO for spare steady-state VRAM.
  capacity={g:c['free']-2048-sum(max(0.,reserve-usage.get(w['PID'],0.)) for w in residents if w['gpu']==g) for g,c in cards.items()}
  remaining=[j for j in tasks if not done(j)]
  own={tag(w['job']) for w in running}
  if base.get('stage') in ('EVALUATING','FULL64_CANARY') and time.time()>=cooldown and shutil.disk_usage(ROOT).free>=20*2**30:
   # Primary can start at most eight forward groups per pass. Sixty-four incomplete
   # groups separate its submission frontier from our reverse-order new claims.
   candidates=remaining[64:]
   for job in reversed(candidates):
    k=tag(job)
    if k in own or k in failed or not lock_available(job):continue
    eligible=[g for g in cards if capacity[g]>=reserve and host>=rss*1.25+8192]
    if not eligible:break
    g=max(eligible,key=capacity.get);lp=ROOT/'logs'/f'parallel_{k}.log';log=lp.open('a');argv=[PYTHON,'-B','-u',str(ROOT/'worker.py'),'--dataset',job['dataset'],'--video',str(job['video']),'--qp',str(job['qp']),'--gpu',str(g)]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (ROOT/'logs/parallel_admissions.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),job=job,gpu=g,measured_peak_nvml_MiB=peak_nvml,reserve_MiB=reserve,card_free_MiB=cards[g]['free'],remaining_after_resident_peak_margin=capacity[g],host_free_MiB=host,argv=argv,frontier_guard_groups=64))+'\n')
    p=subprocess.Popen(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT);running.append(dict(p=p,log=log,logpath=str(lp),gpu=g,job=job));own.add(k);capacity[g]-=reserve;host-=rss*1.25
  terminal=not running and (not remaining or base.get('terminal',False) or (len(remaining)<=64 and not failed))
  status=dict(status='COMPLETE' if terminal else 'RUNNING',terminal=terminal,PID=os.getpid(),heartbeat=time.time(),primary_driver_PID=base.get('driver_PID'),running=[dict(PID=w['p'].pid,gpu=w['gpu'],job=w['job']) for w in running],measured_nvml_peak_MiB=peak_nvml,worker_reservation_MiB=reserve,RSS_peak_MiB=rss,remaining_groups=len(remaining),failures=failed,no_fixed_per_gpu_process_cap=True,primary_finalizer_unchanged=True,frontier_guard_groups=64)
  dump('parallel_resources_status.json',status)
  if terminal:return
  time.sleep(60)
if __name__=='__main__':
 try:main()
 except Exception:dump('parallel_resources_error.json',dict(error=traceback.format_exc(),time=time.time()));raise
