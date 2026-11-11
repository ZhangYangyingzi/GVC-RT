"""Bounded multi-GPU queue for immutable checkpoint evaluation; numerical code unchanged."""
from common_s2 import *
import traceback

def submit(model,step,phase='development'):
 key=f'{phase}_{model}_{step:04d}';dest=ROOT/'evaluation_queue/jobs'/f'{key}.json'
 if not dest.exists():dump(dest,dict(model=model,step=step,phase=phase,created=time.time()))
 return key

def effective_step(model,step):
 if model.endswith('_selected'):
  selection=load(ROOT/'selection.json');assert selection['status']=='FROZEN'
  return selection['groups'][model[:-9]]['selected']['step']
 return step

def progress(model,step,phase='development'):
 return len(list((ROOT/'evaluation'/phase/model/f'step_{step:04d}/points').glob('*.json')))

def point_path(model,step,phase,v,q):
 return ROOT/'evaluation'/phase/model/f'step_{effective_step(model,step):04d}/points'/f'{v["dataset"]}__{v["sequence"]}_qp{q}.json'

def pending_bundles():
 """Each source/QP is assigned once, with all currently registered missing states."""
 bundles={};counts={}
 for path in sorted((ROOT/'evaluation_queue/jobs').glob('*.json'),key=lambda p:p.stat().st_mtime):
  job=load(path);phase=job['phase'];model=job['model'];step=job['step']
  if phase=='final':assert load(ROOT/'selection.json')['status']=='FROZEN'
  videos=load(ROOT/f'manifests/{"development" if phase=="development" else "test"}_videos.json');done=0
  for index,(v,q) in enumerate((v,q) for v in videos for q in [0,5,9]):
   dest=point_path(model,step,phase,v,q)
   if dest.exists():
    a=load(dest);assert a['status']=='PASS';done+=1;continue
   key=f'{phase}_{index:04d}'
   bundle=bundles.setdefault(key,dict(key=key,phase=phase,index=index,video=v,QP=q,tasks=[]))
   task=dict(model=model,step=step)
   if task not in bundle['tasks']:bundle['tasks'].append(task)
  counts[path.stem]=dict(completed=done,expected=len(videos)*3)
  if done==len(videos)*3 and not (ROOT/'audits/evaluation_queue_done'/path.name).exists():
   dump(ROOT/'audits/evaluation_queue_done'/path.name,dict(status='PASS',**job,points=done,time=time.time()))
 return list(bundles.values()),counts

def worker():
 env()
 from pipeline import ready,request,positions
 from cache_codec import cache_paths
 active={};completed=0;begin=time.time();cache_wait=0.;last=time.time()
 with lock('locks/evaluation_queue.lock'):
  try:
   while True:
    cfg=load(ROOT/'scheduler_config.json');gpus=cfg.get('evaluation_GPUs',[3]);assert set(gpus).issubset({0,1,2,3})
    for key,j in list(active.items()):
     code=j['p'].poll()
     if code is None:continue
     try:
      assert code==0,('EVALUATION_BUNDLE_FAILED',j['spec'],code,j['gpu'])
      proof=load(j['receipt']);assert proof['status']=='PASS' and proof['numerical_evaluation_SHA256']==sha(ROOT/'evaluation.py')
      for task in j['spec']['tasks']:
       a=load(point_path(task['model'],task['step'],j['spec']['phase'],j['spec']['video'],j['spec']['QP']));assert a['status']=='PASS'
       for file,h in a['record_SHA256'].items():assert sha(ROOT/file)==h
      completed+=len(j['spec']['tasks']);event('evaluation_bundle_complete',key=key,physical_GPU=j['gpu'],states=len(j['spec']['tasks']),seconds=time.time()-j['start'],peak_allocated_bytes=proof['peak_allocated_bytes'])
     finally:
      j['log'].close();j['cache_lease'].close();j['gpu_lease'].close();active.pop(key)
    stopping=(ROOT/'evaluation_queue/STOP').exists();bundles,counts=pending_bundles()
    pending=[b for b in bundles if b['key'] not in active]
    window=pending[:cfg.get('native_prefetch_points',6)]
    if not stopping:
     for b in window:request(b['video'].get('source_row',b['video']),b['QP'],False)
     pos=min(positions().values());training_ready=pos>=400 or sum(ready(r,r['external_qp'],True) for r in plan_rows()[pos:min(400,pos+32)])>=16
     if training_ready:
      for b in window:
       if len(active)>=cfg.get('max_evaluation_workers',len(gpus)):break
       row=b['video'].get('source_row',b['video']);q=b['QP']
       if not ready(row,q,False):continue
       lease=None;gpu=None
       for candidate in gpus:
        if candidate in {j['gpu'] for j in active.values()}:continue
        f=(ROOT/f'locks/GPU{candidate}.lock').open('a')
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:f.close();continue
        lease=f;gpu=candidate;break
       if lease is None:break
       key,p,rp=cache_paths(row,q,False);cache_lease=(ROOT/f'locks/lease_{key}.lock').open('a')
       try:fcntl.flock(cache_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
       except BlockingIOError:cache_lease.close();lease.close();continue
       if not ready(row,q,False):cache_lease.close();lease.close();continue
       spec=ROOT/'tmp/evaluation_jobs'/f'{b["key"]}_{time.time_ns()}.json';dump(spec,b)
       receipt=ROOT/'audits/evaluation_bundles'/spec.name;log=(ROOT/f'logs/evaluation_GPU{gpu}.log').open('ab')
       child_env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),H_CACHE_LEASE_HELD=key)
       try:child=subprocess.Popen([PYTHON,'-B','-u','evaluation_point_worker.py',str(spec)],cwd=ROOT,env=child_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
       except Exception:log.close();cache_lease.close();lease.close();raise
       active[b['key']]=dict(p=child,spec=b,gpu=gpu,cache_lease=cache_lease,gpu_lease=lease,log=log,receipt=receipt,start=time.time())
    now=time.time()
    waiting=bool(window) and not any(ready(b['video'].get('source_row',b['video']),b['QP'],False) for b in window)
    if waiting:cache_wait+=now-last
    last=now
    dump(ROOT/'evaluation_queue/progress.json',dict(status='DRAINING' if stopping else 'RUNNING' if active else 'WAITING_CACHE' if waiting else 'IDLE',GPU_pool=gpus,active={key:dict(PID=j['p'].pid,physical_GPU=j['gpu'],phase=j['spec']['phase'],point=j['spec']['index'],tasks=j['spec']['tasks'],seconds=now-j['start']) for key,j in active.items()},coverage=counts,remaining_source_QP_bundles=len(bundles),prefetch_bundles=len(window),cache_wait_seconds=cache_wait,new_state_points=completed,state_points_per_second=completed/(now-begin),time=now))
    if stopping and not active:break
    time.sleep(2)
  except Exception as error:
   dump(ROOT/'audits/evaluation_queue_failure.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),active={key:dict(PID=j['p'].pid,GPU=j['gpu'],spec=j['spec']) for key,j in active.items()},time=time.time()))
   for j in active.values():j['p'].wait();j['log'].close();j['cache_lease'].close();j['gpu_lease'].close()
   raise

if __name__=='__main__':worker()
