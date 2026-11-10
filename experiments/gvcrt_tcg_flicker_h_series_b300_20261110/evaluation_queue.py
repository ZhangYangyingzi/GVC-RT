"""One-point GPU3 worker: cache first; GPU lease only during inference/scoring."""
from common_s2 import *

def submit(model,step,phase='development'):
 key=f'{phase}_{model}_{step:04d}';dest=ROOT/'evaluation_queue/jobs'/f'{key}.json'
 if not dest.exists():dump(dest,dict(model=model,step=step,phase=phase,created=time.time()))
 return key

def progress(model,step,phase='development'):
 return len(list((ROOT/'evaluation'/phase/model/f'step_{step:04d}/points').glob('*.json')))

def worker():
 env()
 from pipeline import ready,request,positions
 from cache_codec import cache_paths
 with lock('locks/evaluation_queue.lock'):
  while not (ROOT/'evaluation_queue/STOP').exists():
   jobs=sorted((ROOT/'evaluation_queue/jobs').glob('*.json'),key=lambda p:p.stat().st_mtime);acted=False
   for path in jobs:
    job=load(path);model,step,phase=job['model'],job['step'],job['phase'];videos=load(ROOT/f'manifests/{"development" if phase=="development" else "test"}_videos.json');points=[(v,q) for v in videos for q in [0,5,9]];todo=[]
    if model.endswith('_selected'):step=load(ROOT/'selection.json')['groups'][model[:-9]]['selected']['step']
    for index,(v,q) in enumerate(points):
     dest=ROOT/'evaluation'/phase/model/f'step_{step:04d}/points'/f'{v["dataset"]}__{v["sequence"]}_qp{q}.json'
     if not dest.exists():todo.append((index,v,q))
    if not todo:
     dump(ROOT/'audits/evaluation_queue_done'/path.name,dict(status='PASS',**job,points=len(points),time=time.time()));continue
    index,v,q=todo[0];row=v.get('source_row',v);request(row,q,False)
    pos=min(positions().values());rows=plan_rows();ahead=sum(ready(r,r['external_qp'],True) for r in rows[pos:min(400,pos+32)])
    if pos<400 and ahead<16:continue
    if not ready(row,q,False):continue
    try:
     with lock('locks/GPU3.lock'):
      if not ready(row,q,False):continue
      # Block eviction from the final readiness check until the child finishes.
      key,p,rp=cache_paths(row,q,False)
      with lock('locks/lease_'+key+'.lock',blocking=True):
       if not p.exists():continue
       proof=load(rp);assert sha(p)==proof['cache_sha256'];args=[PYTHON,'-B','-u','evaluation.py','--models',model,'--step',str(job['step']),'--phase',phase,'--rank','0','--job-index',str(index)]
       log=(ROOT/'logs/evaluation_GPU3.log').open('ab');child=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',H_CACHE_LEASE_HELD=key),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
       while child.poll() is None:dump(ROOT/'evaluation_queue/progress.json',dict(status='RUNNING',job=job,point=index,PID=child.pid,physical_GPU=3,time=time.time()));time.sleep(2)
       code=child.returncode;log.close();assert code==0,('EVALUATION_POINT_FAILED',job,index,code)
      acted=True;break
    except BlockingIOError:continue
   if not acted:time.sleep(2)
if __name__=='__main__':
 try:worker()
 except Exception as e:
  import traceback
  dump(ROOT/'audits/evaluation_queue_failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
