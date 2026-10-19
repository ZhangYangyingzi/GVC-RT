"""Bounded, receipt-driven fixed 39-point evaluation queue."""
import json,os,subprocess,sys,time,hashlib,shlex
from pathlib import Path
R=Path(__file__).resolve().parent;PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
# conservative measured reservation from v6.35 complete jobs, plus margin
RESERVE_MIB=5600;GPU_ORDER=[7,6,4,5]
def read(p):return json.load(open(p))
def gpu_free():
 try:
  out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,stderr=subprocess.DEVNULL)
  return {int(a.strip()):int(b.strip()) for a,b in (x.split(',') for x in out.splitlines())}
 except Exception:return {}
def valid_receipt(t):
 name=f'{t["stage"]}_{t["dataset"]}_{t["video"]}_{t["method"]}_{t["qp"]}'
 p=R/'tasks'/f'{name}.json'
 if not p.exists():return False
 try:
  d=read(p);return d.get('status')=='PASS' and d.get('task')==t and Path(d['result']['bitstream_path']).is_file() and Path(d['result']['feature_path']).is_file()
 except Exception:return False
def main():
 tasks=read(R/'task_manifest.json')['tasks'];pending=[t for t in tasks if not valid_receipt(t)]
 R.joinpath('logs').mkdir(exist_ok=True);log=R/'logs/evaluation_queue.log'
 procs={}
 while pending or procs:
  free=gpu_free()
  # one process per selected card; launch only after checking idle task and memory reservation.
  for g in GPU_ORDER:
   if not pending or g in procs:continue
   if free.get(g,0)<RESERVE_MIB:continue
   t=pending.pop(0);arg=json.dumps(t,separators=(',',':'))
   cmd=[PYTHON,'-u',str(R/'evaluate35.py'),'--task',arg,'--gpu',str(g)]
   env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_DEVICE_ORDER='PCI_BUS_ID')
   # evaluate script selects CUDA_VISIBLE_DEVICES internally before importing torch.
   out=open(R/'logs'/f'eval_{t["dataset"]}_{t["video"]}_{t["qp"]}.log','ab',buffering=0)
   p=subprocess.Popen(cmd,cwd=R,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
   procs[g]=(p,t,out,time.time());
   with log.open('a') as f:f.write(json.dumps({'event':'START','gpu':g,'pid':p.pid,'reserved_MiB':RESERVE_MIB,'free_before_MiB':free[g],'task':t,'time':time.time()})+'\n')
  for g,(p,t,out,started) in list(procs.items()):
   rc=p.poll()
   if rc is None:continue
   out.close();name=f'{t["stage"]}_{t["dataset"]}_{t["video"]}_{t["method"]}_{t["qp"]}'
   with log.open('a') as f:f.write(json.dumps({'event':'EXIT','gpu':g,'pid':p.pid,'exit_code':rc,'task':t,'elapsed':time.time()-started,'time':time.time()})+'\n')
   if rc!=0 or not valid_receipt(t):
    # One controlled retry; the task remains the same and its lock/output path is exclusive.
    attempts=R/'tasks'/f'{name}.attempts.json'
    n=json.load(open(attempts)).get('attempts',0) if attempts.exists() else 0
    n+=1;json.dump({'attempts':n,'last_exit_code':rc,'task':t},open(attempts,'w'),indent=2)
    if n<=2:pending.append(t)
    else:
     with log.open('a') as f:f.write(json.dumps({'event':'FAILED_TERMINAL','task':t,'attempts':n,'time':time.time()})+'\n')
   del procs[g]
  status={'status':'RUNNING','expected':len(tasks),'completed':sum(valid_receipt(t) for t in tasks),'pending':len(pending),'active':{str(g):{'pid':p.pid,'task':t} for g,(p,t,_,_) in procs.items()},'updated_unix':time.time()}
  json.dump(status,open(R/'evaluation_status.json','w'),indent=2)
  if pending and not procs and not any(free.get(g,0)>=RESERVE_MIB for g in GPU_ORDER):time.sleep(30)
  elif procs:time.sleep(3)
  elif pending:time.sleep(5)
 status={'status':'PASS' if all(valid_receipt(t) for t in tasks) else 'BLOCKED','expected':len(tasks),'completed':sum(valid_receipt(t) for t in tasks),'pending':0,'updated_unix':time.time()}
 json.dump(status,open(R/'evaluation_status.json','w'),indent=2)
if __name__=='__main__':main()
