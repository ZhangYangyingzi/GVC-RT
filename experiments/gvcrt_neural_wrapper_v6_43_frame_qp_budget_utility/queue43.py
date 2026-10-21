from common43 import *
import subprocess,fcntl,argparse

def execute(plan):
 pending=[t for t in plan if not validate_result(t)];running={};failed=[]
 (ROOT/'logs').mkdir(exist_ok=True)
 while pending or running:
  free={int(a):int(b) for a,b in [line.split(',') for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]}
  for gpu in (6,7,5,4):
   if gpu in running or free[gpu]<18000:continue
   ready=[t for t in pending if t['kind']=='baseline' or validate_result(next(b for b in load(ROOT/'intervention_plan.json') if b['sequence']==t['sequence'] and b['kind']=='baseline'))]
   if not ready:continue
   t=ready[0];pending.remove(t);name=key(t).replace('/','_');log=(ROOT/'logs'/f'{name}.log').open('a')
   args=[PYTHON,'-B','-u',str(ROOT/'evaluate43.py'),'--sequence',t['sequence'],'--variant',t['variant'],'--gpu',str(gpu)]
   p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
   running[gpu]=(p,log,t);append(ROOT/'logs/queue.jsonl',dict(event='START',task=key(t),gpu=gpu,free_MiB=free[gpu],PID=p.pid,time=time.time()));print('START',gpu,key(t),flush=True)
  for gpu,(p,log,t) in list(running.items()):
   if p.poll() is not None:
    log.close();del running[gpu];append(ROOT/'logs/queue.jsonl',dict(event='EXIT',task=key(t),gpu=gpu,exit_code=p.returncode,time=time.time()));print('EXIT',gpu,key(t),p.returncode,flush=True)
    if p.returncode or not validate_result(t):failed.append(t)
  if failed:
   # Do not terminate any active process. Drain current tasks before surfacing failure.
   pending=[]
  if pending or running:time.sleep(2)
 assert not failed,failed

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','impulse','transfer']);a=p.parse_args()
 with (ROOT/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  tasks=load(ROOT/'intervention_plan.json')
  if a.phase=='smoke':
   assert validate_result(next(t for t in tasks if t['sequence']=='ParkScene' and t['kind']=='baseline'))
   execute([t for t in tasks if t['sequence']=='ParkScene' and t['variant'] in ('t08_qplus2','t08_qminus2')])
  else:
   assert all(validate_result(t) for t in tasks if t['sequence']=='ParkScene' and t['variant'] in ('baseline','t08_qplus2','t08_qminus2'))
   execute(tasks if a.phase=='impulse' else load(ROOT/'transfer_plan.json'))
 print('QUEUE_COMPLETE',a.phase,flush=True)
