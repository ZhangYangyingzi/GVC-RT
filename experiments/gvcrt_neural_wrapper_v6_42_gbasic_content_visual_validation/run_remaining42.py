from common42 import *
import subprocess,fcntl
def main():
 (ROOT/'logs').mkdir(exist_ok=True)
 with (ROOT/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  first=load(ROOT/'tasks/hevc_b/Kimono1/q0/pair.json');assert first['status']=='PASS' and first['signature']==signature42()
  pending=[]
  for v in load(ROOT/'data_manifest.json'):
   for q in (0,5,9):
    receipt=ROOT/f"tasks/{v['dataset']}/{v['name']}/q{q}/pair.json"
    if receipt.exists():
     t=load(receipt);assert t['signature']==signature42()
     if t['status']=='PASS':continue
    pending.append((v,q))
  running={};failures=[]
  while pending or running:
   free={int(a):int(b) for a,b in [line.split(',') for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]}
   for gpu in (4,5,6,7):
    if pending and gpu not in running and free[gpu]>=16000:
     v,q=pending.pop(0);key=f"{v['dataset']}_{v['name']}_q{q}";log=(ROOT/'logs'/f'{key}.log').open('a');args=[PYTHON,'-B','-u',str(ROOT/'evaluate42.py'),'--dataset',v['dataset'],'--sequence',v['name'],'--qp',str(q),'--gpu',str(gpu)]
     p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
     running[gpu]=(p,log,key,time.time());append(ROOT/'logs/execution.jsonl',dict(event='START',key=key,gpu=gpu,PID=p.pid,time=time.time(),free_MiB=free[gpu]))
   for gpu,(p,log,key,start) in list(running.items()):
    code=p.poll()
    if code is not None:
     log.close();append(ROOT/'logs/execution.jsonl',dict(event='EXIT',key=key,gpu=gpu,PID=p.pid,start=start,end=time.time(),exit_code=code));del running[gpu]
     if code:failures.append(key)
     print('EXIT',key,code,flush=True)
   if pending or running:time.sleep(2)
  assert not failures,failures
  print('ALL_PAIRS_COMPLETE',flush=True)
if __name__=='__main__':main()
