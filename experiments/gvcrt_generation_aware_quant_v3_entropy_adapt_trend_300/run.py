"""Fixed default GPU queue; resumes committed checkpoints, never extends 300 updates."""
from util import *
import subprocess,concurrent.futures,threading,queue

def gpu_free(gpu):
 lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
 row=next(s.split(',') for s in lines if int(s.split(',')[0])==gpu);uuid=row[1].strip();free=int(row[2])
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 return uuid not in apps and free>=12000,free

def run(command,gpu,key):
 out=P/'server_artifacts';lock=(out/f'gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 while True:
  available,free=gpu_free(gpu)
  append(P/'resource_events.jsonl',dict(event='WAIT_OR_LAUNCH',key=key,gpu=gpu,free_MiB=free,available=available,time=time.time()))
  if available:break
  time.sleep(10)
 env=dict(os.environ,PYTHONPATH=str(P)+os.pathsep+str(ORIGINAL)+os.pathsep+os.environ.get('PYTHONPATH',''),CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (out/'logs'/f'{key}.log').open('a') as log:
  child=subprocess.Popen([PYTHON,'-B','-u',*map(str,command)],cwd=P,stdout=log,stderr=subprocess.STDOUT,env=env)
  append(P/'resource_events.jsonl',dict(event='START',key=key,gpu=gpu,PID=child.pid,time=time.time(),argv=command))
  code=child.wait()
 append(P/'resource_events.jsonl',dict(event='EXIT',key=key,gpu=gpu,PID=child.pid,time=time.time(),exit_code=code))
 assert code==0,(key,code)
 print('TASK_PASS',key,flush=True)

def evaluate(method,step,source,qp):
 p=P/'server_artifacts/evaluation'/f'{method}_s{step}_{source}_q{qp}.json'
 if p.exists() and load(p).get('status')=='PASS':return
 run([str(P/'evaluate.py'),'--branch',method,'--step',str(step),'--source',source,'--qp',str(qp),'--gpu','7'],7,f'eval_{method}_{step}_{source}_{qp}')

def supplements():
 decisions=load(P/'supplement_selection.json') if (P/'supplement_selection.json').exists() else []
 for source in ['UVal0','UVal1','ParkScene','Jockey']:
  for iteration in range(3):
   rows=[load(p) for p in (P/'server_artifacts/evaluation').glob(f'*_s300_{source}_q*.json')];by={b:[r for r in rows if r['branch']==b] for b in ['N0','N1','Q0','Q1']};chosen={}
   for method,ref in [('N1','N0'),('Q1','Q0'),('Q1','N1')]:
    a,b=by[method],by[ref];alo,ahi=min(r['actual_bpp'] for r in a),max(r['actual_bpp'] for r in a);blo,bhi=min(r['actual_bpp'] for r in b),max(r['actual_bpp'] for r in b)
    if max(alo,blo)<min(ahi,bhi):continue
    lower,higher=(method,ref) if ahi<=blo else (ref,method)
    for branch,direction in [(lower,'higher_rate'),(higher,'lower_rate')]:
     used={r['external_qp'] for r in by[branch]}
     if len(used-{2,7})>=3:continue
     options=[q for q in ([9,8,6,5,4,3,1,0] if direction=='higher_rate' else [0,1,3,4,5,6,8,9]) if q not in used and (q>max(used) if direction=='higher_rate' else q<min(used))]
     if options:chosen.setdefault(branch,(options[0],direction))
   if not chosen:break
   evidence={b:[dict(qp=r['external_qp'],actual_bpp=r['actual_bpp']) for r in rs] for b,rs in by.items()}
   for branch,(q,direction) in chosen.items():
    decisions.append(dict(sequence=source,method=branch,QP=q,direction=direction,rate_only_evidence=evidence,iteration=iteration));dump(P/'supplement_selection.json',decisions)
    evaluate(branch,300,source,q)
 dump(P/'supplement_selection.json',decisions)

def evaluate_after_training():
 # Fixed independent tasks, using released training cards while GPU7 is occupied.
 # One worker and one shared process lock per physical GPU; no extra tasks/steps.
 tasks=queue.Queue()
 for step in (0,100,300):
  for branch in ('N1','Q1'):
   for source in (['UVal0','UVal1','ParkScene','Jockey'] if step==300 else ['UVal0','UVal1']):
    for q in (2,7):
     path=P/'server_artifacts/evaluation'/f'{branch}_s{step}_{source}_q{q}.json'
     if not path.exists() or load(path)['status']!='PASS':tasks.put((branch,step,source,q))
 def worker(gpu):
  while True:
   try:branch,step,source,q=tasks.get_nowait()
   except queue.Empty:return
   path=P/'server_artifacts/evaluation'/f'{branch}_s{step}_{source}_q{q}.json'
   if not path.exists() or load(path)['status']!='PASS':
    run([str(P/'evaluate.py'),'--branch',branch,'--step',str(step),'--source',source,'--qp',str(q),'--gpu',str(gpu)],gpu,f'eval_{branch}_{step}_{source}_{q}')
   tasks.task_done()
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(worker,gpu) for gpu in (7,5,4)]
  for f in futures:f.result()

def main():
 lock=(P/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (P/'run_status.json').exists() and load(P/'run_status.json')['status']=='PUBLISHED':
  assert load(P/'publication_status.json')['status']=='PUBLISHED'
  print('ALREADY_PUBLISHED; no optimizer updates or evaluations added',flush=True);return
 while not (P/'preflight.json').exists():time.sleep(10)
 assert load(P/'preflight.json')['status']=='PASS' and load(P/'preflight.json')['signature']==signature()
 import reuse;reuse.main()
 if all(load(P/f'training_status_{b}.json')['status']=='COMPLETE' and load(P/f'training_status_{b}.json')['committed_updates']==300 for b in ('N1','Q1')):
  dump(P/'run_status.json',dict(status='EVALUATION',new_updates={'N1':300,'Q1':300},reused_updates={'N0':300,'Q0':300},signature=signature()))
  evaluate_after_training();supplements()
  dump(P/'run_status.json',dict(status='COMPUTE_COMPLETE',new_updates={'N1':300,'Q1':300},reused_updates={'N0':300,'Q0':300},signature=signature()))
  print('COMPUTE_COMPLETE',flush=True);return
 for b in ('N1','Q1'):
  assert load(P/f'training_status_{b}.json')['committed_updates']>=2
 dump(P/'run_status.json',dict(status='TRAINING_AND_EVALUATION',new_budget={'N1':300,'Q1':300},reused_updates={'N0':300,'Q0':300}))
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures={b:pool.submit(run,[str(P/'train.py'),'--branch',b,'--gpu',str(gpu),'--stop','300'],gpu,'train_'+b) for b,gpu in [('N1',4),('Q1',5)]}
  for step in (0,100,300):
   for branch in ('N1','Q1'):
    cp=P/'server_artifacts'/branch/f'step_{step}.pt'
    while not cp.exists():
     if futures[branch].done():futures[branch].result()
     time.sleep(10)
    for source in (['UVal0','UVal1','ParkScene','Jockey'] if step==300 else ['UVal0','UVal1']):
     for q in (2,7):evaluate(branch,step,source,q)
  for f in futures.values():f.result()
 supplements()
 dump(P/'run_status.json',dict(status='COMPUTE_COMPLETE',new_updates={'N1':300,'Q1':300},reused_updates={'N0':300,'Q0':300},signature=signature()))
 print('COMPUTE_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BlockingIOError:
  print('RUNNER_ALREADY_ACTIVE; preserve active run status',flush=True)
 except BaseException as e:
  dump(P/'run_status.json',dict(status='FAILED',error=repr(e),time=time.time()));raise
