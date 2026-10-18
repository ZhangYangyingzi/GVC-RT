"""Isolated, bounded, resumable v2 driver; never schedules physical GPUs0–3."""
from preflight import *
from data import load
from bootstrap import BRANCHES,V1
import fcntl
PY=sys.executable;CHILDREN=[]
def completed_training_valid():
 # Resume after training only when every bound signature and final model agrees.
 if not all((R/f'branches/{b}/integrity.json').exists() for b in BRANCHES):return False
 import torch
 from train import signature
 from bootstrap import SOURCE_SHA
 sig=signature()
 sig.update(beta=sha(R/'calibration/final.json'),rate_backend=sha(R/'rate_backend.py'),branch_config=sha(R/'config.json'),source_P=SOURCE_SHA,branch_implementation=sha(R/'bootstrap.py'),metric_dependencies=sha(R/'audits/metric_dependencies.json'),rate_audits=sha(R/'audits/rate_audit_index.json'))
 checked={}
 for b in BRANCHES:
  integrity=load(R/f'branches/{b}/integrity.json');assert integrity['status']=='PASS' and integrity['updates']==1000 and integrity['frozen_unchanged']
  index=load(R/f'branches/{b}/checkpoint_index.json')['1000']
  assert sha(index['path'])==index['sha256'] and sha(index['P_path'])==index['P_sha256']
  model=torch.load(index['P_path'],map_location='cpu',weights_only=False)
  assert model['signature']==sig and model['branch']==b and model['adaptation_step']==1000
  assert state_hash(model['P'])==index['module_sha256']
  receipts=[load(p) for p in sorted((R/f'branches/{b}/receipts').glob('*.json'))]
  assert len(receipts)==1000 and [r['effective_updates'] for r in receipts]==list(range(1,1001))
  checked[b]=dict(updates=1000,checkpoint=index,signature_unchanged=True)
 save('audits/recovery/completed_training_revalidation.json',dict(status='PASS',branches=checked,training_repeated=False))
 return True
def state(phase,**extra):save('status.json',dict(status='RUNNING',phase=phase,PID=os.getpid(),time=time.time(),**extra))
def command(name,argv,gpu=None):
 (R/'logs').mkdir(exist_ok=True);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONFAULTHANDLER='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
 if gpu is not None:env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 with (R/'logs/commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=argv,gpu=gpu,time=time.time()))+'\n')
 log=(R/'logs'/f'{name}.log').open('a');p=subprocess.Popen(argv,cwd=GLC,env=env,stdout=log,stderr=subprocess.STDOUT);CHILDREN.append(p);return p,log
def wait(p,log,name):
 while p.poll() is None:time.sleep(15)
 log.close()
 if p.returncode:raise RuntimeError(f'{name} exit {p.returncode}; logs/{name}.log')
def stage(name,args):
 p,l=command(name,[PY,'-B','-u',str(R/args[0]),*args[1:]]);wait(p,l,name)
def resources():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
 return {int(a):float(c)-max(2048,float(b)*.1) for a,b,c in (line.split(',') for line in raw.splitlines()) if int(a) in (4,5,6,7)}
def parallel(jobs,need=10000,max_workers=4,per_gpu=1):
 try:return _parallel(jobs,need,max_workers,per_gpu)
 except BaseException:
  # Each importing coordinator owns only the Popen objects it created.
  # Nested calibration failures must not leave untracked GPU workers behind.
  for p in CHILDREN:
   if p.poll() is None:p.terminate()
  for p in CHILDREN:
   if p.poll() is None:
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
  raise
def _parallel(jobs,need=10000,max_workers=4,per_gpu=1):
 jobs=list(jobs);running=[];wait_deadline=time.time()+3600;deadline=time.time()+172800;retries={};resource_requeues={}
 while jobs or running:
  assert time.time()<deadline,'48-hour stage limit'
  for w in list(running):
   p,l,name,args,g,started=w
   if p.poll() is None:continue
   running.remove(w);l.close()
   if p.returncode:
    tail=(R/'logs'/f'{name}.log').read_text(errors='replace')[-6000:]
    if 'out of memory' in tail.lower() and retries.get(name,0)<2:
     retries[name]=retries.get(name,0)+1;save(f'audits/OOM_{name}_{retries[name]}.json',dict(error=tail,gpu=g,time=time.time()));jobs.append((name,args));max_workers=max(1,max_workers-1);need*=1.15
    else:raise RuntimeError(f'{name} exit {p.returncode}; logs/{name}.log: {tail[-1000:]}')
  free=resources()
  for g,remaining in list(free.items()):
   owned=[w for w in running if w[4]==g]
   if remaining>=0 or not owned:continue
   w=max(owned,key=lambda w:w[5]);p,l,name,args,_,_=w
   if p.poll() is None:p.terminate();p.wait(timeout=30)
   l.close();running.remove(w);resource_requeues[name]=resource_requeues.get(name,0)+1
   save(f'audits/resource_guard_{name}_{time.time_ns()}.json',dict(action='release only owned worker, preserve checkpoint/receipts; others untouched',GPU=g,reserve_shortfall_MiB=-remaining,requeues=resource_requeues[name]))
   assert resource_requeues[name]<=3,f'{name}: reserve shortfall repeated after three requeues'
   jobs.append((name,args));max_workers=max(1,max_workers-1)
  free=resources()
  # Full task budget reserved for workers before allocation becomes visible.
  allocated=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
  mem={int(a):float(b) for a,b in (line.split(',') for line in allocated.splitlines()) if a.strip().isdigit() and b.strip().isdigit()}
  for p,l,n,a,g,t in running:free[g]-=max(0,need-mem.get(p.pid,0))
  for g in sorted(free,key=free.get,reverse=True)*per_gpu:
   if not jobs or len(running)>=max_workers or sum(w[4]==g for w in running)>=per_gpu or free[g]<need:continue
   name,args=jobs.pop(0);p,l=command(name,[PY,'-B','-u',str(R/args[0]),*args[1:],'--gpu',str(g)],g);running.append((p,l,name,args,g,time.time()));free[g]-=need;wait_deadline=time.time()+3600
  save('worker_status.json',dict(running=[dict(name=w[2],PID=w[0].pid,gpu=w[4]) for w in running],pending=len(jobs),need_MiB=need,time=time.time()))
  if jobs and not running:assert time.time()<wait_deadline,'GPU4–7 admission timeout after 3600s'
  time.sleep(15)
def main():
 lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert load(R/'publication_status.json')['status']=='PUBLISHED'
 if (R/'BLOCKED.json').exists():
  archive=R/f'audits/recovery/BLOCKED_before_resume_{time.time_ns()}.json';archive.parent.mkdir(parents=True,exist_ok=True);(R/'BLOCKED.json').rename(archive)
 if completed_training_valid():
  state('training_revalidated')
 else:
  prepare_and_train()
 state('publish_training');stage('publish_training',['publish.py','--phase','training'])
 state('baseline_reuse');stage('baseline_reuse',['reuse.py'])
 state('evaluation');from evaluation_stage import run
 done,expected=run(command,state,PY)
 state('report');stage('report',['report.py'])
 state('publication');stage('publish_final',['publish.py','--phase','final'])
 save('status.json',dict(status='COMPLETE',terminal=True,updates={b:1000 for b in BRANCHES},evaluation_completed=done,evaluation_expected=expected,publication=load(R/'publication_status.json')))
def prepare_and_train():
 if not (R/'audits/source_inventory.json').exists():state('local_source_preflight');stage('local_source_preflight',['preflight.py'])
 if not (R/'plans/shared.json').exists():state('data');stage('data',['data.py'])
 if not (R/'plans/rate_audit.json').exists():
  old=load(V1/'plans/calibration.json')['rows'];assert len(old)==40
  save('plans/rate_audit.json',dict(rows=old,selection='v1 fixed first two training-only windows per domain/q; unchanged 40 windows',source=str(V1/'plans/calibration.json'),source_sha256=sha(V1/'plans/calibration.json'),optimizer_updates=0))
 state('rate_pairs');parallel([(f'rate_pair_{i:02d}_{m}',['rate_audit.py','--index',str(i),'--method',m]) for i in range(40) for m in ('A_original','B_rgb_P','C_latent_P') if not (R/f'audits/rate_pairs/{i:02d}_{m}.json').exists()],need=5000,max_workers=4,per_gpu=2)
 state('component_gradients');parallel([(f'component_{i:02d}_{m}',['gradient_audit.py','--index',str(i),'--method',m]) for i in range(40) for m in ('B_rgb_P','C_latent_P') if not (R/f'audits/component_gradients/{i:02d}_{m}.json').exists()],need=8000,max_workers=4,per_gpu=2)
 state('rate_finalize');stage('rate_finalize',['finalize_rate.py'])
 state('pretraining_checks');parallel([('pretraining_checks',['training_checks.py'])],need=8000,max_workers=1)
 state('training');parallel([(b,['train.py','--branch',b]) for b in BRANCHES if not (R/f'branches/{b}/integrity.json').exists()],need=6000,max_workers=4,per_gpu=2)
if __name__=='__main__':
 try:main()
 except Exception:
  for p in CHILDREN:
   if p.poll() is None:p.terminate()
  error=traceback.format_exc();save('BLOCKED.json',dict(status='BLOCKED',error=error,time=time.time()));save('status.json',dict(status='BLOCKED',terminal=True,error=error));print(error,flush=True);sys.exit(1)
