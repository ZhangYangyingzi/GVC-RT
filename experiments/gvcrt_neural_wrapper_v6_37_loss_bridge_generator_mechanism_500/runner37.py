from io32 import *
import subprocess,fcntl,traceback

def alive(j):
 try:
  args=Path(f'/proc/{j["wrapper_PID"]}/cmdline').read_bytes().decode().split('\0')
  if time.time()-j.get('time',j.get('started',0))<15:return True
  return str(ROOT/'supervise37.py') in args or ('supervise37.py' in args and Path(f'/proc/{j["wrapper_PID"]}/cwd').resolve()==ROOT)
 except (OSError,KeyError):return False
def active():
 out=[]
 for p in (ROOT/'runtime/active').glob('*.json'):
  j=load(p)
  if alive(j):out.append(j)
 return out
def attempts(key):return [load(p) for p in (ROOT/'runtime/attempts').glob(f'{key}_*_exit.json')]
def success(key):
 prefix='D_' if key.startswith('D_') and key!='D_train' else 'replay_'
 p=ROOT/'tasks'/(key+'.json')
 return p.exists() and load(p).get('status')=='PASS'
def train_done():
 p=ROOT/'branches/D_basicDISTS_noP_500/training_status.json'
 return p.exists() and load(p).get('status')=='PASS' and load(p).get('committed_updates')==500
GPU_CACHE={}
def start(key,script,args,estimate,forced=None):
 jobs=active()
 if any(j['key']==key for j in jobs):return False
 if len(attempts(key))>=3:return False
 if time.time()-GPU_CACHE.get('time',0)>10:
  GPU_CACHE['gpus']={int(s.split(',')[0]):int(s.split(',')[1]) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()};GPU_CACHE['time']=time.time()
 gpus=GPU_CACHE['gpus']
 own={j['gpu'] for j in jobs};options=[g for g in (4,5,6,7) if g not in own and gpus[g]>=estimate+2000 and (forced is None or g==forced)]
 if not options:return False
 gpu=max(options,key=lambda g:gpus[g]);job=dict(key=key,script=script,args=args,gpu=gpu,GPU_free_before=gpus[gpu],estimate_MiB=estimate,time=time.time())
 with (ROOT/'logs/supervisors.log').open('a') as f:p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'supervise37.py'),json.dumps(job)],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 # Launch reservation is durable before another task can select the same card.
 dump(ROOT/'runtime/active'/f'{key}.json',dict(job,wrapper_PID=p.pid));return True

def main():
 lock=(ROOT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert load(ROOT/'audits/step0_gradient.json')['status']=='PASS'
 ts=load(ROOT/'evaluation/task_manifest.json')['source_tasks'];blocked=[]
 while True:
  jobs=active()
  if not train_done():start('D_train','train.py',[],6000)
  if train_done() and 'D_basicDISTS_noP_500' not in load(ROOT/'evaluation_config.json')['checkpoints']:
   subprocess.run([PYTHON,'-B',str(ROOT/'evaluate_D37.py'),'--prepare'],cwd=REPO,check=True)
  pilot='replay_ulong_0_q0';pilot_ok=success(pilot)
  if not pilot_ok:start(pilot,'replay37.py',['--task',json.dumps(ts[0])],18000)
  if pilot_ok:
   for t in ts:
    tag=f'{t["dataset"]}_{t["video"]}_q{t["qp"]}';key='replay_'+tag
    if not success(key):start(key,'replay37.py',['--task',json.dumps(t)],18000)
    if success(key) and train_done() and not success('D_'+tag):start('D_'+tag,'evaluate_D37.py',['--task',json.dumps(t)],14000)
  for key in ['D_train']+['replay_'+f'{t["dataset"]}_{t["video"]}_q{t["qp"]}' for t in ts]+['D_'+f'{t["dataset"]}_{t["video"]}_q{t["qp"]}' for t in ts]:
   if len(attempts(key))>=3 and not any(j['key']==key for j in active()) and not (train_done() if key=='D_train' else success(key)):
    if key not in blocked:blocked.append(key);error(dict(stage='runner',key=key,status='FAILED',reason='three failed attempts; see durable worker tracebacks'))
  replays=sum(success('replay_'+f'{t["dataset"]}_{t["video"]}_q{t["qp"]}') for t in ts);devals=sum(success('D_'+f'{t["dataset"]}_{t["video"]}_q{t["qp"]}') for t in ts)
  latest=ROOT/'branches/D_basicDISTS_noP_500/checkpoint_index.json';receipts=list((ROOT/'branches/D_basicDISTS_noP_500/receipts').glob('*.json'))
  dump(ROOT/'run_status.json',dict(status='RUNNING',D_committed_updates=max([int(p.stem) for p in receipts],default=0),D500=train_done(),replay_complete=replays,replay_planned=39,D_evaluation_complete=devals,D_evaluation_planned=39,active=active(),failed_or_blocked=blocked,updated=time.time()))
  if replays==39 and devals==39 and train_done() and not active():break
  if not active() and blocked:break
  time.sleep(45)
 while not (ROOT/'audits/intervention_input_hooks.json').exists():
  start('intervention_input_hooks','hook_verify37.py',[],10000)
  if len(attempts('intervention_input_hooks'))>=3 and not active():raise RuntimeError('Intervention-input hook verification failed three attempts')
  time.sleep(45)
 while active():time.sleep(30)
 with (ROOT/'logs/aggregate.log').open('a') as f:
  env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4');subprocess.run(['/data1/anaconda3_new/anaconda_program/bin/python','-B',str(ROOT/'aggregate37.py')],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 with (ROOT/'logs/integrity.log').open('a') as f:subprocess.run([PYTHON,'-B',str(ROOT/'integrity37.py')],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run([PYTHON,'-B',str(ROOT/'publish37.py')],cwd=REPO,check=True)
 dump(ROOT/'run_status.json',dict(status='COMPLETE',integrity=load(ROOT/'final_integrity.json'),publication=load(ROOT/'publication_status.json'),finished=time.time()))
if __name__=='__main__':
 try:main()
 except Exception:dump(ROOT/'runtime/runner_failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
