"""Matched independent100-update runs; GPU3 immutable-checkpoint evaluation queue."""
from common_s2 import *
import traceback
CHILDREN={}

def live(pid,script=None):
 try:
  if Path(f'/proc/{pid}/cwd').resolve()!=ROOT or Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':return False
  return script is None or script.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
 except (FileNotFoundError,ProcessLookupError,PermissionError):return False
class Adopted:
 def __init__(self,pid):self.pid=pid
 def poll(self):return None if live(self.pid) else 0

def start(name,args,gpus='0,1,2,3'):
 record=ROOT/'receipts/processes'/f'{name}.json'
 if record.exists():
  a=load(record)
  if live(a['PID'],args[0]):p=Adopted(a['PID']);CHILDREN[name]=p;return p
 log=(ROOT/'logs'/f'{name}.log').open('ab');p=subprocess.Popen([PYTHON,'-B','-u',*map(str,args)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();CHILDREN[name]=p;dump(record,dict(PID=p.pid,args=list(map(str,args)),GPUs=gpus,time=time.time()));event('process_started',process_name=name,PID=p.pid,GPUs=gpus);return p

def ensure_auxiliary():
 for failure in ['cache_failure.json','evaluation_queue_failure.json']:
  if (ROOT/'audits'/failure).exists():raise RuntimeError(('AUXILIARY_FAILURE',load(ROOT/'audits'/failure)))
 for name,lockpath,script in [('producer','locks/producer.lock','pipeline.py'),('evaluation_queue','locks/evaluation_queue.lock','evaluation_queue.py')]:
  try:
   with lock(lockpath):pass
  except BlockingIOError:continue
  start(name,[script])

def heartbeat(stage):
 from evaluation_queue import progress
 from pipeline import positions,ready
 a=dict(status='RUNNING',complete=False,current_stage=stage,groups={},cache=load(ROOT/'cache/progress.json') if (ROOT/'cache/progress.json').exists() else None,active_processes={n:dict(PID=p.pid,returncode=p.poll()) for n,p in CHILDREN.items()},GPU_allocation={g:i for i,g in enumerate(GROUPS)},GPU3='shared frozen codec/GT-flow producer and serialized evaluation',time=time.time())
 rows=plan_rows()
 for g in GROUPS:
  root=ROOT/g/'training';p=root/'progress.json';u=load(p)['committed_updates'] if p.exists() else 0;rec=[load(p) for p in (root/'receipts').glob('*.json')];state='MECHANICAL_CHECK_FAILED' if g==GROUPS[2] and (ROOT/'audits/H2_gate.json').exists() and load(ROOT/'audits/H2_gate.json')['status']!='PASS' else 'TRAINING_OR_WAITING_CACHE' if u<100 else 'TRAINING_FINISHED_EVALUATION_PENDING'
  if u==100 and progress(g,50)==progress(g,100)==96:state='DEVELOPMENT_COMPLETE'
  if state=='TRAINING_OR_WAITING_CACHE':state='TRAINING' if all(ready(r,r['external_qp'],True) for r in rows[u*4:u*4+4]) else 'WAITING_WINDOW'
  a['groups'][g]=dict(committed_updates=u,state=state,step50_points=progress(g,50),step100_points=progress(g,100),next_window=rows[u*4]['sample_key'] if u<100 and state!='MECHANICAL_CHECK_FAILED' else None,cache_wait_seconds=sum(r['cache_wait_seconds'] for r in rec),recent_seconds_per_update=sum(r['seconds'] for r in rec[-10:])/len(rec[-10:]) if rec else None)
 a['cache_request_queue_length']=len(list((ROOT/'cache/requests').glob('*.json')))
 a['evaluation_queue_length']=sum(not (ROOT/'audits/evaluation_queue_done'/p.name).exists() for p in (ROOT/'evaluation_queue/jobs').glob('*.json'))
 a['baseline_B300_points']=progress('B300',0)
 dump(ROOT/'status.json',a)
 last=ROOT/'readable_status.json'
 if not last.exists() or time.time()-load(last)['time']>=60:dump(last,a);event('heartbeat',stage=stage,groups=a['groups'],READY=None if a['cache'] is None else a['cache']['READY_training_windows'])

def wait_audit(name,script,audit,gpus):
 if (ROOT/audit).exists():assert load(ROOT/audit)['status']=='PASS';return
 p=start(name,[script],gpus)
 while p.poll() is None:heartbeat(name);time.sleep(5)
 assert p.poll()==0 and load(ROOT/audit)['status']=='PASS',('STAGE_FAILED',name,p.poll())

def main():
 env()
 with lock('locks/runner.lock'):
  # Preflight/calibration can be adopted from the active preparation turn.
  ensure_auxiliary();wait_audit('preflight','preflight.py','audits/preflight.json','0');wait_audit('calibration','calibrate.py','gradient_calibration.json','1')
  from freeze import main as freeze
  freeze();verify_sources();active=[g for g in GROUPS if g!=GROUPS[2] or load(ROOT/'audits/H2_gate.json')['status']=='PASS']
  smokes={}
  for g in active:
   if not (ROOT/f'audits/engineering_{g}.json').exists():smokes[g]=start('smoke_'+g,['train.py','--group',g,'--smoke'],str(GROUPS.index(g)))
  while any(p.poll() is None for p in smokes.values()):heartbeat('engineering');assert all(p.poll() in [None,0] for p in smokes.values());time.sleep(5)
  for g in active:assert load(ROOT/f'audits/engineering_{g}.json')['status']=='PASS'
  from evaluation_queue import submit,progress
  submit('B300',0)
  training={}
  if not (ROOT/'training_completion.json').exists():
   for g in active:
    p=ROOT/g/'training/progress.json';u=load(p)['committed_updates'] if p.exists() else 0
    if u<100:training[g]=start('training_'+g,['train.py','--group',g,'--until','100'],str(GROUPS.index(g)))
   while True:
    ensure_auxiliary();heartbeat('training_and_development')
    all_done=True
    for g in active:
     for s in [50,100]:
      if (ROOT/g/f'training/checkpoints/step_{s:04d}.pt').exists():submit(g,s)
     p=ROOT/g/'training/progress.json';u=load(p)['committed_updates'] if p.exists() else 0
     if u<100:
      all_done=False;assert g in training and training[g].poll() in [None],('GROUP_TRAINING_FAILED',g,u)
    if all_done:break
    time.sleep(5)
   dump(ROOT/'training_completion.json',dict(status='PASS',updates={g:100 if g in active else 0 for g in GROUPS},stop_reason={g:'maximum100' if g in active else 'H2_MECHANICAL_CHECK_FAILED' for g in GROUPS},time=time.time()))
  for g in active:
   for s in [50,100]:submit(g,s)
  while progress('B300',0)<96 or any(progress(g,s)<96 for g in active for s in [50,100]):ensure_auxiliary();heartbeat('development');time.sleep(10)
  from selection import select
  choice=select()
  final_models=['Original','A2000','B300']+[g+'_selected' for g in active if choice['groups'][g]['qualified']]
  if choice['any_qualified']:
   for model in final_models:submit(model,0,'final')
   while any(progress(m,choice['groups'][m[:-9]]['selected']['step'] if m.endswith('_selected') else 0,'final')<225 for m in final_models):ensure_auxiliary();heartbeat('final_evaluation');time.sleep(10)
   dump(ROOT/'final_test_status.json',dict(status='PASS',models=final_models,points_per_model=225,selection_SHA256=sha(ROOT/'selection.json'),time=time.time()))
  else:dump(ROOT/'final_test_status.json',dict(status='NOT_RUN_NO_QUALIFIED_CANDIDATE',new_points=0,reason='pre-registered conditional final test',time=time.time()))
  wait_audit('results','results.py','audits/results.json','0,1,2,3');wait_audit('media','media.py','video_manifest.json','0,1,2,3');wait_audit('integrity','integrity.py','final_integrity.json','0,1,2,3')
  p=start('publication_final',['publication.py','--final'])
  while p.poll() is None:heartbeat('publication');time.sleep(5)
  assert p.poll()==0 and load(ROOT/'publication_status.json')['phase']=='final_results' and load(ROOT/'publication_status.json')['remote_verified']
  dump(ROOT/'evaluation_queue/STOP',dict(time=time.time()));dump(ROOT/'cache/STOP',dict(time=time.time()))
  dump(ROOT/'status.json',dict(status='COMPLETE',complete=True,current_stage='complete',updates=load(ROOT/'training_completion.json')['updates'],experiment_directory=str(ROOT),publication=load(ROOT/'publication_remote_receipt.json'),time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as e:env();dump(ROOT/'audits/runner_failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
