"""Persistent actual subprocess supervisor; CPU cache compression and GPU queues are separate."""
from util import *
import signal
GROUPS=['A_zero_base','B_flow_guided']
def alive(pid):
 p=Path('/proc')/str(pid)
 try:return p.exists() and p.joinpath('stat').read_text().split()[2]!='Z' and p.joinpath('cwd').resolve()==ROOT
 except FileNotFoundError:return False
class Attached:
 def __init__(self,pid,task):self.pid=pid;self.task=task;self.returncode=None
 def poll(self):
  if alive(self.pid):return None
  p=ROOT/'receipts'/f'{self.task}.status.json';self.returncode=0 if p.exists() and load(p).get('stage')=='COMPLETE' else 1;return self.returncode

def run():
 with lock('runner.lock'):
  procs={};handles={};failed={};check_protocol()
  for task in ['cache_train','preflight',*GROUPS,'evaluation_queue','finish']:
   p=ROOT/'receipts'/f'{task}.status.json'
   if p.exists() and alive(load(p)['pid']):procs[task]=(Attached(load(p)['pid'],task),load(p).get('physical_gpu'));handles[task]=open(os.devnull,'w')
  def start(task,gpu,args):
   if task in procs or task in failed:return
   h=(ROOT/'logs'/f'{task}.log').open('a');env=dict(os.environ)
   if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
   p=subprocess.Popen([PYTHON,'-B',*args],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True);procs[task]=(p,gpu);handles[task]=h;event('TASK_STARTED',task=task,child_pid=p.pid,physical_gpu=gpu)
  while True:
   for task,(p,gpu) in list(procs.items()):
    if p.poll() is not None:
     handles.pop(task).close();del procs[task];event('TASK_EXIT',task=task,returncode=p.returncode)
     if p.returncode:failed[task]=dict(returncode=p.returncode,log=str(ROOT/'logs'/f'{task}.log'))
   cache_complete=ROOT/'receipts/cache_train.status.json'
   if not (cache_complete.exists() and load(cache_complete).get('stage')=='COMPLETE'):start('cache_train',2,['cache_train.py'])
   if not (ROOT/'audits/preflight.json').exists():start('preflight',0,['preflight.py'])
   ready=(ROOT/'audits/preflight.json').exists() and load(ROOT/'audits/preflight.json')['status']=='PASS' and (ROOT/'cache/evidence_stats.pt.json').exists()
   if ready and (ROOT/'train.py').exists():
    for g,gpu in zip(GROUPS,[0,1]):
     if not (ROOT/'checkpoints'/g/'completion.json').exists():start(g,gpu,['train.py','--group',g])
   if ready and (ROOT/'evaluate.py').exists() and not (ROOT/'receipts/evaluation_queue.complete.json').exists():start('evaluation_queue',None,['evaluate.py','--queue'])
   done=all((ROOT/'checkpoints'/g/'completion.json').exists() for g in GROUPS)
   if done and (ROOT/'receipts/evaluation_queue.complete.json').exists() and (ROOT/'finish.py').exists() and not (ROOT/'final_integrity.json').exists():start('finish',None,['finish.py'])
   workers={p.stem:load(p) for p in (ROOT/'receipts').glob('*.status.json')};gs={g:{'committed_updates':workers.get(g+'.status',{}).get('committed_updates',0),'status':workers.get(g+'.status',{}).get('stage','NOT_STARTED')} for g in GROUPS}
   st=dict(status='RUNNING',phase='TRAINING_AND_EVALUATION' if ready else 'PREFLIGHT_AND_CACHE',local_time=stamp(),heartbeat=time.time(),pid=os.getpid(),groups=gs,caches_completed=len(list((ROOT/'cache/decoded_refs').glob('*.npz.json'))),evaluations_completed=len(list((ROOT/'evaluation/points').glob('*.json'))),children={n:dict(pid=p.pid,physical_gpu=gpu) for n,(p,gpu) in procs.items()},workers=workers,failed=failed)
   if (ROOT/'final_integrity.json').exists() and not procs:st.update(status='LOCAL_COMPLETE',phase='PUBLICATION_PENDING');dump(ROOT/'status.json',st);return 0
   if failed and not procs:st.update(status='STOPPED_WITH_ERRORS');dump(ROOT/'status.json',st);return 1
   dump(ROOT/'status.json',st);time.sleep(5)
if __name__=='__main__':sys.exit(run())
