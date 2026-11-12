"""Resumable parent process; all committed receipts are verified by workers."""
from util import *
import signal,argparse

class Attached:
 def __init__(self,pid,name):self.pid=pid;self.name=name;self.returncode=None
 def poll(self):
  p=Path('/proc')/str(self.pid)
  if p.exists() and p.joinpath('stat').read_text().split()[2]!='Z':return None
  task='devcache0' if self.name in ['quickdev','devcache'] else self.name
  st=ROOT/'receipts'/f'{task}.status.json';self.returncode=0 if st.exists() and load(st).get('stage')=='COMPLETE' else 1;return self.returncode

def run():
 with lock('runner.lock'):
  check_protocol();procs={};handles={};failed={};started=time.time();old=load(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
  for n,r in old.get('children',{}).items():
   proc=Path('/proc')/str(r['pid'])
   if proc.exists() and proc.joinpath('cwd').resolve()==ROOT:
    procs[n]=(Attached(r['pid'],n),r['physical_gpu'],r['args']);handles[n]=open(os.devnull,'w')
  def launch(name,gpu,args):
   h=(ROOT/'logs'/f'{name}.log').open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));p=subprocess.Popen([PYTHON,'-B',*args],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True);procs[name]=(p,gpu,args);handles[name]=h;event('TASK_STARTED',task=name,child_pid=p.pid,physical_gpu=gpu)
  for shard,gpu in enumerate([1,2,3]):
   if 'traincache'+str(shard) not in procs:launch('traincache'+str(shard),gpu,['cache_data.py','--shard',str(shard)])
  if 'quickdev' not in procs and not all(cachepath(r['id'],5).with_suffix('.pt.json').exists() for r in manifest()['dev']):launch('quickdev',0,['cache_data.py','--shard','0','--dev','--quick'])
  stage='CACHE_AND_PREFLIGHT';joint_started=False;pre_started=False;dev_started=False
  while True:
   for name,(p,gpu,args) in list(procs.items()):
    if p.poll() is not None:
     handles.pop(name).close();del procs[name]
     event('TASK_EXIT',task=name,returncode=p.returncode)
     if p.returncode:failed[name]=dict(returncode=p.returncode,log=str(ROOT/'logs'/f'{name}.log'))
   if (ROOT/'audits/preflight.json').exists() and not pre_started and (ROOT/'train.py').exists() and 'quickdev' not in procs and 'quickdev' not in failed:
    launch('aligner',0,['train.py','--phase','aligner']);pre_started=True;stage='CACHE_AND_ALIGNER_PRETRAIN'
   if pre_started and (ROOT/'checkpoints/aligner_0500.pt.json').exists() and not joint_started and (ROOT/'audits/calibration.json').exists():
    if not any(n.startswith('traincache') for n in procs) and 'aligner' not in procs:
     for gr,gpu in [('T_aligned',0),('U_unaligned',1),('S_current',2)]:launch(gr,gpu,['train.py','--phase','joint','--group',gr])
     joint_started=True;stage='JOINT_TRAINING'
   if not dev_started and not any(n.startswith('traincache') for n in procs):
    launch('devcache',3,['cache_data.py','--shard','0','--dev']);dev_started=True
   if joint_started and not any(n in procs for n in ['T_aligned','U_unaligned','S_current']) and (ROOT/'evaluate.py').exists() and 'evaluation' not in procs and not (ROOT/'receipts/evaluation.complete.json').exists() and 'evaluation' not in failed:
    launch('evaluation',0,['evaluate.py']);stage='EVALUATING'
   statuses={p.stem:load(p) for p in (ROOT/'receipts').glob('*.status.json')}
   result=dict(status='RUNNING',stage=stage,pid=os.getpid(),heartbeat=time.time(),started=started,children={n:dict(pid=p.pid,physical_gpu=gpu,args=args) for n,(p,gpu,args) in procs.items()},failed=failed,cache_completed=len(list((ROOT/'cache/train').glob('*.pt.json'))),dev_cache_completed=len(list((ROOT/'cache/dev').glob('*.pt.json'))),evaluation_completed=len(list((ROOT/'metrics').glob('*.complete.json'))),workers=statuses)
   if failed and not procs:result.update(status='STOPPED_WITH_ERRORS',stage='ERROR');dump(ROOT/'status.json',result);return 1
   if (ROOT/'receipts/evaluation.complete.json').exists() and not procs:result.update(status='LOCAL_COMPUTE_COMPLETE',stage='DELIVERY_PENDING');dump(ROOT/'status.json',result);return 0
   dump(ROOT/'status.json',result);time.sleep(5)

if __name__=='__main__':sys.exit(run())
