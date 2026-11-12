"""Persistent resume scheduler. Active own workers are attached; completed jobs are skipped."""
from util import *
import signal
GROUPS=['S_current','U_unaligned','T_aligned']
def alive(pid):
 p=Path('/proc')/str(pid)
 try:return p.exists() and p.joinpath('stat').read_text().split()[2]!='Z' and p.joinpath('cwd').resolve()==ROOT
 except (FileNotFoundError,ProcessLookupError):return False
class Attached:
 def __init__(self,pid,name):self.pid=pid;self.name=name;self.returncode=None
 def poll(self):
  if alive(self.pid):return None
  task='devcache0' if self.name in ['quickdev','devcache'] else self.name;st=ROOT/'receipts'/f'{task}.status.json';self.returncode=0 if st.exists() and load(st).get('stage') in ['COMPLETE','LOCAL_COMPLETE'] else 1;return self.returncode

def run():
 if (ROOT/'receipts/local_delivery.complete.json').exists():event('RUNNER_NO_OP_ALREADY_COMPLETE');return 0
 with lock('runner.lock'):
  check_protocol();procs={};handles={};failed={};start=time.time();old=load(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
  for name,r in old.get('children',{}).items():
   if alive(r['pid']):procs[name]=(Attached(r['pid'],name),r.get('physical_gpu'),r['args']);handles[name]=open(os.devnull,'w')
  def launch(name,gpu,args):
   if name in procs or name in failed:return
   h=(ROOT/'logs'/f'{name}.log').open('a');env=dict(os.environ)
   if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
   p=subprocess.Popen([PYTHON,'-B',*args],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True);procs[name]=(p,gpu,args);handles[name]=h;event('TASK_STARTED',task=name,child_pid=p.pid,physical_gpu=gpu)
  while True:
   for name,(p,gpu,args) in list(procs.items()):
    code=p.poll()
    if code is not None:
     handles.pop(name).close();del procs[name];event('TASK_EXIT',task=name,returncode=code)
     if code:failed[name]=dict(returncode=code,log=str(ROOT/'logs'/f'{name}.log'))
   busy={g for p,g,a in procs.values() if g is not None}
   for shard,gpu in enumerate([1,2,3]):
    wanted=manifest()['train'][shard::3]
    if gpu not in busy and not all(cachepath(r['id']).with_suffix('.pt.json').exists() for r in wanted):launch('traincache'+str(shard),gpu,['cache_data.py','--shard',str(shard)]);busy.add(gpu)
   qdev=all(cachepath(r['id'],5).with_suffix('.pt.json').exists() for r in manifest()['dev'])
   if not qdev and 0 not in busy:launch('quickdev',0,['cache_data.py','--shard','0','--dev','--quick']);busy.add(0)
   pre_done=(ROOT/'checkpoints/aligner_0500.pt.json').exists() and (ROOT/'metrics/aligner_0500.csv').exists()
   if qdev and not pre_done and 0 not in busy:launch('aligner',0,['train.py','--phase','aligner']);busy.add(0)
   devrows=manifest()['dev']+load(ROOT/'additional_development_manifest.json')['dev'];dev_done=all(cachepath(r['id'],q).with_suffix('.pt.json').exists() for r in devrows for q in [0,5,9])
   if not dev_done and 3 not in busy:launch('devcache',3,['cache_data.py','--shard','0','--dev']);busy.add(3)
   checks=all((ROOT/'audits'/p).exists() and load(ROOT/'audits'/p)['status']=='PASS' for p in ['preflight.json','extended_checks.json'])
   if pre_done and checks and (ROOT/'audits/calibration.json').exists():
    for g,gpu in [('T_aligned',0),('U_unaligned',1),('S_current',2)]:
     if gpu not in busy and not (ROOT/'checkpoints'/g/'freeze_audit.json').exists():launch(g,gpu,['train.py','--phase','joint','--group',g]);busy.add(gpu)
   complete=all((ROOT/'checkpoints'/g/'freeze_audit.json').exists() for g in GROUPS)
   if complete and dev_done and not (ROOT/'receipts/evaluation.complete.json').exists() and not busy.intersection([0,1,2]):launch('evaluation',None,['evaluate.py'])
   if complete and dev_done and not (ROOT/'receipts/local_delivery.complete.json').exists():
    st=ROOT/'receipts/delivery.status.json'
    if not st.exists() or not alive(load(st)['pid']):launch('delivery',None,['delivery_runner.py'])
   workers={p.stem:load(p) for p in (ROOT/'receipts').glob('*.status.json')};groups={}
   for g in GROUPS:
    w=workers.get(g+'.status',{});groups[g]=dict(committed_updates=300 if (ROOT/'checkpoints'/g/'freeze_audit.json').exists() else w.get('committed_updates',0),status=w.get('stage','NOT_STARTED'),pid=w.get('pid'),physical_gpu=w.get('physical_gpu'))
   result=dict(status='RUNNING',phase='EVALUATING_OR_DELIVERING' if complete else 'JOINT_TRAINING' if pre_done else 'PRETRAIN_AND_CACHE',pid=os.getpid(),heartbeat=time.time(),started=start,aligner_committed_updates=500 if pre_done else workers.get('aligner.status',{}).get('committed_updates',0),groups=groups,cache_completed=len(list((ROOT/'cache/train').glob('*.pt.json'))),dev_cache_completed=len(list((ROOT/'cache/dev').glob('*.pt.json'))),evaluation_completed=len(list((ROOT/'metrics').glob('*.complete.json'))),children={n:dict(pid=p.pid,physical_gpu=gpu,args=args) for n,(p,gpu,args) in procs.items()},workers=workers,failed=failed)
   if (ROOT/'receipts/local_delivery.complete.json').exists() and not procs:result.update(status='LOCAL_COMPLETE',phase='PUBLICATION_PENDING');dump(ROOT/'status.json',result);return 0
   if failed and not procs:result.update(status='STOPPED_WITH_ERRORS');dump(ROOT/'status.json',result);return 1
   dump(ROOT/'status.json',result);time.sleep(5)

if __name__=='__main__':sys.exit(run())
