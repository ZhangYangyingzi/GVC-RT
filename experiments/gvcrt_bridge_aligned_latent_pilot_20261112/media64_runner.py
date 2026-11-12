"""Extend the two predeclared visual cases to their existing 64-frame clips; no metric selection."""
from util import *

def child(name,gpu,args):
 st=ROOT/'receipts'/f'{name}.status.json';old=load(st) if st.exists() else {};pid=old.get('pid');proc=Path('/proc')/str(pid)
 if old.get('stage')=='COMPLETE':return
 if pid and proc.exists() and proc.joinpath('cwd').resolve()==ROOT:
  while proc.exists() and proc.joinpath('stat').read_text().split()[2]!='Z':heartbeat('media64','ATTACHED_RUNNING',child_task=name,child_pid=pid,child_physical_gpu=gpu);time.sleep(5)
  assert load(st)['stage']=='COMPLETE',('ATTACHED_MEDIA_FAILED',name);return
 h=(ROOT/'logs'/f'{name}.log').open('a');p=subprocess.Popen([PYTHON,'-B',*args],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=h,stderr=subprocess.STDOUT);event('MEDIA_TASK_STARTED',task=name,child_pid=p.pid,physical_gpu=gpu)
 while p.poll() is None:heartbeat('media64','RUNNING',child_task=name,child_pid=p.pid,child_physical_gpu=gpu);time.sleep(5)
 h.close();assert p.returncode==0,(name,p.returncode)
with lock('media64_runner.lock'):
 while not (ROOT/'checkpoints/S_current/freeze_audit.json').exists():heartbeat('media64','WAITING_FOR_FREE_GPU2');time.sleep(5)
 child('media64_cache',2,['media64.py','--phase','cache'])
 child('media64_render',2,['media64.py','--phase','render'])
 heartbeat('media64','COMPLETE')
