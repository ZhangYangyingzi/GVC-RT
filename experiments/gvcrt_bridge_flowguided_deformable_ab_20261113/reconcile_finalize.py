"""Resume only verified offline metric corrections and final delivery, without any training."""
from util import *
with lock('reconcile_finalize.lock'):
 children=[]
 for gpu in range(4):
  log=(ROOT/'logs'/f'fix_metrics{gpu}.log').open('a');p=subprocess.Popen([PYTHON,'-B','fix_native_metrics.py','--shard',str(gpu)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT);children.append((p,log,gpu))
 while any(p.poll() is None for p,_,_ in children):
  for p,_,gpu in children:
   if p.poll() not in [None,0]:raise RuntimeError(('METRIC_CORRECTION_FAILED',gpu,p.pid,p.returncode))
  count=len(list((ROOT/'audits/native_validity_correction/points').glob('*.json')));status=dict(stage='CORRECTING_NATIVE_METRIC_VALIDITY',committed_updates={'A_zero_base':300,'B_flow_guided':300},corrected_points=count,total_points=72,pid=os.getpid(),children=[dict(pid=p.pid,physical_gpu=gpu,returncode=p.poll()) for p,_,gpu in children],heartbeat=time.time(),local_time=stamp());dump(ROOT/'status.json',status);heartbeat('reconcile_finalize','CORRECTING_NATIVE_METRIC_VALIDITY',corrected_points=count,children=status['children']);time.sleep(10)
 for p,h,gpu in children:assert p.wait()==0;h.close()
 subprocess.run([PYTHON,'-B','fix_native_metrics.py','--finish'],cwd=ROOT,check=True)
 heartbeat('reconcile_finalize','FINALIZING_DELIVERY')
 log=(ROOT/'logs/finalize.log').open('a');p=subprocess.Popen([PYTHON,'-B','finalize.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 while p.poll() is None:
  dump(ROOT/'status.json',dict(stage='FINALIZING_DELIVERY',committed_updates={'A_zero_base':300,'B_flow_guided':300},evaluated_points=150,evaluated_frames=9600,pid=os.getpid(),finalizer_pid=p.pid,heartbeat=time.time(),local_time=stamp()));time.sleep(10)
 assert p.returncode==0,('FINALIZER_FAILED',p.pid,p.returncode)
 heartbeat('reconcile_finalize','LOCAL_COMPLETE_GITHUB_PENDING')
