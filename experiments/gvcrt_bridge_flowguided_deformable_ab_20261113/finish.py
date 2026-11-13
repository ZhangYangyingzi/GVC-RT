from util import *
with lock('finish.lock'):
 if not (ROOT/'audits/performance.json').exists():
  h=(ROOT/'logs/performance.log').open('a');p=subprocess.Popen([PYTHON,'-B','performance.py'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=h,stderr=subprocess.STDOUT)
  while p.poll() is None:heartbeat('finish','PERFORMANCE',child_pid=p.pid);time.sleep(5)
  h.close();assert p.returncode==0,'PERFORMANCE_FAILED'
 while not (ROOT/'finalize.py').exists():heartbeat('finish','WAITING_FOR_FINALIZER');time.sleep(3)
 h=(ROOT/'logs/finalize.log').open('a');p=subprocess.Popen([PYTHON,'-B','finalize.py'],cwd=ROOT,stdout=h,stderr=subprocess.STDOUT)
 while p.poll() is None:heartbeat('finish','REPORT_FIGURES_VIDEOS',child_pid=p.pid);time.sleep(5)
 h.close();assert p.returncode==0,'FINALIZER_FAILED';heartbeat('finish','COMPLETE')
