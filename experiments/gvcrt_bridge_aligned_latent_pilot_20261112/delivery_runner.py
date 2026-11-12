from util import *
with lock('delivery_runner.lock'):
 while not (ROOT/'receipts/evaluation.complete.json').exists() or not (ROOT/'receipts/media64.complete.json').exists():heartbeat('delivery','WAITING_FOR_FIXED_EVALUATION_AND_64FRAME_MEDIA');time.sleep(5)
 while not (ROOT/'receipts/evaluation.complete.json').exists():heartbeat('delivery','WAITING_FOR_EVALUATION_AND_FREE_GPU3');time.sleep(5)
 if not (ROOT/'audits/runtime_profile.json').exists():
  h=(ROOT/'logs/runtime_profile.log').open('a');p=subprocess.Popen([PYTHON,'-B','profile_runtime.py'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='3'),stdout=h,stderr=subprocess.STDOUT);event('PROFILE_STARTED',child_pid=p.pid,physical_gpu=3)
  while p.poll() is None:heartbeat('delivery','RUNTIME_PROFILE',child_pid=p.pid);time.sleep(5)
  h.close();assert p.returncode==0,'PROFILE_FAILED'
 while not (ROOT/'receipts/evaluation.complete.json').exists():heartbeat('delivery','WAITING_FOR_EVALUATION');time.sleep(5)
 while not (ROOT/'finalize.py').exists():heartbeat('delivery','WAITING_FOR_DELIVERY_IMPLEMENTATION');time.sleep(5)
 h=(ROOT/'logs/finalize.log').open('a');p=subprocess.Popen([PYTHON,'-B','finalize.py'],cwd=ROOT,stdout=h,stderr=subprocess.STDOUT);event('FINALIZE_STARTED',child_pid=p.pid)
 while p.poll() is None:heartbeat('delivery','WRITING_REPORT_FIGURES_VIDEOS',child_pid=p.pid);time.sleep(5)
 h.close();assert p.returncode==0,'FINALIZE_FAILED';heartbeat('delivery','LOCAL_COMPLETE')
