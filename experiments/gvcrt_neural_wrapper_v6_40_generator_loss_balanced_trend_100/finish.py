"""Wait for all fixed tasks, then validate, plot, index and publish."""
from io32 import *
import subprocess,fcntl
if __name__=='__main__':
 lock=(ROOT/'finish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (ROOT/'publication_status.json').exists() and load(ROOT/'publication_status.json').get('push_success') and (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json')['status']=='PASS':sys.exit(0)
 while True:
  s=load(ROOT/'run_status.json') if (ROOT/'run_status.json').exists() else {}
  if s.get('status') in ('COMPUTE_COMPLETE','COMPLETE','PARTIAL','BLOCKED_PREFLIGHT'):break
  time.sleep(10)
 assert s['status'] in ('COMPUTE_COMPLETE','COMPLETE'),s
 for name in ['summarize.py','plot_results.py','index_artifacts.py','publish.py']:
  print('RUN',name,flush=True);subprocess.run([PYTHON,'-B','-u',str(ROOT/name)],cwd=ROOT,check=True)
 print('ALL_FINISHED',flush=True)
