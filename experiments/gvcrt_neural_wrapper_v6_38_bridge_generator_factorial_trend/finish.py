"""Wait for computation, validate, plot and publish; never launches extra training."""
from io32 import *
import subprocess,fcntl
if __name__=='__main__':
 l=(ROOT/'finish.lock').open('a');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (ROOT/'publication_status.json').exists() and load(ROOT/'publication_status.json').get('push_success') and (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json')['status']=='PASS':
  print('ALREADY_COMPLETE_AND_PUBLISHED');sys.exit(0)
 while True:
  p=ROOT/'run_status.json';s=load(p) if p.exists() else {}
  if s.get('status') in ('COMPUTE_COMPLETE','PARTIAL','BLOCKED_PREFLIGHT','COMPLETE'):break
  time.sleep(30)
 if s['status'] not in ('COMPUTE_COMPLETE','COMPLETE'):
  print('COMPUTATION_NOT_COMPLETE',s,flush=True);sys.exit(1)
 for cmd in ([PYTHON,'-B','-u',str(ROOT/'summarize.py')],['/data1/anaconda3_new/anaconda_program/bin/python','-B',str(ROOT/'plot_results.py')],[PYTHON,'-B','-u',str(ROOT/'publish.py')]):
  print('RUN',cmd,flush=True);subprocess.run(cmd,cwd=ROOT,check=True)
 print('ALL_FINISHED',flush=True)
