"""Wait for the fixed queue, then audit/report/plot locally. Never publish."""
from util import *
import time,subprocess,fcntl
def main():
 lock=(P/'prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  r=load(P/'run_status.json')
  if r['status']=='FAILED':raise RuntimeError(r)
  if r['status']=='COMPUTE_COMPLETE':break
  time.sleep(10)
 env=dict(os.environ,PYTHONPATH=str(P)+os.pathsep+str(ORIGINAL),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 subprocess.run([PYTHON,'-B','-u',str(P/'finalize.py')],env=env,cwd=P,check=True)
 subprocess.run(['/data1/anaconda3_new/anaconda_program/bin/python','-B','-u',str(P/'plot.py')],env=env,cwd=P,check=True)
 dump(P/'delivery_status.json',dict(status='READY',training_updates=0,tasks=40,audits='PASS',report=str(P/'REPORT.md'),publication_receipt='server-local COMPLETE.json is written after verified push'))
 print('READY_FOR_PUBLICATION',flush=True)
if __name__=='__main__':main()
