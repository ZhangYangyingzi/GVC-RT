"""Idempotent detached launch; never replace an active scheduler or delete locks."""
from io21 import *
import fcntl

def main():
 lock=(ROOT/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s=load(ROOT/'pipeline_status.json');pid=s.get('PID')
 if pid:
  try:
   cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes();stat=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()
   if stat[0]!='Z' and str(ROOT/'run_pipeline.py').encode() in cmd:print('ALREADY RUNNING',pid);return
  except OSError:pass
 if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json')['status']=='PASS' and load(ROOT/'publication_status.json')['status']=='PASS':print('ALREADY COMPLETE',load(ROOT/'publication_status.json')['commit']);return
 frozen();env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');log=(ROOT/'logs/run_pipeline.log').open('a');argv=[PYTHON,'-B','-u',str(ROOT/'run_pipeline.py')];p=subprocess.Popen(argv,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);dump(ROOT/'audits/resume_launch.json',dict(PID=p.pid,argv=argv,time=time.time(),detached_session=True,existing_locks_preserved=True));print('LAUNCHED',p.pid)
if __name__=='__main__':main()
