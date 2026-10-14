"""Single persistent screen driver; no duplicate session and no old output writes."""
from io32 import *
import subprocess
def main():
 p=ROOT/'launch.json'
 if p.exists():
  old=load(p);pid=old.get('PID')
  if pid:
   try:
    os.kill(pid,0);cmd=Path(f'/proc/{pid}/cmdline').read_bytes();assert str(ROOT/'run_all.py').encode() in cmd
    print('ATTACHED',old);return
   except ProcessLookupError:pass
 session='gvcrt_v632_'+str(int(time.time()));env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 argv=[PYTHON,'-B','-u',str(ROOT/'run_all.py')];log=ROOT/'logs/driver.log'
 dump(p,dict(session=session,argv=argv,log=str(log),started_unix=time.time(),PID=None))
 subprocess.run(['screen','-dmS',session,'-L','-Logfile',str(log),*argv],cwd=REPO,env=env,check=True)
 print('SCREEN',session,flush=True)
if __name__=='__main__':main()
