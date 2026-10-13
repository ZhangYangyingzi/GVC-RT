"""Screen-backed launch with explicit session and command provenance."""
from audit31 import *
import subprocess
if __name__=='__main__':
 session='gvcrt_v631_'+time.strftime('%Y%m%d_%H%M%S')
 argv=[PYTHON,'-B','-u',str(ROOT/'run_all.py'),'--resume']
 subprocess.run(['screen','-L','-Logfile',str(ROOT/'logs/driver.log'),'-dmS',session,*argv],check=True,cwd=REPO)
 dump(ROOT/'launch.json',dict(session=session,command=argv,cwd=str(REPO),started=time.time(),persistent='screen',recovery='run_all.py --resume; single instance fcntl lock'))
 print('LAUNCHED',session)
