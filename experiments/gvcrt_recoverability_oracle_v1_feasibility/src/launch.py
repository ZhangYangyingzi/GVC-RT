"""Detach this process from the terminal; preserve the driver's actual exit code."""
from base import *
import fcntl
import traceback

if __name__=='__main__':
 with (ROOT/'manifests/driver.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  command=[PYTHON,'-B','-u',str(ROOT/'src/driver.py')]
  dump('manifests/driver_launch.json',dict(pid=os.getpid(),command=command,start=time.time(),cwd=str(REPO)))
  try:
   with (ROOT/'logs/driver_resumed.log').open('a') as log:
    p=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    dump('manifests/driver_process.json',dict(pid=p.pid,launcher_pid=os.getpid(),command=command))
    rc=p.wait()
   dump('logs/driver_exit.json',dict(exit_code=rc,finished=time.time(),command=command,pid=p.pid))
   sys.exit(rc)
  except Exception:
   dump('logs/driver_exit.json',dict(exit_code=1,traceback=traceback.format_exc(),finished=time.time()))
   raise
