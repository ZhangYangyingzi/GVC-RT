"""Own scheduler lock; preserve real exit code before final report and publication."""
from protocol import *
import fcntl,traceback

if __name__=='__main__':
 with (ROOT/'manifests/driver.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   command=[PYTHON,'-B','-u',str(ROOT/'src/driver.py')]
   with (ROOT/'logs/driver.log').open('a') as log:
    process=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    dump('manifests/driver_process.json',dict(pid=process.pid,launcher_pid=os.getpid(),command=command,started=time.time()))
    rc=process.wait()
   dump('logs/driver_exit.json',dict(pid=process.pid,exit_code=rc,finished=time.time(),command=command))
   with (ROOT/'logs/finalize.log').open('a') as log:summary=subprocess.run([PYTHON,'-B',str(ROOT/'src/summarize.py')],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
   dump('logs/finalize_exit.json',dict(exit_code=summary.returncode,finished=time.time()))
   summary.check_returncode()
   with (ROOT/'logs/final_publication.log').open('a') as log:pub=subprocess.run([PYTHON,'-B',str(ROOT/'src/publish.py')],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
   dump('logs/publication_exit.json',dict(exit_code=pub.returncode,finished=time.time()))
   pub.check_returncode()
   dump('logs/launcher_exit.json',dict(exit_code=rc,finished=time.time()))
   sys.exit(rc)
  except Exception:
   dump('logs/launcher_exit.json',dict(exit_code=1,traceback=traceback.format_exc(),finished=time.time()));raise
