"""Detach the existing controller and retain its recorded worker identities."""
import fcntl
from mixed_io import *
if __name__=='__main__':
    with (ROOT/'pipeline.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Controller already running');sys.exit(0)
    argv=[PYTHON,'-B','-u',str(ROOT/'run_pipeline.py')]
    command(argv)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (ROOT/'logs/pipeline.log').open('a') as stream:
        p=subprocess.Popen(argv,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print('RESUMED CONTROLLER',p.pid,flush=True)
