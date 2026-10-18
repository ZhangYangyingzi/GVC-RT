"""Detached launcher; never replaces an active driver's lock or worker state."""
from io32 import *
import fcntl,subprocess
f=(ROOT/'driver.lock').open('a')
try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
 print('Already active:',load(ROOT/'pipeline_status.json').get('driver'));raise SystemExit(0)
fcntl.flock(f,fcntl.LOCK_UN)
log=(ROOT/'logs/driver.log').open('a');argv=[PYTHON,'-B','-u',str(ROOT/'driver32b.py')]
p=subprocess.Popen(argv,cwd=REPO,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
dump(ROOT/'launcher_receipt.json',dict(PID=p.pid,argv=argv,started=time.time(),persistence='detached POSIX session'));print('Driver PID',p.pid)
