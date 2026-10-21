"""Durable per-task process exit records and task lock, including failed attempts."""
from io32 import *
import subprocess,argparse,fcntl
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--key',required=True);a.add_argument('--gpu',type=int,required=True);a.add_argument('command',nargs=argparse.REMAINDER);x=a.parse_args();l=(ROOT/f'tasks/{x.key}.worker.lock').open('a');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cmd=[PYTHON,'-B','-u',str(ROOT/x.command[0]),*x.command[1:],'--gpu',str(x.gpu)];start=time.time();append(ROOT/'logs/task_events.jsonl',dict(event='START',key=x.key,PID=os.getpid(),gpu=x.gpu,start=start,argv=cmd))
 with (ROOT/f'logs/{x.key}.log').open('a') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1'));code=p.wait()
 append(ROOT/'logs/task_events.jsonl',dict(event='EXIT',key=x.key,PID=os.getpid(),child_PID=p.pid,gpu=x.gpu,start=start,end=time.time(),exit_code=code))
 sys.exit(code)
