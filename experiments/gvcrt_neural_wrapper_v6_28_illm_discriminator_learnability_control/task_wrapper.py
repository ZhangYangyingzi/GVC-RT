from io21 import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--gpu',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();args=a.command[1:] if a.command[0]=='--' else a.command;cmd=[PYTHON,str(ROOT/args[0]),*args[1:],'--gpu',a.gpu];rc=subprocess.call(cmd,cwd=REPO);dump(ROOT/'task_exit'/f'{a.id}.json',dict(exit_code=rc,ended_unix=time.time(),PID=os.getpid(),physical_gpu=int(a.gpu)));sys.exit(rc)
