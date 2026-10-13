"""Persist exit evidence even when the worker raises; no model/API calls."""
from audit_io import *
import argparse, traceback
p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--exit',required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7));a=p.parse_args();t=load(ROOT/'config.json');task=json.loads(a.task);code=0
try:
    if task['kind']=='selftest':
        from selftest30 import perform
        perform(task)
    elif task['kind']=='cpu':
        from statistics_worker import main
        main(task)
    else:
        from worker import main
        main(task,a.gpu)
except BaseException:
    code=1;error=traceback.format_exc();print(error,flush=True);append_error(dict(task=task,traceback=error,worker=os.getpid()))
finally:dump(Path(a.exit),dict(exit_code=code,task=task,PID=os.getpid(),finished=time.time()))
sys.exit(code)
