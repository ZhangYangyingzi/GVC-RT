from io21 import *
name=sys.argv[1];rc=subprocess.call([PYTHON,'-B','-u',*sys.argv[2:]])
dump(ROOT/'task_exits'/f'{name}.json',dict(exit_code=rc,unix=time.time(),PID=os.getpid()))
sys.exit(rc)
