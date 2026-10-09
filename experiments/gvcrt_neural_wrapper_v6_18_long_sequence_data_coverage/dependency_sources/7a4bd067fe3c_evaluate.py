import traceback
from retention_io import *
if __name__=='__main__':
    command([PYTHON,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    try:
        from adapter import evaluator,engine
        e=evaluator();sys.modules['engine']=engine();e.main()
    except Exception:
        dump(ROOT/'logs'/f'failure_worker_{os.getpid()}_{time.time_ns()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
