import argparse,traceback
from io18 import *
if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a,_=p.parse_known_args();os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    command([PYTHON,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    (ROOT/'parts').mkdir(exist_ok=True)
    try:
        from adapter import evaluator,engine
        e=evaluator();sys.modules['engine']=engine();e.main()
    except Exception:
        dump(ROOT/'logs'/f'failure_worker_{os.getpid()}_{time.time_ns()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
