"""No adapted holdout evaluation before training and checkpoint freeze."""
import argparse,traceback
from v614_io import *
if __name__=='__main__':
    try:
        command([PYTHON,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]])
        p=argparse.ArgumentParser(add_help=False);p.add_argument('--video',type=int);p.add_argument('--method');a,_=p.parse_known_args()
        v=next(v for v in sources() if v['video_index']==a.video)
        if a.method.startswith('uvg_adapt_') and split(v)=='holdout':
            assert load(ROOT/'training_status.json')['status']=='PASS'
            freeze=load(ROOT/'frozen_checkpoint_index.json');assert freeze['status']=='PASS' and freeze['checkpoint_index_sha256']==sha(ROOT/'checkpoint_index.json')
        from eval_adapter import evaluator
        evaluator().main()
    except Exception:dump(ROOT/'logs'/f'failure_eval_{os.getpid()}_{time.time_ns()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
