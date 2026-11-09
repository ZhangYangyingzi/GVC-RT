"""One canonical stream per video/QP, then two independent model measurements."""
from common_s2 import *
from task import tasks,verify
import argparse,traceback

def run(phase,index):
    env();assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7';assert verify();task=tasks()[index]
    rank=task['rank'];profile=task['profile'];qp=task['QP']
    with lock(f'locks/{phase}_{index}.lock'):
        if phase=='cache':
            os.environ['MODEL_KIND']='original'
            from cache_codec import ensure
            proof=ensure(profile['row'],qp,False,rank)
            dump(ROOT/f'audits/cache_task_{index}.json',dict(status='PASS',cache_identity=proof['cache_identity'],fresh_original_encode_and_independent_decode=True,physical_GPU=physical_gpu(rank),PID=os.getpid(),time=time.time()))
        else:
            os.environ.pop('MODEL_KIND',None)
            from evaluation import point
            for group in ['Original','B']:
                result=point(profile,qp,group,300 if group=='B' else 0,rank,export=True)
                clip=result['lossless_video'];assert clip and sha(clip['path'])==clip['SHA256']
                assert result['result']['frames']==128 and result['result']['transitions']==127
                dump(ROOT/f'audits/eval_task_{index}_{group}.json',dict(status='PASS',model=group,physical_GPU=physical_gpu(rank),PID=os.getpid(),source_frames=profile['row']['source_frame_indices'],training_updates=0,time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cache','evaluate'],required=True);p.add_argument('--index',type=int,choices=range(4),required=True);a=p.parse_args()
    try:run(a.phase,a.index)
    except Exception as e:
        env();dump(ROOT/f'audits/failure_{a.phase}_{a.index}_{int(time.time())}.json',dict(status='FAILED',exception=repr(e),traceback=traceback.format_exc(),training_updates=0,time=time.time()));raise
