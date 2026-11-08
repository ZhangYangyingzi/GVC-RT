"""QP9 only; fresh full-sequence model/decoder state for each new measurement."""
from common_s2 import *
from audit import video_list,gain_batch,key
import argparse,traceback

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,required=True);ap.add_argument('--gain',type=float,choices=[0.,.5,1.],required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    env();verify_sources();assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7' and 0<=a.rank<4
    from model_context import backend
    torch,*_=backend(False,a.rank)
    optimizer_calls=[]
    def forbidden_optimizer(*args,**kwargs):
        optimizer_calls.append(True);raise RuntimeError('OPTIMIZER_CREATION_FORBIDDEN_IN_INFERENCE_DIAGNOSTIC')
    torch.optim.Optimizer.__init__=forbidden_optimizer
    from evaluate_core import point
    if a.smoke:items=video_list()[:1]
    else:
        assert a.gain==.5,'ONLY_HALF_GAIN_FORMAL_POINTS_ALLOWED'
        assert load(ROOT/'audits/smoke.json')['status']=='PASS' and load(ROOT/'audits/endpoint_reuse.json')['status']=='PASS'
        items=[v for i,v in enumerate(video_list()) if i%4==a.rank]
    batch=gain_batch(a.gain)
    for v in items:
        with lock('locks/'+digest(dict(sequence=v['sequence'],gain=a.gain,QP=9))+'.lock'):
            p=point(v,9,CHECKPOINT,batch,a.rank,inference_gain=a.gain)
            assert not optimizer_calls and p['inference_gain']==a.gain
            event('GAIN_POINT_COMPLETE',rank=a.rank,physical_GPU=physical_gpu(a.rank),sequence=v['sequence'],inference_gain=a.gain,smoke=a.smoke)
    dump(ROOT/f'audits/worker_{"smoke" if a.smoke else "formal"}_{a.rank}_s{a.gain}.json',dict(status='PASS',points=len(items),physical_GPU=physical_gpu(a.rank),training_updates=0,optimizer_calls=0,time=time.time()))

if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
