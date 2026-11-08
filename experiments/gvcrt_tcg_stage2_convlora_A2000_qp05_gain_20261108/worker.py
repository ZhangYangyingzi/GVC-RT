"""Only QP0/5 new decodes, one worker per GPU, complete cache verification on resume."""
from common_s2 import *
from audit import *
import argparse,traceback

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,required=True);ap.add_argument('--gain',type=float,choices=[0.,.5,1.],default=.5)
    ap.add_argument('--smoke-qp',type=int,choices=[0,5]);a=ap.parse_args()
    env();verify_sources();assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7' and 0<=a.rank<4
    from model_context import backend
    torch,*_=backend(False,a.rank)
    optimizer_calls=[]
    def forbidden_optimizer(*args,**kwargs):
        optimizer_calls.append(True);raise RuntimeError('INFERENCE_ONLY_NO_OPTIMIZER_CREATION_OR_RESTORE')
    torch.optim.Optimizer.__init__=forbidden_optimizer
    from evaluate_core import point
    if a.smoke_qp is not None:items=[(video_list()[0],a.smoke_qp)]
    else:
        assert a.gain==.5 and load(ROOT/'audits/smoke.json')['status']=='PASS' and load(ROOT/'audits/source_reuse.json')['status']=='PASS'
        items=[item for i,item in enumerate(task_list()) if i%4==a.rank]
    decoded=0;cached=0
    for v,qp in items:
        batch=gain_batch(a.gain,qp);root=ROOT/batch
        with lock('locks/'+digest(dict(sequence=v['sequence'],external_QP=qp,gain=a.gain))+'.lock'):
            if (root/f'points/{key(v,qp)}.json').exists():
                p=validate_point(root,v,qp,a.gain)[0];cached+=1
            else:
                p=point(v,qp,CHECKPOINT,batch,a.rank,inference_gain=a.gain)
                validate_point(root,v,qp,a.gain);decoded+=1
            assert not optimizer_calls and p['inference_gain']==a.gain
            expected_components=load(QP9_SOURCE/f'evaluation/s05/points/{key(v,9)}.json')['component_hashes']
            assert p['component_hashes']==expected_components,'COMPONENT_IDENTITY_DIFFERS_FROM_VERIFIED_QP9'
            event('GAIN_POINT_COMPLETE',rank=a.rank,physical_GPU=physical_gpu(a.rank),sequence=v['sequence'],external_QP=qp,gain=a.gain,smoke=a.smoke_qp is not None)
    name=f'smoke_qp{a.smoke_qp}_s{a.gain:g}_rank{a.rank}' if a.smoke_qp is not None else f'formal_rank{a.rank}'
    dump(ROOT/f'audits/{name}.json',dict(status='PASS',points=len(items),new_decodes=decoded,validated_cached_points=cached,
        physical_GPU=physical_gpu(a.rank),training_updates=0,optimizer_calls=0,time=time.time()))

if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
