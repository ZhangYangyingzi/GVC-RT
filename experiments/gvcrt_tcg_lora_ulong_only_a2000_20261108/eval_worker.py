from common_s2 import *
import argparse,traceback

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,required=True);ap.add_argument('--batch',required=True)
    ap.add_argument('--split',choices=['development','final'],default='development');ap.add_argument('--model',choices=['original','baseline','lora'],required=True)
    ap.add_argument('--checkpoint');ap.add_argument('--probe-dev',action='store_true');ap.add_argument('--probe-mcl',action='store_true');ap.add_argument('--probe-qp',type=int,choices=[0,5,9]);a=ap.parse_args()
    env();verify_sources();assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'
    if a.model=='original':os.environ['MODEL_KIND']='original'
    else:os.environ.pop('MODEL_KIND',None)
    from evaluate import backend,videos,point
    from evaluation_support import validate_point,key,register_mcl
    backend(False,a.rank)
    if a.probe_dev:vs=sorted(videos('development'),key=lambda v:v['sequence'])[:1];tasks=[(v,q) for v in vs for q in ([a.probe_qp] if a.probe_qp is not None else [0,5,9])]
    elif a.probe_mcl:tasks=[(load(ROOT/'manifests/MCL_JCV_30.json')[0],5)]
    else:
        assert a.split=='development' or load(ROOT/'selection.json')['status']=='FROZEN_BEFORE_FINAL_TEST'
        vs=videos(a.split)
        # Baseline old135 is explicitly reused, never merged with MCL RGB-bypass rows.
        if a.model=='baseline' and a.split=='final':vs=[v for v in vs if v['domain']=='mcljcv']
        tasks=[item for i,item in enumerate([(v,q) for v in vs for q in [0,5,9]]) if i%4==a.rank]
    for v,qp in tasks:
        dest=ROOT/a.batch/f'points/{key(v,qp)}.json'
        if dest.exists():validate_point(ROOT/a.batch,v,qp,a.model,a.checkpoint,not a.probe_mcl)
        else:
            reuse=not a.probe_mcl
            if v['domain']=='mcljcv' and a.model=='baseline':reuse=(ROOT/f'canonical_MCL/points/{key(v,qp)}.json').exists() and not a.probe_mcl
            p=point(v,qp,a.checkpoint,a.batch,a.rank,scoring=not a.probe_mcl,candidate=a.model,reuse_input=reuse)
            if v['domain']=='mcljcv' and a.model=='baseline':register_mcl(v,qp,p,a.batch)
            validate_point(ROOT/a.batch,v,qp,a.model,a.checkpoint,not a.probe_mcl)
        event('EVALUATION_POINT_COMPLETE',batch=a.batch,rank=a.rank,model=a.model,sequence=v['sequence'],QP=qp,scoring=not a.probe_mcl)
    dump(ROOT/f'{a.batch}/worker{a.rank}_status.json',dict(status='PASS',completed_points=len(tasks),physical_GPU=physical_gpu(a.rank),training_updates=0,time=time.time()))

if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
