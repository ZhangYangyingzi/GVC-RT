"""Measure unchanged independent Original FP32 receiver on prescribed dev point0."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common_s2 import *


def main():
    env();import torch
    from cache_codec import cache_paths
    video=load(ROOT/'manifests/development_videos.json')[0]
    key,path,receipt=cache_paths(video,0,False);original=ROOT/'cache/original_FP32'/f'{key}.pt'
    assert path.exists() and original.exists()
    import original_runtime
    loader=original_runtime.load_models;timings=[];started=[]
    def timed_loader(rt):
        im,pm=loader(rt)
        def pre(module,args):torch.cuda.synchronize();started.append(time.time())
        def post(module,args,out):torch.cuda.synchronize();timings.append(time.time()-started.pop())
        pm.recon_generation_net.register_forward_pre_hook(pre)
        pm.recon_generation_net.register_forward_hook(post)
        return im,pm
    original_runtime.load_models=timed_loader
    from original_reconstruction import main as render
    out=ROOT/'cache/original_latency_comparison.pt';render(str(path),str(out))
    a=torch.load(original,map_location='cpu',weights_only=True,mmap=True);b=torch.load(out,map_location='cpu',weights_only=True,mmap=True)
    assert len(timings)==63 and all(torch.equal(x,y) for x,y in zip(a['values'],b['values']))
    total=sum(timings)
    record=dict(status='PASS',model='Original',dataset=video['dataset'],sequence=video['sequence'],QP=0,P_frames=63,seconds=total,reference_fps=63/total,per_frame_seconds=timings,FP32_P_receiver=True,optimized_CUDA_performance_claim=False,bitwise_equal_to_scored_Original=True,stream_SHA256=load(receipt)['stream_SHA256'],GPU=0,co_resident_work='other authorized experiment/other-user processes were present; this is measured reference-path wall time',driver_SHA256=sha(Path(__file__)),time=time.time())
    dump(ROOT/'audits/Original_reference_latency.json',record)
    point=ROOT/'evaluation/Original/step_0000/points'/f'{video["sequence"]}_qp0.json'
    p=load(point);p['result'].update(decoder_seconds=total,decoder_reference_fps=63/total,decoder_latency_source=str(ROOT/'audits/Original_reference_latency.json'))
    dump(point,p);out.unlink()


if __name__=='__main__':main()
