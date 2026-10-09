"""Exact residual statistics; CPU disk-backed order statistics avoid GPU accumulation."""
import fcntl,math,tempfile
import numpy as np
import torch
from scipy import ndimage
from parallel_utils import *

def gray(a):return a@np.array([.2126,.7152,.0722],dtype=np.float32)
def sobel(a):return np.hypot(ndimage.sobel(a,axis=0,mode='reflect')/8,ndimage.sobel(a,axis=1,mode='reflect')/8)
@torch.inference_mode()
def compute(root,video,sender,wrapper,frames,device):
    path=root/'parts/proxy'/video['dataset']/sender/f"video_{video['video_index']}.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():return
        n=len(frames);h,w=frames[0].shape[-2:];size=n*h*w*3
        l1=mse=s1=s2=sd=ld=sr=0.0;tx=[];tp=[];previous_x=previous_p=None
        with tempfile.TemporaryFile(dir=root/'parts') as handle:
            handle.truncate(size*4);magnitudes=np.memmap(handle,dtype=np.float32,mode='r+',shape=(size,));offset=0
            for frame in frames:
                x=frame.to(device);p=wrapper(x)
                if previous_x is not None:
                    tx.append(float((x-previous_x).abs().mean()));tp.append(float((p-previous_p).abs().mean()))
                previous_x,previous_p=x,p
                a=x[0].permute(1,2,0).cpu().numpy();b=p[0].permute(1,2,0).cpu().numpy();r=b-a;absr=np.abs(r)
                amount=r.size;magnitudes[offset:offset+amount]=absr.ravel();offset+=amount
                l1+=absr.sum(dtype=np.float64);mse+=np.square(r,dtype=np.float64).sum();s1+=r.sum(dtype=np.float64);s2+=np.square(r,dtype=np.float64).sum()
                ga,gb=gray(a),gray(b)
                sd+=float(np.abs(sobel(gb)-sobel(ga)).mean(dtype=np.float64))
                ld+=float(ndimage.laplace(gb,mode='reflect').var(dtype=np.float64)-ndimage.laplace(ga,mode='reflect').var(dtype=np.float64))
                sr+=float(sobel(absr.mean(axis=2)).mean(dtype=np.float64))
            pos=.95*(size-1);lo=math.floor(pos);hi=math.ceil(pos);magnitudes.partition((lo,hi))
            p95=float(magnitudes[lo]+(magnitudes[hi]-magnitudes[lo])*(pos-lo));del magnitudes
        row=dict(dataset=video['dataset'],scope='video',video=video['name'],P=sender,num_frames=n,
            proxy_source_L1=l1/size,proxy_source_MSE=mse/size,sobel_difference=sd/n,laplacian_variance_difference=ld/n,
            source_temporal_difference=float(np.mean(tx)),proxy_temporal_difference=float(np.mean(tp)),temporal_delta=float(np.mean(tp)-np.mean(tx)),
            residual_mean_abs=l1/size,residual_std=math.sqrt(max(s2/size-(s1/size)**2,0)),residual_p95_abs=p95,residual_sobel_abs=sr/n,
            definition='RGB [0,1]; Sobel difference=mean abs difference of grayscale gradient magnitudes, Sobel /8 reflect; signed Laplacian variance difference; Sobel(|r|) on channel-mean absolute residual')
        assert all(math.isfinite(v) for v in row.values() if isinstance(v,float))
        dump(path,row)
