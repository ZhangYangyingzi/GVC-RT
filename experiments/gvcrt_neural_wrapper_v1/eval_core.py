import hashlib, io, math, os, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from core import ROOT, REPO, load_models, codec_input, unit, basic_metrics, model_hash
from wrapper_model import NeuralWrapper

from src.utils.stream_helper import write_sps, write_ip, read_header, read_sps_remaining, read_ip_remaining, NalType, SPSHelper
from gvc_hooks import INDEX_MAP

SPS={'sps_id':0,'height':1088,'width':1920,'ec_part':1,'use_ada_i':0}


def sha256_bytes(data): return hashlib.sha256(data).hexdigest()
def tensor_sha(x): return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def load_wrapper(path, device):
    if path is None: return None
    obj=torch.load(path,map_location='cpu',weights_only=True); model=NeuralWrapper().to(device).float().eval(); model.load_state_dict(obj['wrapper'],strict=True); model.requires_grad_(False); return model


def matched_frames(tag, count):
    base=REPO/'experiments/gvcrt_vs_dcvc_rt_matched_rate/source_frames'/tag
    return [torch.from_numpy(np.asarray(Image.open(base/f'im{i}.png').convert('RGB'),dtype=np.uint8).copy()).permute(2,0,1).unsqueeze(0).float()/255 for i in range(1,count+1)]


def video_frames(path, count):
    cmd=['ffmpeg','-v','error','-i',str(path),'-frames:v',str(count),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'];p=subprocess.Popen(cmd,stdout=subprocess.PIPE);out=[];size=1920*1080*3
    try:
        for _ in range(count):
            raw=p.stdout.read(size)
            if len(raw)!=size: raise RuntimeError('incomplete validation source')
            a=np.frombuffer(raw,np.uint8).reshape(1080,1920,3).copy();out.append(torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/255)
    finally:
        p.stdout.close(); rc=p.wait()
        if rc: raise RuntimeError('ffmpeg validation decode failed')
    return out


def ms_ssim_rgb(a,b):
    c=a.shape[1];x=torch.arange(11,device=a.device,dtype=a.dtype)-5;g=torch.exp(-(x*x)/(2*1.5*1.5));g=g/g.sum();w=(g[:,None]*g[None,:]).expand(c,1,11,11)
    weights=torch.tensor([.0448,.2856,.3001,.2363,.1333],device=a.device,dtype=a.dtype);xx=a;yy=b;mss=[];mcs=[]
    for level in range(5):
        ux=F.conv2d(xx,w,groups=c);uy=F.conv2d(yy,w,groups=c);vx=F.conv2d(xx*xx,w,groups=c)-ux*ux;vy=F.conv2d(yy*yy,w,groups=c)-uy*uy;cov=F.conv2d(xx*yy,w,groups=c)-ux*uy
        cs=(2*cov+.03**2)/(vx+vy+.03**2);ss=((2*ux*uy+.01**2)/(ux*ux+uy*uy+.01**2))*cs;mss.append(ss.mean((0,2,3)));mcs.append(cs.mean((0,2,3)))
        if level<4:xx=F.avg_pool2d(F.pad(xx,(0,1,0,1),mode='reflect'),2,2);yy=F.avg_pool2d(F.pad(yy,(0,1,0,1),mode='reflect'),2,2)
    return float((torch.stack(mcs[:-1]).clamp_min(1e-12).pow(weights[:-1,None]).prod(0)*mss[-1].clamp_min(1e-12).pow(weights[-1])).mean())


def save_png(x,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);a=torch.clamp(x*255,0,255).round().byte()[0].permute(1,2,0).cpu().numpy();Image.fromarray(a).save(path,compress_level=1)


def run_stream(frames_cpu,qp,wrapper,device,quality,fps,bitstream_path,save_root=None):
    im,enc=load_models(device); dim,dec=load_models(device); frozen_before=model_hash(im,enc,dim,dec);enc.clear_dpb();dec.clear_dpb();enc.set_curr_poc(0);dec.set_curr_poc(0)
    stream=io.BytesIO();write_sps(stream,SPS);rows=[];hashes=[];state=[]
    with torch.inference_mode():
        for fi,cpu in enumerate(frames_cpu):
            gt=cpu.to(device);proxy=gt if wrapper is None else wrapper(gt);aq=qp if fi==0 else enc.shift_qp(qp,INDEX_MAP[fi%8]);is_i=fi==0
            if is_i:
                e=im.compress(codec_input(proxy),aq); enc.clear_dpb();enc.add_ref_frame(None,e['x_hat']);d=dim.decompress(e['bit_stream'],SPS,aq);dec.clear_dpb();dec.add_ref_frame(None,d['x_hat'])
            else:
                e=enc.compress(codec_input(proxy),aq);d=dec.decompress(e['bit_stream'],SPS,aq);state.append(float((enc.dpb[0].feature-dec.dpb[0].feature).abs().max()))
            write_ip(stream,is_i,0,aq,e['bit_stream']);out=unit(d['x_hat'],1080,1920);sse,mse,psnr,ss=basic_metrics(out,gt);lp=float(quality[0](out,gt,normalize=True));di=float(quality[1](out,gt));ms=ms_ssim_rgb(out,gt)
            if not all(math.isfinite(v) for v in (sse,mse,psnr,ss,lp,di,ms)):raise RuntimeError('nonfinite evaluation metric')
            rows.append({'frame':fi,'actual_qp':aq,'payload_bytes':len(e['bit_stream']),'pixel_SSE':sse,'MSE':mse,'PSNR':psnr,'SSIM':ss,'MS_SSIM':ms,'LPIPS':lp,'DISTS':di});hashes.append(tensor_sha(d['x_hat']))
            if save_root is not None:save_png(proxy,Path(save_root)/'proxy'/f'frame_{fi:06d}.png');save_png(out,Path(save_root)/'recon'/f'frame_{fi:06d}.png')
    data=stream.getvalue();Path(bitstream_path).parent.mkdir(parents=True,exist_ok=True);Path(bitstream_path).write_bytes(data)
    pixels=len(rows)*3*1080*1920;sse=sum(r['pixel_SSE'] for r in rows);mse=sse/pixels
    summary={'bytes':len(data),'bits':len(data)*8,'bpp':len(data)*8/(len(rows)*1080*1920),'kbps':len(data)*8*fps/len(rows)/1000,'total_pixel_SSE':sse,'aggregate_MSE':mse,'sequence_PSNR':-10*math.log10(max(mse,1e-15)),'mean_PSNR':sum(r['PSNR'] for r in rows)/len(rows),'SSIM':sum(r['SSIM'] for r in rows)/len(rows),'MS_SSIM':sum(r['MS_SSIM'] for r in rows)/len(rows),'LPIPS':sum(r['LPIPS'] for r in rows)/len(rows),'DISTS':sum(r['DISTS'] for r in rows)/len(rows),'bitstream_path':str(bitstream_path),'bitstream_sha256':sha256_bytes(data),'state_sync_pass':all(x==0 for x in state),'model_hash_before':frozen_before,'model_hash_after':model_hash(im,enc,dim,dec)}
    # Fresh independent decode of the serialized stream.
    aim,apm=load_models(device);apm.clear_dpb();apm.set_curr_poc(0);buf=io.BytesIO(data);helper=SPSHelper();decoded_hash=[];consumed=0
    with torch.inference_mode():
        while buf.tell()<len(data):
            head=read_header(buf)
            while head['nal_type']==NalType.NAL_SPS:
                helper.add_sps_by_id(read_sps_remaining(buf,head['sps_id']));head=read_header(buf)
            sp=helper.get_sps_by_id(head['sps_id']);q,payload=read_ip_remaining(buf)
            if head['nal_type']==NalType.NAL_I:d=aim.decompress(payload,sp,q);apm.clear_dpb();apm.add_ref_frame(None,d['x_hat'])
            else:d=apm.decompress(payload,sp,q)
            decoded_hash.append(tensor_sha(d['x_hat']))
        consumed=buf.tell()
    summary['independent_decode_pass']=decoded_hash==hashes and consumed==len(data);summary['bytes_consumed']=consumed;summary['finite']=True
    if not summary['independent_decode_pass'] or not summary['state_sync_pass'] or summary['model_hash_before']!=summary['model_hash_after']:raise RuntimeError('stream audit failed')
    return summary,rows

