#!/usr/bin/env python3
import argparse,csv,gc,hashlib,json,math,os,re,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent;GVC=ROOT.parents[1];DCVC=Path('/Huang_group/zyyz/Projects/DCVC_RT')
sys.path.insert(0,str(DCVC));sys.path.insert(0,str(GVC/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src'))

def load_png(p,device=None):
 a=np.asarray(Image.open(p).convert('RGB'),dtype=np.uint8)
 t=torch.from_numpy(a.copy()).permute(2,0,1).float().div_(255)
 return t if device is None else t.to(device)
def csv_write(p,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r));
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def global_ssim(a,b):
 ux=a.mean();uy=b.mean();vx=((a-ux)**2).mean();vy=((b-uy)**2).mean();cov=((a-ux)*(b-uy)).mean()
 return float(((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux**2+uy**2+.01**2)*(vx+vy+.03**2)))
def ms_ssim_rgb(a,b):
 # Standard five-scale, 11x11 Gaussian RGB MS-SSIM on [0,1] tensors.
 c=a.shape[0];x=torch.arange(11,device=a.device,dtype=a.dtype)-5
 g=torch.exp(-(x*x)/(2*1.5*1.5));g=g/g.sum();w=(g[:,None]*g[None,:]).expand(c,1,11,11)
 weights=torch.tensor([.0448,.2856,.3001,.2363,.1333],device=a.device,dtype=a.dtype)
 xx=a[None];yy=b[None];mss=[];mcs=[]
 for level in range(5):
  ux=F.conv2d(xx,w,groups=c);uy=F.conv2d(yy,w,groups=c)
  vx=F.conv2d(xx*xx,w,groups=c)-ux*ux;vy=F.conv2d(yy*yy,w,groups=c)-uy*uy
  cov=F.conv2d(xx*yy,w,groups=c)-ux*uy
  cs=(2*cov+.03**2)/(vx+vy+.03**2)
  ss=((2*ux*uy+.01**2)/(ux*ux+uy*uy+.01**2))*cs
  mss.append(ss.mean(dim=(0,2,3)));mcs.append(cs.mean(dim=(0,2,3)))
  if level<4:
   xx=F.avg_pool2d(F.pad(xx,(0,1,0,1),mode='reflect'),2,2)
   yy=F.avg_pool2d(F.pad(yy,(0,1,0,1),mode='reflect'),2,2)
 vals=torch.stack(mcs[:-1]).clamp_min(1e-12).pow(weights[:-1,None]).prod(0)*mss[-1].clamp_min(1e-12).pow(weights[-1])
 return float(vals.mean())
def parse_rc(name):
 if name.startswith('GVC_qp'): return 'GVC-RT',name.replace('GVC_qp','QP')
 if name.startswith('DCVC_qi'):
  m=re.match(r'DCVC_qi(\d+)_qp(\d+)',name);return 'DCVC-RT',f'I{m.group(1)}_P{m.group(2)}'
 if name.startswith('DCVC_q'): return 'DCVC-RT',f"I{name[6:]}_P{name[6:]}"
 raise ValueError(name)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--tags',required=True);a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);dev=torch.device('cuda:0')
 import lpips
 from DISTS_pytorch import DISTS
 lp=lpips.LPIPS(net='alex',verbose=False).to(dev).eval().requires_grad_(False);di=DISTS().to(dev).eval().requires_grad_(False)
 man=json.loads((ROOT/'manifest.json').read_text());lookup={f"{v['dataset']}_{int(v['video_id']):02d}":v for v in man['videos']}
 seqrows=[];framerows=[]
 for tag in a.tags.split(','):
  v=lookup[tag];srcdir=ROOT/'source_frames'/tag; recroot=ROOT/'reconstructions'/tag
  source_cpu=[load_png(srcdir/f'im{i:01d}.png') for i in range(1,65)]
  dirs=sorted(p for p in recroot.iterdir() if p.is_dir())
  for rd in dirs:
   codec,rc=parse_rc(rd.name);sum_sse=0.;npx=0;vals=[]
   for fi in range(64):
    s=source_cpu[fi].to(dev);r=load_png(rd/f'im{fi+1:05d}.png',dev)
    d=(r-s).float();sse=float(d.square().sum());mse=float(d.square().mean());psnr=-10*math.log10(max(mse,1e-15))
    ss=global_ssim(r,s);ms=ms_ssim_rgb(r,s)
    with torch.inference_mode():l=float(lp(r[None],s[None],normalize=True));dd=float(di(r[None],s[None]))
    if not all(math.isfinite(x) for x in (sse,mse,psnr,ss,ms,l,dd)):raise RuntimeError('nonfinite metric')
    vals.append((sse,mse,psnr,ss,ms,l,dd));sum_sse+=sse;npx+=r.numel()
    framerows.append({'codec':codec,'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'rate_control':rc,'frame':fi,'pixel_SSE':sse,'MSE':mse,'PSNR':psnr,'SSIM':ss,'MS_SSIM':ms,'LPIPS':l,'DISTS':dd,'FloLPIPS':''})
   agg=sum_sse/npx;seqrows.append({'codec':codec,'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'rate_control':rc,'num_frames':64,'fps':v['fps'],'total_pixel_SSE':sum_sse,'aggregate_MSE':agg,'sequence_PSNR':-10*math.log10(max(agg,1e-15)),'mean_frame_PSNR':sum(x[2] for x in vals)/64,'SSIM':sum(x[3] for x in vals)/64,'MS_SSIM':sum(x[4] for x in vals)/64,'LPIPS':sum(x[5] for x in vals)/64,'DISTS':sum(x[6] for x in vals)/64,'FloLPIPS':'','reconstruction_dir':str(rd)})
   print(tag,rd.name,'spatial done',flush=True)
  del source_cpu;torch.cuda.empty_cache()
 # FloLPIPS in a separate model phase.
 del lp,di;gc.collect();torch.cuda.empty_cache()
 from flolpips_metric import flolpips_for_videos
 from flolpips_compat import load_models,reference_flows
 flow,flp=load_models(dev)
 for tag in a.tags.split(','):
  v=lookup[tag];srcdir=ROOT/'source_frames'/tag; recroot=ROOT/'reconstructions'/tag
  src=torch.stack([load_png(srcdir/f'im{i}.png') for i in range(1,65)]).to(dev)
  flows=reference_flows(src,flow)
  for rd in sorted(p for p in recroot.iterdir() if p.is_dir()):
   codec,rc=parse_rc(rd.name);rec=torch.stack([load_png(rd/f'im{i:05d}.png') for i in range(1,65)]).to(dev)
   score,per=flolpips_for_videos(src,rec,flow,flp,flows)
   next(x for x in seqrows if x['dataset']==v['dataset'] and int(x['video_id'])==int(v['video_id']) and x['codec']==codec and x['rate_control']==rc)['FloLPIPS']=score
   for fi,val in enumerate(per):
    next(x for x in framerows if x['dataset']==v['dataset'] and int(x['video_id'])==int(v['video_id']) and x['codec']==codec and x['rate_control']==rc and int(x['frame'])==fi)['FloLPIPS']=val
   del rec;torch.cuda.empty_cache();print(tag,rd.name,'FloLPIPS done',flush=True)
  del src,flows;torch.cuda.empty_cache()
 suffix=a.tags.replace(',','__')
 csv_write(ROOT/'parts'/f'stream_metrics_{suffix}.csv',seqrows);csv_write(ROOT/'parts'/f'frame_metrics_{suffix}.csv',framerows)
if __name__=='__main__':main()
