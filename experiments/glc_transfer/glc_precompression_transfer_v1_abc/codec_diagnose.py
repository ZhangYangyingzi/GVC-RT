"""Bounded first-difference probe; independent receiver, no optimizer calls."""
from preflight import *
from data import load,Frames
import argparse
def main(gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch
 from runtime import models,step,state_hash
 from src.entropy_coder import GaussianRansCodec
 torch.set_num_threads(2);im,pm=models('cuda:0');di,dp=models('cuda:0');enc=GaussianRansCodec();dec=GaussianRansCodec();v=load(R/'eval_manifest.json')['ulong'][1];f=Frames(v);a=b=None;rows=[]
 with torch.inference_mode():
  for i in range(64):
   e,out,a,pad=step(im,pm,f.tensor(i,'cuda:0'),a,63,i,mode='compress',coder=enc)
   d=di.decompress(e['bit_stream'],dec) if i==0 else dp.decompress(e['bit_stream'],b,dec)
   b=dict(ref_frame=d['x_hat'],ref_latent=d['ref_latent'],ref_y=None) if i==0 else d['dpb']
   x={k:t for k,t in a.items() if k in ('ref_frame','ref_latent','ref_y') and t is not None};y={k:t for k,t in b.items() if t is not None}
   row=dict(frame=i,byte_hash_equal=state_hash(x)==state_hash(y),tensors={})
   for k in x:
    u,w=x[k],y[k];row['tensors'][k]=dict(shape=list(u.shape),finite=bool(torch.isfinite(u).all() and torch.isfinite(w).all()),numeric_equal=bool(torch.equal(u,w)),max_abs_difference=float((u-w).abs().max()),different_values=int((u!=w).sum()),signed_zero_differences=int(((u==0)&(w==0)&(torch.signbit(u)!=torch.signbit(w))).sum()),encoder_sha256=state_hash({k:u}),decoder_sha256=state_hash({k:w}))
   rows.append(row);save('audits/codec_first_difference.json',dict(status='CHECKING',dataset='ulong',video=1,q=63,optimizer_steps=0,rows=rows))
   print(json.dumps(row),flush=True)
   if not row['byte_hash_equal']:
    save('audits/codec_first_difference.json',dict(status='DIFFERENCE_LOCATED',dataset='ulong',video=1,q=63,optimizer_steps=0,rows=rows,peak_MiB=torch.cuda.max_memory_reserved()/2**20));return
 save('audits/codec_first_difference.json',dict(status='PASS_INTERLEAVED',dataset='ulong',video=1,q=63,optimizer_steps=0,rows=rows,peak_MiB=torch.cuda.max_memory_reserved()/2**20))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True);args=a.parse_args();main(args.gpu)
