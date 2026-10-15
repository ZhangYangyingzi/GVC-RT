import os,argparse,sys,time,traceback,json
from preflight import *
def main(gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch,numpy as np
 from PIL import Image
 from runtime import models,quality,Sender,step,losses
 im,pm=models('cuda:0');qualitys=quality('cuda:0');frozen=[state_hash(m.state_dict()) for m in (im,pm,*qualitys)]
 v=json.loads((GVC/'experiments/gvcrt_neural_wrapper_v6_22_long_reference_reconstruction_loss_control/train_manifest_uvg.json').read_text())['videos'][0]
 def source(i):
  with Image.open(Path(v['input_dir'])/f'frame_{i:06d}.png') as f:a=np.asarray(f,dtype=np.uint8).copy()[:256,:256]
  return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float().cuda()/255
 x0,x1=source(0),source(1);e,_,d,_=step(im,pm,x0,None,0,0)
 evidence=[]
 for kind in ('B_rgb_P','C_latent_P'):
  torch.manual_seed(20261015);p=Sender(kind).cuda();params=list(p.parameters());e,out,_,_=step(im,pm,x1,d,0,1,p)
  raw=losses(out,x1,qualitys);raw['rate']=e['bpp'].mean()
  row=dict(branch=kind,parameters=sum(t.numel() for t in params),loss={k:float(v) for k,v in raw.items()},gradients={})
  for name,value in raw.items():
   g=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
   norm=float(sum(t.double().square().sum() for t in g if t is not None).sqrt());valid=all(t is None or torch.isfinite(t).all() for t in g)
   row['gradients'][name]=dict(norm=norm,finite=bool(valid),nonzero_parameters=sum(t is not None and bool(t.count_nonzero()) for t in g))
   if not valid or norm<=0:save('audits/gradient.json',dict(status='FAIL',rows=evidence+[row]));raise RuntimeError(f'{kind}: invalid {name} gradient norm={norm}')
  evidence.append(row);save('audits/gradient.json',dict(status='CHECKING',rows=evidence))
 assert abs(evidence[0]['parameters']-evidence[1]['parameters'])/evidence[0]['parameters']<=.05
 assert frozen==[state_hash(m.state_dict()) for m in (im,pm,*qualitys)]
 save('audits/gradient.json',dict(status='PASS',rows=evidence,optimizer_updates=0,frozen_unchanged=True,peak_gpu_MiB=torch.cuda.max_memory_reserved()/2**20))
 from src.entropy_coder import GaussianRansCodec
 c=GaussianRansCodec();before=[state_hash(m.state_dict()) for m in (im,pm)];traces=[];encoded=[];dpb=None
 with torch.no_grad():
  for i in range(64):
   e,out,dpb,pad=step(im,pm,source(i),dpb,0,i,mode='compress',coder=c);encoded.append(e['bit_stream']);traces.append(state_hash({k:v for k,v in dpb.items() if k in ('ref_frame','ref_latent','ref_y') and v is not None}))
  dpb=None
  for i,b in enumerate(encoded):
   e=im.decompress(b,c) if i==0 else pm.decompress(b,dpb,c)
   dpb=dict(ref_frame=e['x_hat'],ref_latent=e['ref_latent'],ref_y=None) if i==0 else e['dpb']
   assert state_hash({k:v for k,v in dpb.items() if v is not None})==traces[i],('independent_reference_state_mismatch',i)
   # Encoder includes latent_ori only for observation; not part of decoder reference.
   assert all(torch.isfinite(v).all() for v in dpb.values() if torch.is_tensor(v))
 assert before==[state_hash(m.state_dict()) for m in (im,pm)]
 save('audits/causal64.json',dict(status='PASS',frames=64,rate_type='real_local_nonofficial_rans',bytes=sum(len(b) for b in encoded),peak_gpu_MiB=torch.cuda.max_memory_reserved()/2**20))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True);args=a.parse_args()
 try:main(args.gpu)
 except Exception:
  e=traceback.format_exc();save('BLOCKED.json',dict(status='BLOCKED',stage='gradient_and_causal_probe',error=e,B_updates=0,C_updates=0,evaluation_completed=0));print(e,flush=True);sys.exit(1)
