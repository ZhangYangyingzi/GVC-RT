"""Isolated component gradients and local rate-direction responses; no optimizer."""
from preflight import *
from data import load,Frames
from bootstrap import V1
import argparse,itertools
def run(index,method,gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch,numpy as np
 from runtime import models,quality,Sender,step,losses
 from train import norm,seed,rng,restore
 from src.entropy_coder import GaussianRansCodec
 from src.entropy_coder.bitstream import parse_frame_packet
 seed(20261016);random_state=rng();row=load(R/'plans/rate_audit.json')['rows'][index];im,pm=models('cuda:0');ql=quality('cuda:0');p=Sender(method).cuda().eval();p.load_state_dict(torch.load(V1/'final_P'/f'{method}_1000.pt',map_location='cpu',weights_only=False)['P'],strict=True)
 frozen=[state_hash(m.state_dict()) for m in (im,pm,*ql)];original={k:v.detach().clone() for k,v in p.state_dict().items()};p_hash=state_hash(original);params=list(p.parameters());keys=['LPIPS','DISTS','L1','rate'];gs={k:[torch.zeros_like(t) for t in params] for k in keys};values={k:0. for k in keys};f=Frames(row,True);dpb=None;probe=None
 for pos in range(64):
  x=f.tensor(pos,'cuda:0')
  if pos==49:probe=(x,dict(dpb))
  with torch.set_grad_enabled(pos>=49):
   e,out,dpb,_=step(im,pm,x,dpb,row['q'],pos,p)
   if pos<49:continue
   terms=losses(out,x,ql);terms['rate']=e['bpp'].mean()
   for j,k in enumerate(keys):
    gg=torch.autograd.grad(terms[k]/15,params,retain_graph=j<3,allow_unused=True)
    for acc,g in zip(gs[k],gg):
     if g is not None:acc.add_(g.detach())
    values[k]+=float(terms[k])/15
 for k in keys:assert norm(gs[k])>0 and all(torch.isfinite(g).all() for g in gs[k]),('invalid component gradient',k)
 dot=lambda a,b:float(sum((u.double()*v.double()).sum() for u,v in zip(a,b)))
 cosine={f'{a}__{b}':dot(gs[a],gs[b])/(norm(gs[a])*norm(gs[b])) for a,b in itertools.combinations(keys,2)}
 beta=load(V1/'calibration/final.json')['beta'][str(row['q'])]
 weights={b:dict(LPIPS=1.,DISTS=c['dists'],L1=.1383589248954108,rate=beta*c['rate_multiplier']) for b,c in load(R/'config.json')['branches'].items()}
 gradients={k:dict(norm=norm(gs[k]),finite=True,nonzero_parameters=sum(bool(g.count_nonzero()) for g in gs[k])) for k in keys}
 responses=[]
 if method=='C_latent_P' and index%2==0:
  coder=GaussianRansCodec();x,reference=probe
  for size in (0.,1e-5,1e-4):
   p.load_state_dict(original,strict=True)
   with torch.no_grad():
    if size:
     for param,g in zip(params,gs['rate']):param.add_(g,alpha=-size/norm(gs['rate']))
    e,out,_,_=step(im,pm,x,reference,row['q'],49,p);c,rec,_,_=step(im,pm,x,reference,row['q'],49,p,mode='compress',coder=coder);assert torch.equal(out,rec)
    packet=parse_frame_packet(c['bit_stream']);responses.append(dict(parameter_direction_norm=size,estimated_bits=float(e['bit']),payload_bits=len(packet['y_payload'])*8,frame_bytes=len(c['bit_stream']),reconstruction_hash=state_hash({'rec':rec}),quantization_plateau_permitted=True))
  p.load_state_dict(original,strict=True)
 assert state_hash(p.state_dict())==p_hash and frozen==[state_hash(m.state_dict()) for m in (im,pm,*ql)]
 restore(random_state);f.close()
 save(f'audits/component_gradients/{index:02d}_{method}.json',dict(status='PASS',index=index,method=method,domain=row['domain'],q=row['q'],sample=row,values=values,unweighted=gradients,weighted={b:{k:dict(weight=w[k],norm=w[k]*gradients[k]['norm']) for k in keys} for b,w in weights.items()},cosines_unweighted=cosine,cosines_weighted=cosine,local_responses=responses,local_reference_position=48,local_probe_position=49,local_gradient_direction='negative normalized 15-supervised-frame official rate gradient; fixed causal reference at frame48',optimizer_updates=0,source_and_quality_unchanged=True,RNG_restored=True,P_restored=True,peak_MiB=torch.cuda.max_memory_reserved()/2**20))
 print('GRADIENT PASS',index,method,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--index',type=int,required=True);a.add_argument('--method',choices=['B_rgb_P','C_latent_P'],required=True);a.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True);x=a.parse_args();run(x.index,x.method,x.gpu)
