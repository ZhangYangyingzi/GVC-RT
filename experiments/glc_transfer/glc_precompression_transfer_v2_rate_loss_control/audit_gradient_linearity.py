"""No-update investigation of arithmetic precision in weighted VJP checks."""
from preflight import *
from data import load,Frames
import argparse
def main(gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch
 from runtime import models,quality,Sender,step,losses
 from train import norm,seed,rng,restore
 from bootstrap import V1,BRANCHES
 from rate_backend import install
 from src.entropy_coder import GaussianRansCodec
 seed(20261016);rs=rng();im,pm=models('cuda:0');ql=quality('cuda:0');p=Sender('C_latent_P').cuda().eval();p.load_state_dict(torch.load(V1/'final_P/C_latent_P_1000.pt',map_location='cpu',weights_only=False)['P'],strict=True)
 fixed=[state_hash(m.state_dict()) for m in (im,pm,p,*ql)]
 install(pm,load(R/'audits/rate_backend_decision.json')['backend'],GaussianRansCodec())
 row=load(R/'plans/rate_audit.json')['rows'][18];f=Frames(row,True);dpb=None
 for pos in range(50):
  x=f.tensor(pos,'cuda:0')
  with torch.set_grad_enabled(pos==49):e,out,dpb,_=step(im,pm,x,dpb,row['q'],pos,p)
 terms=losses(out,x,ql);terms['rate']=e['bpp'].mean();params=list(p.parameters());beta=load(R/'calibration/final.json')['beta'][str(row['q'])]
 flags=dict(cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
 records=[]
 try:
  for mode in ('inherited_backward','full_fp32_backward_audit_only'):
   if mode!='inherited_backward':torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
   gs={k:[g if g is not None else torch.zeros_like(t) for g,t in zip(torch.autograd.grad(v,params,retain_graph=True,allow_unused=True),params)] for k,v in terms.items()}
   for branch,c in BRANCHES.items():
    weights=dict(LPIPS=1.,DISTS=c['dists'],L1=.1383589248954108,rate=beta*c['rate_multiplier']);total=sum(weights[k]*v for k,v in terms.items())
    coefficient_grad=torch.autograd.grad(total,tuple(terms.values()),retain_graph=True);assert all(torch.equal(g,torch.as_tensor(weights[k],dtype=g.dtype,device=g.device)) for k,g in zip(terms,coefficient_grad))
    actual=[g if g is not None else torch.zeros_like(t) for g,t in zip(torch.autograd.grad(total,params,retain_graph=True,allow_unused=True),params)]
    expected=[sum(weights[k]*gs[k][j] for k in terms) for j in range(len(params))];err=norm([a-b for a,b in zip(actual,expected)]);denom=norm(expected)
    records.append(dict(mode=mode,branch=branch,difference_norm=err,expected_norm=denom,relative_error=err/max(denom,1e-12),loss_boundary_coefficients_verified=True))
 finally:torch.backends.cudnn.allow_tf32=flags['cudnn_allow_tf32'];torch.backends.cuda.matmul.allow_tf32=flags['matmul_allow_tf32'];restore(rs);f.close()
 assert fixed==[state_hash(m.state_dict()) for m in (im,pm,p,*ql)]
 save('audits/recovery/gradient_linearity_diagnostic.json',dict(status='DIAGNOSTIC_COMPLETE',inherited_flags=flags,forward_recomputed=False,backward_precision_restored=True,training_precision_unchanged=True,optimizer_updates=0,parameters_unchanged=True,rows=records))
 print(json.dumps(records,indent=2),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True);main(a.parse_args().gpu)
