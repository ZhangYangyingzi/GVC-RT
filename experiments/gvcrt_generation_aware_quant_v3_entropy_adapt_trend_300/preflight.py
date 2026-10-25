"""v3 probability/quantization checks. No repeated H00/H10/H01 ablation."""
from util import *
import argparse

def gradient_checks(gpu):
 gpu_env(gpu);sys.path.insert(0,str(P));import train as tr
 assert Path(tr.__file__).resolve()==P/'train.py'
 tr.setup_path()
 import torch,numpy as np,random
 from core import codec_input
 from runtime import PHI_SCOPE
 from model_io import phi_hash,assert_cdf
 torch.set_num_threads(2);results={};scope=[]
 row=load(P/'train_plan.json')['rows'][0]
 import lpips
 lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
 for branch in ('N1','Q1'):
  random.seed(20261023);np.random.seed(20261023);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023)
  im,pm=tr.load_models(branch);base_branch='N0' if branch=='N1' else 'Q0';bi,bp=tr.load_models(base_branch)
  assert phi_hash(pm,True)==phi_hash(bp,True)
  assert assert_cdf(pm)['CDF']==assert_cdf(bp)['CDF']
  assert model_hash(pm.recon_generation_net.decoder)==model_hash(bp.recon_generation_net.decoder)
  if pm.use_quant:assert model_hash(pm.quant_adjust)==model_hash(bp.quant_adjust)
  frames=tr.load_frames(row,torch.device('cuda:0'))
  for image,model in ((im,pm),(bi,bp)):
   model.clear_dpb();model.set_curr_poc(0)
   with torch.no_grad():model.add_ref_frame(None,image.compress(codec_input(frames[0]),row['external_qp'])['x_hat'])
  initial_errors=[]
  for pos in range(1,13):
   q=pm.shift_qp(row['external_qp'],[0,1,0,2,0,2,0,2][pos%8])
   with torch.no_grad():brec,bf,brate,bt=tr.causal_forward(bp,frames[pos],q)
   with torch.enable_grad() if pos==12 else torch.no_grad():rec,feature,rate,t=tr.causal_forward(pm,frames[pos],q)
   for key in ('q0','q1','qmap','y_hat'):
    assert torch.equal(t[key],bt[key]),(branch,pos,key)
   assert torch.equal(feature,bf) and torch.equal(rec,brec),(branch,pos,'initial_forward')
   assert float(rate.detach())==float(brate),(branch,pos,'proxy_rate')
   if pos<12:
    with torch.no_grad():
     pm.add_ref_frame(feature.detach(),(rec.detach()*2-1).half());bp.add_ref_frame(bf.detach(),(brec.detach()*2-1).half())
  reconstruction=lp(rec,frames[12],normalize=True).mean()+.1383589248954108*(rec-frames[12]).abs().mean()
  ys=[p for name in PHI_SCOPE[:-1] for p in getattr(pm,name).parameters()];zs=list(pm.bit_estimator_z.parameters());gs=list(pm.recon_generation_net.decoder.parameters())
  yn=tr.norm(torch.autograd.grad(t['y_bits']/65536,ys,retain_graph=True,allow_unused=True));zn=tr.norm(torch.autograd.grad(t['z_bits']/65536,zs,retain_graph=True,allow_unused=True));gn=tr.norm(torch.autograd.grad(reconstruction,gs,retain_graph=True,allow_unused=True))
  assert all(x>0 and __import__('math').isfinite(x) for x in (yn,zn,gn))
  hnorms={}
  if branch=='Q1':
   hs=list(pm.quant_adjust.parameters());hnorms=dict(reconstruction=tr.norm(torch.autograd.grad(reconstruction,hs,retain_graph=True,allow_unused=True)),rate=tr.norm(torch.autograd.grad(rate/65536,hs,retain_graph=True,allow_unused=True)))
   assert all(x>0 and __import__('math').isfinite(x) for x in hnorms.values())
   assert torch.equal(pm.quant_adjust(t['common'],q),torch.ones_like(pm.quant_adjust(t['common'],q)))
  assert all(p.grad is None for p in im.parameters()) and all(p.grad is None for p in pm.parameters() if not p.requires_grad)
  for name,p in pm.named_parameters():
   if p.requires_grad:assert p.dtype==torch.float32
   if any(name.startswith(k+'.') for k in PHI_SCOPE):scope.append(dict(branch=branch,name='pm.'+name,shape=list(p.shape),parameters=p.numel(),master_dtype=str(p.dtype),trainable=p.requires_grad))
  results[branch]=dict(Phi_y_rate_gradient_norm=yn,Phi_z_rate_gradient_norm=zn,G_reconstruction_gradient_norm=gn,H_gradients=hnorms,initial_symbols_qmap_latent_reference_RGB_exact=True,initial_proxy_rate_exact=True,initial_CDF_exact=True,initial_forward_Phi_exact=True,frozen_parameter_gradients_absent=True)
  del im,pm,bi,bp,frames,rec,feature,rate,t,bt,bf,brec,reconstruction;torch.cuda.empty_cache()
 dump(P/'audits/probability_gradient_initial.json',dict(status='PASS',checks=results,parameters=scope,signature=signature()))
 dump(P/'probability_parameter_scope.json',scope)
 print('INITIAL_GRADIENT_CHECKS_PASS',results,flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,default=6);args=a.parse_args();gradient_checks(args.gpu)
