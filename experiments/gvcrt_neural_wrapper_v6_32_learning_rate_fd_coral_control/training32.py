"""Historical fused C1 backward; isolated full-resolution streaming replay."""
from io32 import *
import random,math
def backend():
 import torch
 sys.path.insert(0,str(REPO));sys.path.insert(0,str(ROOT.parent/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies'))
 from src.utils.common import set_torch_env
 torch.set_num_threads(2);set_torch_env()
 return dict(setup_sha256=sha(REPO/'src/utils/common.py'),deterministic=torch.are_deterministic_algorithms_enabled(),cudnn_benchmark=torch.backends.cudnn.benchmark,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)
def official_model():
 import torch
 sys.path.insert(0,str(V29/'vendor/pytorch-fid/src'))
 import pytorch_fid.inception as src
 p=Path('/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth')
 assert sha(p)=='6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'
 src.load_state_dict_from_url=lambda *a,**k:torch.load(p,map_location='cpu',weights_only=True)
 net=src.InceptionV3([3],resize_input=True,normalize_input=True,requires_grad=False,use_fid_inception=True).cuda().eval().requires_grad_(False)
 class Flat(torch.nn.Module):
  def __init__(self):super().__init__();self.net=net
  def forward(self,x):return self.net(x.float())[0].flatten(1)
 return Flat().eval().requires_grad_(False)
def rng_state():
 import torch,numpy as np
 n=np.random.get_state()
 return dict(python=random.getstate(),numpy=(n[0],torch.tensor(n[1].astype('int64')),*n[2:]),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def restore_rng(s):
 import torch,numpy as np
 random.setstate(s['python']);n=s['numpy'];np.random.set_state((n[0],n[1].numpy().astype('uint32'),*n[2:]));torch.set_rng_state(s['torch']);torch.cuda.set_rng_state(s['cuda'])
def seed(n):
 import torch,numpy as np
 random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def buffer_state(models):return {k:{n:t.detach().clone() for n,t in m.named_buffers()} for k,m in models.items()}
def restore_buffers(models,buf):
 import torch
 with torch.no_grad():
  for k,m in models.items():
   for n,t in m.named_buffers():t.copy_(buf[k][n])
def norm(gs):return sum(float(g.double().square().sum()) for g in gs if g is not None)**.5
def dot(a,b):return sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b) if x is not None and y is not None)
def groups(models,gs):
 result={};offset=0
 for k in ('bridge','generator'):
  n=len(list(models[k].parameters()));g=gs[offset:offset+n];offset+=n
  result[k]=dict(norm=norm(g),nonzero=sum(x is not None and bool(x.any()) for x in g),finite=all(x is None or bool(x.isfinite().all()) for x in g),dtype='FP32')
 return result
def zero(params):
 import torch
 return [torch.zeros_like(p) for p in params]
def add(dst,gs,scale=1.,weight=1.):
 for b,g in zip(dst,gs):
  if g is not None:
   value=g.detach().float()/scale*weight;assert value.isfinite().all();b.add_(value)
def setup():
 import torch
 backend();v4,im,pm,models,quality=legacy_io().load_training(torch.device('cuda:0'))
 common=legacy_io().module('v632_common',legacy_io().V62/'fullqp_common.py')
 params=[p for k in ('bridge','generator') for p in models[k].parameters()]
 assert all(p.dtype==torch.float32 and p.requires_grad for p in params)
 frozen=dict(wrapper=v4.module_hash(models['wrapper']),compression=v4.compression_hash(im,pm),quality=[v4.module_hash(m) for m in quality])
 assert frozen['compression']==load(ROOT/'evaluation_config.json')['compression_hash']
 return v4,im,pm,models,quality,common,params,frozen
def check_frozen(ctx):
 v4,im,pm,models,quality,common,params,frozen=ctx
 assert frozen==dict(wrapper=v4.module_hash(models['wrapper']),compression=v4.compression_hash(im,pm),quality=[v4.module_hash(m) for m in quality])
def crop_c1(ctx,frames,q):
 import torch
 import torch.nn.functional as F
 v4,im,pm,models,quality,common,params,_=ctx
 stat=module('v632_stat',V29/'stat_loss.py');base=zero(params);recon=zero(params);stats=zero(params);capt=[];trace=[]
 hook=pm.dec.register_forward_hook(lambda _m,_a,o:capt.append(o))
 def ref(rec):
  assert len(capt)==1
  with torch.no_grad():pm.add_ref_frame(capt.pop().detach(),(rec.detach()*2-1).half())
  assert not pm.dpb[0].frame.requires_grad and not pm.dpb[0].feature.requires_grad
 try:
  pm.clear_dpb();pm.set_curr_poc(0)
  with torch.no_grad():pm.add_ref_frame(None,im.compress(v4.codec_input(frames[0]),q)['x_hat'])
  beta=common.beta_q('schedule_s1p0',q)
  for pos in range(1,64):
   gt=frames[pos];qa=common.frame_qp(pm,q,pos)
   with torch.enable_grad() if pos>=49 else torch.no_grad():
    proxy=models['wrapper'](gt);rec,rate,_=v4.joint_ste_forward(pm,proxy,qa)
    if pos>=49:
     lp=quality[0](rec,gt,normalize=True).mean();ds,st,_,_=stat.dists_and_stat(quality[1],rec,gt);rl=F.l1_loss(rec,gt);pl=F.l1_loss(proxy,gt);rb=rate/65536
     fused=lp+.5*ds+beta*rb+.01*pl
     add(base,torch.autograd.grad(fused/15,params,retain_graph=True,allow_unused=True))
     add(recon,torch.autograd.grad(rl*(4096/15),params,retain_graph=True,allow_unused=True),4096,REC)
     add(stats,torch.autograd.grad(st*(4096/15),params,allow_unused=True),4096,STAT)
     vals=dict(LPIPS=float(lp.detach()),DISTS=float(ds.detach()),weighted_DISTS=float(ds.detach())*.5,rate_bpp=float(rb.detach()),beta_rate=float(rb.detach())*beta,proxy_L1=float(pl.detach()),weighted_proxy_L1=float(pl.detach())*.01,reconstruction_L1=float(rl.detach()),weighted_reconstruction_L1=float(rl.detach())*REC,stat_loss=float(st.detach()),weighted_stat_loss=float(st.detach())*STAT)
     assert all(math.isfinite(x) for x in vals.values())
     trace.append(dict(position=pos,actual_qp=qa,normalization_denominator=15,**vals,total_loss=float(fused.detach())+REC*float(rl.detach())+STAT*float(st.detach())))
   ref(rec)
 finally:hook.remove()
 assert len(trace)==15
 total=[a+b+c for a,b,c in zip(base,recon,stats)];summary={n:sum(r[n] for r in trace)/15 for n in vals}
 summary.update(total_loss=sum(r['total_loss'] for r in trace)/15,beta_q=beta,base_gradient_norm=norm(base),reconstruction_gradient_norm=norm(recon),stat_gradient_norm=norm(stats),C1_gradient_norm=norm(total),stat_base_cosine=dot(stats,[a+b for a,b in zip(base,recon)])/max(norm(stats)*norm([a+b for a,b in zip(base,recon)]),1e-30),proxy_gradient_status='EXPECTED_FROZEN_OR_INDEPENDENT',rate_gradient_status='EXPECTED_FROZEN_OR_INDEPENDENT')
 return total,summary,trace
def full_pass(ctx,frames,q,official,feature_grad=None,expected=None):
 """64 causal frames, separate buffer/RNG scope supplied by caller. Per-frame VJP."""
 import torch
 import torch.nn.functional as F
 v4,im,pm,models,quality,common,params,_=ctx;capt=[];out=[];real=[];replay=[];gs=zero(params)
 hook=pm.dec.register_forward_hook(lambda _m,_a,o:capt.append(o))
 try:
  pm.clear_dpb();pm.set_curr_poc(0)
  gt=frames[0];assert gt.shape==(1,3,1080,1920)
  with torch.no_grad():pm.add_ref_frame(None,im.compress(v4.codec_input(F.pad(gt,(0,0,0,8),mode='replicate')),q)['x_hat'])
  for pos in range(1,64):
   gt=frames[pos];qa=common.frame_qp(pm,q,pos);grad=feature_grad is not None and pos>=49
   with torch.enable_grad() if grad else torch.no_grad():
    proxy=models['wrapper'](gt);canvas=F.pad(proxy,(0,0,0,8),mode='replicate');rec,rate,_=v4.joint_ste_forward(pm,canvas,qa);rgb=rec[:,:,:1080,:1920]
    if pos>=49:
     z=official(rgb);assert z.shape==(1,2048) and torch.isfinite(z).all();out.append(z.detach().cpu())
     with torch.no_grad():real.append(official(gt).cpu())
     if expected is not None:
      e=expected[pos-49].to(z.device);diff=z.detach()-e;rel=float(diff.double().norm()/e.double().norm().clamp_min(1e-12));ma=float(diff.abs().max());lim=1e-4+1e-4*float(e.abs().max());assert rel<=1e-4 and ma<=lim
      replay.append(dict(position=pos,feature_relative_L2=rel,feature_maxabs=ma,maxabs_tolerance=lim,status='PASS'))
     if grad:
      g=feature_grad[pos-49].to(z.device,dtype=z.dtype).reshape_as(z)
      gg=torch.autograd.grad(z,params,grad_outputs=g*4096,allow_unused=True);add(gs,gg,4096)
   assert len(capt)==1
   with torch.no_grad():pm.add_ref_frame(capt.pop().detach(),(rec.detach()*2-1).half())
   assert not pm.dpb[0].frame.requires_grad and not pm.dpb[0].feature.requires_grad
 finally:hook.remove()
 return torch.cat(out),torch.cat(real),gs,replay
