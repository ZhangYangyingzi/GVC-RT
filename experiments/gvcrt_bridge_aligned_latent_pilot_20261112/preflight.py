from util import *
import torch,types,math
from model import *

def mechanical():
 seed();device='cuda';x=torch.arange(20*24,device=device).reshape(1,1,20,24).float();coords=grid(20,24,device)
 for dx in [0,1,-1,.5,-.5]:
  u=torch.zeros(1,2,20,24,device=device);u[:,0]=dx;y=sample(x,coords+u);v=bounds(coords+u,20,24)
  expected=x+dx;assert torch.allclose(y[...,2:-2],expected[...,2:-2],atol=1e-4,rtol=0)
 u=torch.ones(1,2,40,48,device=device)*2;u16=F.interpolate(u,size=(20,24),mode='bilinear',align_corners=False)/2;assert torch.equal(u16,torch.ones_like(u16))
 feat=torch.randn(1,32,8,12,device=device);u=torch.ones(1,2,8,12,device=device)*10000;c,e,v=cost_volume(feat,feat,u);assert not v.any() and torch.isfinite(c).all() and torch.isfinite(e).all()
 stats=dict(f_rms=torch.ones(256),l_rms=torch.ones(18),qp_mean=5.,qp_std=3.);f=torch.randn(1,256,32,32,device=device);l=torch.randn(1,18,16,16,device=device)
 for group in ['S_current','U_unaligned','T_aligned']:
  m=AlignedLatent(stats,group).cuda();y,a=m(l,f,5,'x',1);assert torch.equal(y,l) and not a['q'].any();y,a=m(l+1,f+1,6,'x',2);assert torch.equal(y,l+1)
  if group=='S_current':assert a['q'].eq(.5).all()
  assert 'corrected_l' not in m.previous
  if group!='S_current':assert torch.equal(m.previous['l'],l+1) and not m.previous['l'].requires_grad
  y,a=m(l,f,7,'x',4);assert not a['q'].any();y,a=m(l,f,7,'other',5);assert not a['q'].any();m.reset();assert m.previous is None
 return dict(known_positive_negative_fractional_translation=True,displacement_unit_conversion=True,out_of_bounds_all_candidates_no_NaN=True,original_only_cache=True,frame_video_reset=True)

def run():
 setup();check_protocol();res=mechanical()
 import common_s2 as cs
 # Re-resolve invalid old checkpoint paths only against matching hashes inside this project.
 migrations=[]
 for name,hname in [('B300','B300_SHA'),('INIT','INIT_SHA'),('PILOT','PILOT_SHA'),('BASELINE','BASELINE_SHA')]:
  p=getattr(cs,name);h=getattr(cs,hname)
  if not p.exists():
   r=subprocess.run(['rg','--files','--hidden','-g',p.name,str(PROJECT)],capture_output=True,text=True,check=True)
   matches=[Path(s) for s in r.stdout.splitlines() if Path(s).is_file() and sha(s)==h];assert matches,('MISSING_CHECKPOINT',name,str(p),h)
   setattr(cs,name,matches[0]);migrations.append(dict(name=name,old=str(p),new=str(matches[0]),sha256=h))
  assert sha(getattr(cs,name))==h
 dump(ROOT/'audits/path_migrations.json',migrations)
 from restore import restore_b300
 c=restore_b300();g=c['models']['generator'];g.eval();f=torch.randn(1,256,32,32,device='cuda');stats=dict(f_rms=torch.ones(256),l_rms=torch.ones(18),qp_mean=5.,qp_std=3.)
 frozen=thash(c['pm'].state_dict().items())+thash(c['im'].state_dict().items());bas=[]
 with torch.no_grad():
  g.reset_temporal_buffer()
  for i in range(4):bas.append((c['pm'].recon_generation_net(f+i/100,c['pm'].q_scale_recon[5:6].float())+1)/2)
  for group in ['S_current','U_unaligned','T_aligned']:
   seed();m=AlignedLatent(stats,group).cuda();g.reset_temporal_buffer()
   for i in range(4):
    l=bridge(c,f+i/100);lp,a=m(l,f+i/100,5,'test',i);y=decode(c,lp,5);assert torch.equal(y,bas[i]),('ZERO_BRANCH_MISMATCH',group,i,float((y-bas[i]).abs().max()))
 # Full frozen decoder differentiation, including TCG graph across consecutive corrected latents.
 g.reset_temporal_buffer();z=bridge(c,f).detach().requires_grad_();a=decode(c,z,5,True);z2=bridge(c,f+.01).detach().requires_grad_();b=decode(c,z2,6,True)
 grad=torch.autograd.grad(b.mean(),z,allow_unused=True)[0];assert grad is not None and grad.norm()>0,'TCG_CROSS_FRAME_GRADIENT_MISSING'
 assert all(getattr(g,n)._previous.grad_fn is not None for n in ['tcg_low','tcg_mid','tcg_out'])
 assert not c['pm'].dpb,'CANDIDATE_WROTE_CODEC_DPB'
 assert frozen==thash(c['pm'].state_dict().items())+thash(c['im'].state_dict().items())
 assert not any(p.requires_grad or p.grad is not None for p in c['pm'].parameters())
 loras=[m for m in g.modules() if hasattr(m,'inference_gain')];assert len(loras)==5 and all(m.inference_gain==.5 and m.gain_forward_calls>0 for m in loras)
 csproto=cs.load(cs._ef);parent=cs._parent
 res.update(status='PASS',zero_branch_4_continuous_frames_bitwise_equal=True,decoder_input_gradient=True,TCG_cross_frame_gradient=float(grad.norm()),no_candidate_DPB_writes=True,LoRA_forward_calls=[m.gain_forward_calls for m in loras],LoRA_gain=.5,frozen_model_sha256=frozen,loading_order=['Original','pilot','B1500 TCG','A2000 LoRA','B300 last'],sources={n:dict(path=str(getattr(cs,n)),sha256=sha(getattr(cs,n))) for n in ['B300','INIT','PILOT','BASELINE','RAFT']},provenance=c['provenance'],parameter_count={gr:sum(p.numel() for p in AlignedLatent(stats,gr).parameters()) for gr in ['S_current','U_unaligned','T_aligned']})
 dump(ROOT/'audits/preflight.json',res);event('PREFLIGHT_PASS',frozen_sha256=frozen)

if __name__=='__main__':run()
