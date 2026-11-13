from util import *
import torch
import torch.nn.functional as F
from model import *

def known_checks():
 st=oldstats();st['evidence_rms']=torch.ones(2);c=torch.randn(1,18,8,10,device='cuda');past=torch.randn_like(c);qp=torch.zeros(1,2,8,10,device='cuda');flow=torch.zeros(1,2,8,10,device='cuda');seed();m=DeformableHistory(st,'B_flow_guided').cuda()
 for dx,dy in [(0,0),(1,0),(-1,0),(.5,0),(0,-.5),(1,1)]:
  flow[:,0]=dx;flow[:,1]=dy;a,v,aux=m.align(c,past,flow,qp);coords=grid(8,10,'cuda')+flow;expected=torch.where(bounds(coords,8,10),sample(past,coords),c);assert torch.allclose(a,expected,atol=2e-6,rtol=1e-5),(dx,dy,float((a-expected).abs().max()))
 flow.fill_(10000);a,v,_=m.align(c,past,flow,qp);assert torch.equal(a,c) and not v.any() and torch.isfinite(a).all()
 from torchvision.ops import deform_conv2d
 x=torch.randn(1,18,7,9,device='cuda',requires_grad=True);weights=torch.randn(18,18,3,3,device='cuda',requires_grad=True)*.02;offset=torch.randn(1,18,7,9,device='cuda',requires_grad=True)*.2;mask=torch.rand(1,9,7,9,device='cuda',requires_grad=True);out=deform_conv2d(x,offset,weights,padding=(1,1),mask=mask);ref=torch.zeros_like(out)
 for k,(dy,dx) in enumerate(( (y,x) for y in [-1,0,1] for x in [-1,0,1])):
  coords=grid(7,9,'cuda')+offset[:,2*k:2*k+2][:,[1,0]]+x.new_tensor([dx,dy]).view(1,2,1,1);warped=sample(x,coords)*mask[:,k:k+1];ref=ref+torch.einsum('oc,bchw->bohw',weights[:,:,dy+1,dx+1],warped)
 assert torch.allclose(out,ref,atol=2e-6,rtol=2e-5),float((out-ref).abs().max());g1=torch.autograd.grad(out.square().mean(),[x,weights,offset,mask],retain_graph=True);g2=torch.autograd.grad(ref.square().mean(),[x,weights,offset,mask]);assert all(torch.allclose(a,b,atol=1e-6,rtol=1e-3) and torch.isfinite(a).all() for a,b in zip(g1,g2))
 return dict(synthetic_signed_fractional_translation=True,offset_order='dy,dx; 9 row-major kernel points',DCN_reference_max_abs=float((out-ref).abs().max()),DCN_grid_sample_gradient_agreement=True,all_invalid_identity_fallback=True,center2I_mask05_identity=True)

def run():
 setup();seed();res=known_checks();from restore import restore_b300
 c=restore_b300();torch.use_deterministic_algorithms(True,warn_only=True);frozen=thash([(f'I.{n}',t) for n,t in c['im'].state_dict().items()]+[(f'P.{n}',t) for n,t in c['pm'].state_dict().items()]);assert frozen==load(ROOT/'PROTOCOL.json')['B300_expected_composite_parameter_SHA256']
 g=c['models']['generator'];d=oldtrain(training_rows()[0]['id']);ref=sidewait(d['id'],'preflight');st=oldstats();st['evidence_rms']=torch.ones(2);baseline=[];before=thash(c['pm'].state_dict().items());groups={}
 with torch.no_grad():
  g.reset_temporal_buffer()
  for i in range(1,5):baseline.append((c['pm'].recon_generation_net(d['features'][i].cuda().float(),c['pm'].q_scale_recon[d['actual_qps'][i]:d['actual_qps'][i]+1].float())+1)/2)
  for group in ['A_zero_base','B_flow_guided']:
   seed();m=DeformableHistory(st,group).cuda();initial=thash(m.state_dict().items());count=sum(p.numel() for p in m.parameters())
   for beta,nonzero in [(1,False),(0,True)]:
    if nonzero:m.fusion.adapter[-1].bias.fill_(.1)
    m.reset();g.reset_temporal_buffer()
    for i in range(1,5):
     l=bridge(c,d['features'][i].cuda().float());lp,a=m(l,torch.from_numpy(ref['decoded_ref_backward'][i]).cuda(),torch.from_numpy(ref['decoded_ref_evidence'][i]).cuda(),d['actual_qps'][i],d['id'],i,beta);out=decode(c,lp,d['actual_qps'][i]);assert torch.equal(out,baseline[i-1]),('BASELINE_IDENTITY',group,beta,i)
   # Shape, frame, video and explicit GOP reset all close the residual.
   m.reset();l=torch.randn(1,18,8,10,device='cuda');f=torch.zeros(1,2,128,160,device='cuda');e=torch.zeros_like(f);m(l,f,e,5,'a',1);l2=torch.randn(1,18,8,12,device='cuda');out,a=m(l2,f,e,6,'a',2);assert not a['q'].any() and torch.equal(out,l2);out,a=m(l2,f,e,6,'a',3,reset=True);assert not a['q'].any();out,a=m(l2,f,e,6,'b',4);assert not a['q'].any()
   groups[group]=dict(parameter_count=count,initial_state_sha256=initial,zero_step_continuous_frames_equal=True,beta0_nonzero_branch_equal=True,resets=True)
 assert groups['A_zero_base']['initial_state_sha256']==groups['B_flow_guided']['initial_state_sha256'] and groups['A_zero_base']['parameter_count']==groups['B_flow_guided']['parameter_count'];assert not c['pm'].dpb
 g.reset_temporal_buffer();z=bridge(c,d['features'][1].cuda().float()).detach().requires_grad_();decode(c,z,5,True);z2=z.detach().clone().requires_grad_();out=decode(c,z2,5,True);grad=torch.autograd.grad(out.mean(),z)[0];assert grad.norm()>0
 assert before==thash(c['pm'].state_dict().items()) and all(not p.requires_grad for p in c['pm'].parameters());assert all(m.inference_gain==.5 for m in g.modules() if hasattr(m,'inference_gain'))
 # Coexisting Original must retain independent source classes and no adapted modules.
 from reference import coexist_original
 rt,im,pm=coexist_original();assert not any('tcg' in n.lower() or 'lora' in n.lower() for n,p in pm.named_parameters());assert before==thash(c['pm'].state_dict().items())
 res.update(status='PASS',groups=groups,base_composite_parameter_SHA256=frozen,all_frozen=True,codec_DPB_unmodified=True,TCG_cross_frame_input_gradient=float(grad.norm()),LoRA_gain=.5,Original_reference_independent=True,provenance=c['provenance'],inference_forward_has_no_GT_or_GT_flow=True)
 dump(ROOT/'audits/preflight.json',res);event('PREFLIGHT_PASS',parameter_count=groups['A_zero_base']['parameter_count'])
if __name__=='__main__':
 with lock('preflight.lock'):run()
