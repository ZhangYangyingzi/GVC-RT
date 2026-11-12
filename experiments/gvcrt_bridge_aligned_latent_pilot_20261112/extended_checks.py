from util import *
import torch,types
from model import *

def run():
 setup();seed();from restore import restore_b300
 c=restore_b300();torch.use_deterministic_algorithms(True,warn_only=True);stats=readtensor(ROOT/'cache/statistics.pt');d=waitcache(manifest()['train'][0]['id'],'extended_checks');g=c['models']['generator'];base=[];zero={};lhash=[]
 with torch.no_grad():
  g.reset_temporal_buffer()
  for i in range(1,8):base.append((c['pm'].recon_generation_net(d['features'][i].cuda().float(),c['pm'].q_scale_recon[d['actual_qps'][i]:d['actual_qps'][i]+1].float())+1)/2)
  for group in ['S_current','U_unaligned','T_aligned']:
   m=AlignedLatent(stats,group).cuda();g.reset_temporal_buffer();hashes=[]
   for i in range(1,8):
    f=d['features'][i].cuda().float();l=bridge(c,f);lp,a=m(l,f,d['actual_qps'][i],d['id'],i);y=decode(c,lp,d['actual_qps'][i]);assert torch.equal(y,base[i-1]),('REAL_ZERO_MISMATCH',group,i);hashes.append(thash([('f',f),('l',l)]))
   lhash.append(hashes);zero[group]=True
  assert lhash[0]==lhash[1]==lhash[2] and not c['pm'].dpb
  # A nonzero residual still has an exact identity fallback when every geometric matching candidate is invalid.
  m=AlignedLatent(stats,'T_aligned').cuda();f=d['features'][1].cuda().float();l=bridge(c,f);m(l,f,5,'case',1)
  class Invalid(torch.nn.Module):
   def forward(self,f,p):
    u=f.new_zeros(f.shape[0],2,*f.shape[-2:]);e=f.new_zeros(f.shape[0],2,*f.shape[-2:]);ok=torch.zeros_like(u[:,:1],dtype=torch.bool);return u,e,{'valid8':ok,8:u}
  m.aligner=Invalid();m.fusion.adapter[-1].bias.fill_(.5);lp,a=m(l,f,6,'case',2);assert torch.equal(lp,l) and not a['q'].any() and torch.equal(a['a'],l)
  # Correlation direction: cur(p)=past(p+[1,0]), so current->past +x is the maximum.
  past=torch.randn(1,32,16,20,device='cuda');cur=sample(past,grid(16,20,'cuda')+past.new_tensor([1.,0]).view(1,2,1,1));cv,e,ok=cost_volume(cur,past,torch.zeros(1,2,16,20,device='cuda'));assert (cv.argmax(1)[...,5:-5,5:-5]==41).all()
  # S output is independent of altered historical content with identical current metadata.
  s=AlignedLatent(stats,'S_current').cuda();s.fusion.adapter[-1].bias.fill_(.02);s(l,f,5,'s',1);y1,_=s(l,f,6,'s',2);s.reset();s(l*100,f*-50,1,'s',1);y2,_=s(l,f,6,'s',2);assert torch.equal(y1,y2)
 modelhash=thash([(f'I.{n}',t) for n,t in c['im'].state_dict().items()]+[(f'P.{n}',t) for n,t in c['pm'].state_dict().items()])
 dump(ROOT/'audits/extended_checks.json',dict(status='PASS',zero_real_7_continuous_frames_bitwise_equal=zero,actual_QPs=d['actual_qps'][:8],original_feature_and_latent_hashes=lhash[0],codec_invariance='One independent Original bitstream/cache; exact f and raw Bridge l hashes identical across all candidates; decoder replay leaves codec DPB empty',final_composite_parameter_SHA256=modelhash,all_invalid_full_identity_with_nonzero_residual=True,matching_positive_dx_argmax=41,S_no_historical_content_dependence=True,inference_inputs=['current codec f','current original Bridge latent','past original codec f/Bridge latent','actual current/previous QP','frame/video/reset metadata'],GT_future_flow_inference_inputs=False))
 event('EXTENDED_CHECKS_PASS',parameter_sha256=modelhash)
if __name__=='__main__':run()
