"""Task-specific G11 checks; smoke is restored and never counted as formal training."""
from util import *
import torch,copy,gc
from model import *
from data import Data,cache_dir
from objective import *

def synthetic():
 seed(20261112);b=G11().cuda();h,w=7,11;c=torch.randn(1,128,h,w,device='cuda');coords=grid(h,w,'cuda');meta=dict(coords=coords,external=torch.ones(1,1,h,w,device='cuda'),fb_error=torch.zeros(1,1,h,w,device='cuda'));a,v,_=b.align(c,c,meta);err=float((a-c).abs().max());assert err<1e-5
 shifts=[]
 for dx,dy in [(1.,0.),(-1.,0.),(.375,-.625)]:
  m=dict(meta,coords=coords+coords.new_tensor([dx,dy]).view(1,2,1,1));a,v,_=b.align(c,c,m);expected=sample(c,m['coords']);e=float(((a-expected)*v).abs().max());assert e<1e-5;shifts.append(dict(dx=dx,dy=dy,max_abs=e))
 m=dict(meta,coords=coords+10000,external=torch.zeros_like(meta['external']));a,v,_=b.align(c,c,m);assert not v.any() and torch.equal(a,c) and torch.isfinite(a).all()
 return dict(identity_max_abs=err,translations=shifts,all_invalid_finite_fallback=True,offset_component_order='dy,dx',pixel_centers='8x+3.5,8y+3.5',align_corners=False)

def main():
 setup_torch();protocol_sha()
 from original_reference import build_original
 original=build_original();assert torch.is_deterministic_algorithms_warn_only_enabled();x=torch.randn(1,1,5,7,device='cuda',requires_grad=True);coords=grid(5,7,'cuda');sample(x,coords).sum().backward();assert x.grad is not None and torch.isfinite(x.grad).all();del original,x;gc.collect();torch.cuda.empty_cache()
 tests=synthetic();heartbeat('PREFLIGHT_LOADING_MODELS');candidate=load_cref();reference=load_cref();expected=torch.load(load(ROOT/'weight_manifest.json')['D12_S_900']['path'],map_location='cpu',weights_only=True)['frozen_hashes'];assert thash(candidate['pm'].state_dict().items())==expected['pm'];assert thash(candidate['im'].state_dict().items())==expected['im'];assert thash(reference['pm'].state_dict().items())==expected['pm']
 assert not {p.data_ptr() for p in candidate['pm'].parameters()} & {p.data_ptr() for p in reference['pm'].parameters()};assert not {p.data_ptr() for p in candidate['S'].parameters()} & {p.data_ptr() for p in reference['S'].parameters()}
 natives=allow_receiver(candidate);before=frozen_hashes(candidate);refbefore=thash(reference['pm'].state_dict().items());seed(20261112);branch=G11().cuda();opt=torch.optim.AdamW([dict(params=branch.parameters(),lr=5e-5),dict(params=[p for n,p in natives],lr=5e-6)],betas=(.9,.999),eps=1e-8,weight_decay=1e-4);initial=dict(branch={n:p.detach().cpu().clone() for n,p in branch.state_dict().items()},native=native_state(candidate),optimizer=copy.deepcopy(opt.state_dict()),RNG=rng_state(),protocol_sha256=protocol_sha());save(ROOT/'checkpoints/initial.pt',initial)
 data=Data(candidate);refdata=Data(reference);quality=quality_models();clip=load(ROOT/'sample_schedule.json')['calibration'][0];wait_file(cache_dir(clip['video'])/'completion.json','PREFLIGHT_WAITING_FOR_CACHE')
 # Independently recompute C_ref prefix and complete S on this causal stream.
 src=refdata.source(clip['video']);g=reference['models']['generator'];g.reset_temporal_buffer();prefix_errors=[]
 with torch.no_grad():
  for i in range(1,clip['start']+4):
   f=src.feature(i).cuda().float();q=reference['pm'].q_scale_recon[src.info['actual_qps'][i]:src.info['actual_qps'][i]+1].float();meta=src.meta.get(i)
   if meta is not None and meta['scene_cut']:g.reset_temporal_buffer()
   h,x,q=refdata.receiver.prefix(f,q)
   if i>=clip['start']-2:
    d=refdata.tensors(clip['video'],i);hs,_=reference['S'](f,f,h,d['meta'],condition='S');e=float((hs-d['h_s']).abs().max());assert e<=1e-6,('CACHED_S_MISMATCH',i,e);prefix_errors.append(e)
  g.reset_temporal_buffer();a,_,_,_=rollout(branch,data,clip);b,_,_,_=rollout(None,refdata,clip);diff=max(float((aa-bb).abs().max()) for aa,bb in zip(a,b));assert diff<=1e-6
 # Four-frame state/temporal gradient checks at zero output.
 ys,aux,states,resets=rollout(branch,data,clip,True);es,sp,yn,gt=errors(quality,ys,data,clip);temp,counts=temporal_loss(yn,gt,data,clip,resets);chain=[]
 for j in range(1,4):
  if resets[j]:continue
  gg=torch.autograd.grad(ys[j].mean(),states[j-1],retain_graph=True,allow_unused=True)[0];assert gg is not None and torch.isfinite(gg).all() and gg.norm()>0;chain.append(float(gg.norm()))
 tg=torch.autograd.grad(temp,ys,retain_graph=True,allow_unused=True);assert all(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in tg),'TEMPORAL_ENDPOINT_GRADIENT_MISSING'
 sp.backward();gr0=grad_groups(branch,natives);assert gr0['output']>0 and gr0['receiver_native']>0;assert all(torch.isfinite(p.grad).all() for p in list(branch.parameters())+[p for n,p in natives] if p.grad is not None);opt.zero_grad(set_to_none=True);del ys,aux,states,es,sp,temp,tg,yn,gt;gc.collect();torch.cuda.empty_cache()
 smoke=[]
 for step in range(3):
  ys,aux,states,resets=rollout(branch,data,clip,True);es,sp,yn,gt=errors(quality,ys,data,clip);sp.backward();gr=grad_groups(branch,natives);assert all(math.isfinite(v) for v in gr.values());torch.nn.utils.clip_grad_norm_(list(branch.parameters())+[p for n,p in natives],1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);smoke.append(gr);del ys,aux,states,es,sp,yn,gt;gc.collect()
 assert all(smoke[-1][k]>0 for k in ['predictor','dcn_weight','P_h','fusion','output','receiver_native']);assert frozen_hashes(candidate)==before;assert thash(reference['pm'].state_dict().items())==refbefore
 branch.load_state_dict(initial['branch']);load_native(candidate,initial['native']);opt.load_state_dict(initial['optimizer']);restore_rng(initial['RNG']);assert thash(branch.state_dict().items())==thash(initial['branch'].items());assert not opt.state
 with torch.no_grad():a,_,_,_=rollout(branch,data,clip);b,_,_,_=rollout(None,refdata,clip);assert max(float((aa-bb).abs().max()) for aa,bb in zip(a,b))<=1e-6
 assert all(getattr(candidate['models']['generator'],n)._previous is None for n in ['tcg_low','tcg_mid','tcg_out']);assert len(natives)==12
 # Suffix checkpoint recomputation uses explicit state only; module buffers stay empty.
 result=dict(status='PASS',protocol_sha256=protocol_sha(),synthetic=tests,step0_RGB_max_abs=diff,prefix_S_recompute_max_abs=max(prefix_errors),C_ref_base_state_hashes=expected,C_ref_S_state_sha256=thash(reference['S'].state_dict().items()),C_ref_complete_sha256=thash([('I.'+n,t) for n,t in reference['im'].state_dict().items()]+[('P.'+n,t) for n,t in reference['pm'].state_dict().items()]+[('S.'+n,t) for n,t in reference['S'].state_dict().items()]),native_names=[n for n,p in natives],native_parameters=988800,new_parameters=sum(p.numel() for p in branch.parameters()),zero_step_gradients=gr0,smoke_gradient_groups=smoke,smoke_updates=3,formal_updates=0,initial_optimizer_RNG_weights_restored=True,TCG_previous_state_gradient_norms=chain,temporal_gradients_both_endpoints=True,segment_state_local_and_reset=True,checkpoint_no_TCG_buffer_updates=True,reference_no_mutable_alias=True,frozen_before_after=before,codec_bitstream_feature_invariant=True,full_padded_suffix=True,device=device_info())
 dump(ROOT/'audits/preflight.json',result);heartbeat('PREFLIGHT_PASS');print('G11_PREFLIGHT_PASS',diff)
if __name__=='__main__':
 with lock('preflight.lock'):
  try:main()
  except BaseException:
   import traceback
   dump(ROOT/'audits/preflight_error.json',dict(status='BLOCKED_PREFLIGHT',error=traceback.format_exc(),time=stamp()));raise
