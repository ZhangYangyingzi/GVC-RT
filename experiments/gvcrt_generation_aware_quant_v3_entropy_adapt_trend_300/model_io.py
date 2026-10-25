"""Explicit source resolution, complete Phi checkpoints, freshly rebuilt integer CDFs."""
from util import *
import torch
from runtime import PHI_SCOPE,upgrade,install_phi

def phi_state(pm):return {name:getattr(pm,name).state_dict() for name in PHI_SCOPE}
def phi_hash(pm,forward=False):
 return state_hash({name+'.'+k:(v.half() if forward and v.is_floating_point() else v) for name,sd in phi_state(pm).items() for k,v in sd.items()})
def cdf_hashes(pm):
 out={}
 for name,m in [('z',pm.bit_estimator_z),('y_gaussian',pm.gaussian_encoder)]:
  h=hashlib.sha256()
  for arr in m.get_cdf_info():
   h.update(str(arr.dtype).encode());h.update(str(arr.shape).encode());h.update(arr.tobytes())
  out[name]=h.hexdigest()
 return out

def rebuild_cdf(pm):
 pm.gaussian_encoder.decode_index_cache.clear();pm.gaussian_encoder.decode_zeros_cache.clear()
 pm.masks.clear();pm.update(force_zero_thres=.12)
 pm._cdf_generation_phi=phi_hash(pm,True)
 return dict(Phi_master_sha256=phi_hash(pm),Phi_forward_half_sha256=pm._cdf_generation_phi,CDF=cdf_hashes(pm),fresh_entropy_coder=True,decode_caches_cleared=True)

def assert_cdf(pm):
 assert pm._cdf_generation_phi==phi_hash(pm,True),'stale CDF after Phi change'
 return dict(Phi_master_sha256=phi_hash(pm),Phi_forward_half_sha256=phi_hash(pm,True),CDF=cdf_hashes(pm))

def build_models(branch):
 from gvc_hooks import load_models as native_load
 INIT=Path(next(p for p in load(P/"weight_manifest.json") if Path(p).name=="G_basic_100.pt"))
 im,pm=native_load(torch.device('cuda:0'),force_zero_thres=.12)
 init=torch.load(INIT,map_location='cpu',weights_only=True);assert not init['wrapper']
 pm.recon_generation_net.mlp.load_state_dict(init['bridge'],strict=True)
 pm.recon_generation_net.decoder.load_state_dict(init['generator'],strict=True)
 pm=upgrade(pm,branch);im.requires_grad_(False);pm.requires_grad_(False)
 # Preserve v2 rounding and the Generator's FP32 forward exactly.
 pm.recon_generation_net.decoder.float().requires_grad_(branch in ('N1','Q1'))
 pm.recon_generation_net.mlp.float().requires_grad_(False)
 if pm.use_quant:
  e=load(V2/'checkpoint_index.json')['H11']['0'];assert sha(e['path'])==e['sha256']
  h=torch.load(e['path'],map_location='cpu',weights_only=False)['H'];pm.quant_adjust.load_state_dict(h,strict=True)
  pm.quant_adjust.float().requires_grad_(branch=='Q1')
 if branch in ('N1','Q1'):
  path=next(p for p in load(P/'weight_manifest.json') if Path(p).name=='GVC-RT_P.pt')
  st=torch.load(path,map_location='cpu',weights_only=True);st=st.get('student_ema',st.get('student',st.get('state_dict',st)))
  install_phi(pm,st)
 rebuild_cdf(pm)
 return im,pm

def checkpoint_state(pm):
 return dict(G=pm.recon_generation_net.decoder.state_dict(),H=pm.quant_adjust.state_dict() if pm.use_quant else None,Phi=phi_state(pm))

def restore_model(pm,st):
 pm.recon_generation_net.decoder.load_state_dict(st['G'],strict=True)
 if pm.use_quant:pm.quant_adjust.load_state_dict(st['H'],strict=True)
 if st.get('Phi') is not None:
  assert set(st['Phi'])==set(PHI_SCOPE)
  for name in PHI_SCOPE:getattr(pm,name).load_state_dict(st['Phi'][name],strict=True)
 rebuild_cdf(pm)

def load_checkpoint(branch,step,path=None):
 im,pm=build_models(branch)
 path=Path(path) if path else legacy_checkpoint(branch,step) if branch in ('N0','Q0') else P/'server_artifacts'/branch/f'step_{step}.pt'
 st=torch.load(path,map_location='cpu',weights_only=False)
 assert st['step']==step and st['branch']==({'N0':'A','Q0':'H11'}.get(branch,branch))
 if branch in ('N1','Q1'):assert st['signature']==signature() and st.get('Phi') is not None
 restore_model(pm,st)
 im.eval().requires_grad_(False);pm.eval().requires_grad_(False)
 return im,pm,sha(path)
