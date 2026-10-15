"""Observe real encoder tensors without changing any numerical result or coder call."""
import types
from audit_io import *

def attach(pm,folder,baseline,alpha_zero):
 import torch,numpy as np
 state={'rows':[],'entropy':[],'current':None,'frame':0}
 prior=pm.compress_prior_2x;process=pm.process_with_mask;encode=pm.entropy_coder.encode_y
 def packed(_self,symbols,group):
  # This is the exact int16 symbol/index array handed to the native RANS coder.
  if state['current'] is not None:
   a=symbols.detach().cpu().numpy().astype(np.int32);index=a&255;sym=a>>8
   coder=pm.gaussian_encoder
   assert group==coder.cdf_group_index
   cdf,length,offset=coder.get_cdf_info();pos=sym-offset[index];tail=length[index]-2
   escape=(pos<0)|(pos>=tail);pos=np.minimum(np.maximum(pos,0),tail);pos=np.where(escape,tail,pos)
   freq=cdf[index,pos+1].astype(np.int64)-cdf[index,pos].astype(np.int64)
   assert (freq>0).all()
   bits=-np.log2(freq.astype(np.float64)/65536.)
   state['entropy'].append(dict(frame=state['frame'],latent_group=state['current'].get('encoded_groups',0),encoded_symbols=len(a),in_support_symbols=int((~escape).sum()),escape_symbols=int(escape.sum()),cdf_entry_bits=float(bits.sum()),in_support_bits=float(bits[~escape].sum()),escape_event_bits=float(bits[escape].sum()),bypass_escape_bits='unavailable',definition='actual 16bit CDF bin mass, excluding escape bypass; not complete stream likelihood'))
   state['current']['encoded_groups']=state['current'].get('encoded_groups',0)+1
  return encode(symbols,group)
 def mask(_self,y,scales,means,m):
  result=process(y,scales,means,m)
  cur=state['current']
  if cur is not None:
   active=m.bool();res=result[0].detach().float()[active];symbols=result[1].detach()[active].float();sc=scales.detach().float()[active]
   eligible=(sc>.12)&(res>-127.5)&(res<126.5)
   dist=(res-(res.floor()+.5)).abs();d=dist[eligible]
   cur['symbols'].add_(result[1].detach().to(torch.int16))
   cur['groups'].append(dict(symbols=len(res),rounding_boundary_eligible=int(eligible.sum()),distance_sum=float(d.sum()),near_count=int((d<=.05).sum()),skip_count=int((sc<=.12).sum()),saturation_count=int(((res<=-127.5)|(res>=126.5)).sum()),scale_skip_boundary_abs_mean=float((sc-.12).abs().mean()),normalized_quant_error_L1=float((res-symbols).abs().mean())))
  return result
 def quant(_self,y,params,spatial):
  # y here is the actual analysis transform output, before clamp_reciprocal_with_quant.
  state['frame']+=1;i=state['frame'];raw=y.detach().clone();state['current']={'symbols':torch.zeros_like(y,dtype=torch.int16),'groups':[]}
  out=prior(y,params,spatial);cur=state['current'];hat=out[-1].detach();sym=cur['symbols'];err=raw.float()-hat.float();groups=cur['groups'];n=sum(g['rounding_boundary_eligible'] for g in groups)
  row=dict(frame=i,prequant_mean=float(raw.float().mean()),prequant_std=float(raw.float().std(unbiased=False)),prequant_abs_mean=float(raw.float().abs().mean()),quantized_mean=float(hat.float().mean()),quantized_std=float(hat.float().std(unbiased=False)),quant_error_L1=float(err.abs().mean()),quant_error_RMSE=float(err.square().mean().sqrt()),quant_error_relative_L2=float(err.norm()/raw.float().norm().clamp_min(1e-12)),distance_to_nearest_quantization_boundary=sum(g['distance_sum'] for g in groups)/n if n else None,fraction_near_quantization_boundary=sum(g['near_count'] for g in groups)/n if n else None,rounding_boundary_eligible=n,skip_fraction=sum(g['skip_count'] for g in groups)/sym.numel(),saturation_fraction=sum(g['saturation_count'] for g in groups)/sym.numel(),scale_skip_boundary_abs_mean=sum(g['scale_skip_boundary_abs_mean']*g['symbols'] for g in groups)/sym.numel(),y_sha256=tensor_hash({'y':raw}),y_hat_sha256=tensor_hash({'y_hat':hat}),symbols_sha256=tensor_hash({'symbols':sym}),boundary_units='conditional mean-centered normalized residual; active nonsaturated symbols only',y_hat_semantics='conditional means restored then multiplied by actual q_dec; continuous reconstruction, not integer symbols')
  base=baseline/f'frame{i:02d}.npz'
  if alpha_zero:
   base.parent.mkdir(parents=True,exist_ok=True);tmp=base.with_suffix('.tmp')
   with tmp.open('wb') as f:np.savez(f,y=raw.cpu().numpy(),symbols=sym.to(torch.int8).cpu().numpy())
   tmp.replace(base);row.update(prequant_shift_vs_alpha0_L1=0.,prequant_shift_vs_alpha0_relative_L2=0.,fraction_quantized_symbols_changed=0.)
  elif base.exists():
   with np.load(base) as ref:
    a=torch.from_numpy(ref['y'].copy()).to(raw.device).float();s=torch.from_numpy(ref['symbols'].copy()).to(sym.device);diff=raw.float()-a
    row.update(prequant_shift_vs_alpha0_L1=float(diff.abs().mean()),prequant_shift_vs_alpha0_relative_L2=float(diff.norm()/a.norm().clamp_min(1e-12)),fraction_quantized_symbols_changed=float((sym!=s).float().mean()))
  state['rows'].append(row)
  # Keep current through encode_y (which happens AFTER compress_prior_2x returns).
  return out
 pm.process_with_mask=types.MethodType(mask,pm);pm.compress_prior_2x=types.MethodType(quant,pm);pm.entropy_coder.encode_y=types.MethodType(packed,pm.entropy_coder)
 return state
