"""CDF audit accounting and optional table-forward/continuous-backward rate."""
import math,torch
def table_bits(y,scales,coder):
 # This matches the installed DCVC-FM rANS escape source, not a byte autograd.
 dev=y.device
 cdf=torch.as_tensor(coder.cdfs,dtype=torch.long,device=dev);sizes=torch.as_tensor(coder.cdf_sizes,dtype=torch.long,device=dev);offset=torch.as_tensor(coder.offsets,dtype=torch.long,device=dev)
 ix=((scales.detach().float().clamp(coder.scale_min,coder.scale_max).log()-coder.log_scale_min)/coder.log_scale_step).round().long().clamp(0,coder.scale_levels-1)
 value=y.detach().round().long()-offset[ix];maximum=sizes[ix]-2
 escape=(value<0)|(value>=maximum);raw=torch.where(value<0,-2*value-1,2*(value-maximum)).clamp_min(0)
 symbol=torch.where(escape,maximum,value).clamp_min(0)
 freq=(cdf[ix,symbol+1]-cdf[ix,symbol]).double();assert (freq>0).all()
 n=torch.where(raw>0,torch.floor(torch.log2(raw.double().clamp_min(1))/2)+1,0)
 bypass=2*(n+torch.floor(n/3)+1)
 bits=16-torch.log2(freq)+torch.where(escape,bypass,0)
 return bits.to(y.dtype),dict(escape_symbols=int(escape.sum()),symbols=y.numel(),bypass_bits=float(torch.where(escape,bypass,0).sum()))
def install(model,backend,coder):
 if backend=='official_verified':return
 assert backend=='coder_cdf_forward_continuous_backward'
 original=model.get_y_gaussian_bits
 def corrected(y,scales):
  exact,_=table_bits(y,scales,coder)
  # Gaussian mass with table-range scale clamp; probability floor is a smooth
  # lower bound for a continuous tail approximation. It is not a byte derivative.
  sigma=scales.clamp(coder.scale_min,coder.scale_max)
  gaussian=torch.distributions.Normal(torch.zeros_like(sigma),sigma)
  mass=gaussian.cdf(y+.5)-gaussian.cdf(y-.5)
  smooth=-torch.log2(mass+1/65536)
  return smooth+(exact-smooth).detach()
 model.get_y_gaussian_bits=corrected
 model._v2_original_gaussian_bits=original
 model._v2_known_rate_header_bits=(20+4)*8+11*8/64
