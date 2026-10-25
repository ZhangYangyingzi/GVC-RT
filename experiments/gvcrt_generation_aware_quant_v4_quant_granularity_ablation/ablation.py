"""All modes share immutable v2 H11 coordinates; no spatial side information."""
import torch
from runtime import SkipDomainDMC
from util import MODES
def transform(raw,mode):
 assert mode in MODES and raw.ndim==4
 if mode in ('A','IDENTITY'):return torch.ones_like(raw)
 if mode=='FRAME':return raw.mean((1,2,3),keepdim=True).expand_as(raw)
 if mode=='CHANNEL':return raw.mean((2,3),keepdim=True).expand_as(raw)
 return raw
class GranularityDMC(SkipDomainDMC):
 def quant_step(self,params):
  base=params[:,:128].clamp_min(.5)
  raw=self.quant_adjust(params,self.current_qp) if self.use_quant else torch.ones_like(base[:,:1],dtype=torch.float32)
  variant=transform(raw,self.mode)
  q=(base.float()*variant).clamp_min(.5).to(base.dtype)
  self.maps={'q_base':base,'qmap':q,'rho':q/base,'raw_multiplier':raw,'variant_multiplier':variant,'effective_multiplier':q/base}
  self.clamp_fraction=float((base.float()*variant<.5).float().mean())
  return q
def upgrade(pm,mode):
 assert isinstance(pm,SkipDomainDMC)
 pm.__class__=GranularityDMC;pm.mode=mode;pm.domain=True;pm.fix_skip=True
 return pm
