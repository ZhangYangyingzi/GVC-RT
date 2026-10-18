"""CPU-only numerical parity and native-rate-resource lifetime check."""
from preflight import *
import gc,weakref,importlib.util
def main():
 import torch
 from src.entropy_coder import GaussianRansCodec
 import rate_backend as new
 old_path=R/'audits/recovery/rate_backend_before_lifetime_fix.py'
 spec=importlib.util.spec_from_file_location('previous_rate_backend',old_path)
 old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 class Probe(torch.nn.Module):
  def get_y_gaussian_bits(self,y,scales):return y+scales
 torch.manual_seed(20261016);rows=[]
 for trial in range(3):
  coder=GaussianRansCodec();a=Probe();b=Probe()
  old.install(a,'coder_cdf_forward_continuous_backward',coder)
  new.install(b,'coder_cdf_forward_continuous_backward',coder)
  y=torch.randn(256,requires_grad=True);s=(torch.rand(256)*70+.01).requires_grad_(True)
  va=a.get_y_gaussian_bits(y,s);vb=b.get_y_gaussian_bits(y,s)
  ga=torch.autograd.grad(va.sum(),(y,s));gb=torch.autograd.grad(vb.sum(),(y,s))
  assert torch.equal(va,vb) and all(torch.equal(u,v) for u,v in zip(ga,gb))
  wa=weakref.ref(a);wb=weakref.ref(b);del a,b
  old_retained=wa() is not None;new_released=wb() is None
  assert old_retained and new_released
  gc.collect();assert wa() is None
  rows.append(dict(trial=trial,forward_exactly_equal=True,gradients_exactly_equal=True,old_model_self_cycle_confirmed=old_retained,new_model_released_without_cyclic_GC=new_released))
  del coder
 save('audits/recovery/rate_backend_lifetime.json',dict(status='PASS',CPU_only=True,optimizer_updates=0,old_code_sha256=sha(old_path),new_code_sha256=sha(R/'rate_backend.py'),trials=rows,source_codec_and_rans_unmodified=True,crash_cause='self-cycle confirmed and removed; interpreter-shutdown crash causation not asserted without a native backtrace'))
 print('BACKEND LIFETIME AND NUMERICAL PARITY PASS',flush=True)
if __name__=='__main__':main()
