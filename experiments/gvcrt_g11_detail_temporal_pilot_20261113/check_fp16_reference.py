"""Read-only two-frame identity check for the actual parent reference timing path."""
from util import *
import torch,copy
from original_reference import build_original
from data import Source
setup_torch();rt,im,pm=build_original();g=copy.deepcopy(pm.recon_generation_net).half().eval().requires_grad_(False);row=videos('validation')[0];src=Source(row);h,w=src.info['native_shape'];results=[]
with torch.no_grad():
 for i in [55,56]:
  f=src.feature(i).cuda().half();q=src.info['actual_qps'][i];out=((g(f,pm.q_scale_recon[q:q+1].half()).float().clamp(-1,1)+1)/2)[...,:h,:w];ref=src.aux(i)['original'].cuda().float();diff=float((out-ref).abs().max());results.append(dict(frame=i,actual_QP=q,max_abs_diff=diff));assert diff<=1e-6,('ORIGINAL_FP16_TIMING_REFERENCE_MISMATCH',diff)
dump(ROOT/'audits/actual_Original_FP16_reference_identity.json',dict(status='PASS',sequence=row['sequence'],frames=results,device=device_info(),GT_not_used=True,actual_parent_reference_not_candidate_RGB=True));print('ACTUAL_PARENT_ORIGINAL_FP16_IDENTITY_PASS',results)
