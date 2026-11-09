# Independent four-window gradient reference

Executed without optimizer updates. Production six-component segment VJPs are
compared against one combined-objective VJP per segment. Both use the same four
sealed ULong windows. This document's SHA256 is recorded in the result.

```python
from common_s2 import *
from experiment_model import construct,reconstruct,detach_history
from model_context import assert_reset,frozen_items
from cache_codec import cache_paths
from temporal import temporal_errors
from loss import window_gradients,gradient_norm
import torch
import torch.nn.functional as F
env();c=construct('A',True,0);seed(PLAN_SEED);guard=FrozenGuard(frozen_items(c));before=tensor_hash(c['named'])
lam=load(ROOT/'audits/calibration.json')['lambda_temp'];production=[torch.zeros_like(p) for p in c['params']];reference=[torch.zeros_like(p) for p in c['params']]
rows=plan_rows()[:4];sample_keys=[]
for row in rows:
    key,path,rp=cache_paths(row,row['external_qp'],True);assert sha(path)==load(rp)['cache_sha256'];data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    gradients,stats=window_gradients(c,data,lam)
    for dst,g in zip(production,gradients):dst.add_(g/4)
    sample_keys.append(row['sample_key']);del data,gradients
for row in rows:
    key,path,rp=cache_paths(row,row['external_qp'],True);data=torch.load(path,map_location='cpu',weights_only=True,mmap=True);assert_reset(c)
    def gt(i):return data['GT'][i].permute(2,0,1)[None].to(c['device']).float()/255
    with torch.no_grad():
        previous=c['core'].unit(data['original'][0].to(c['device']))
        for i in range(1,49):previous=c['core'].unit(reconstruct(c,data['features'][i].to(c['device']),data['info']['actual_qps'][i]))
    for first,last in [(49,52),(53,56),(57,60),(61,63)]:
        detach_history(c);previous=previous.detach();objective=0
        for i in range(first,last+1):
            target=gt(i);out=c['core'].unit(reconstruct(c,data['features'][i].to(c['device']),data['info']['actual_qps'][i],retain=True))
            spatial=c['quality'][0](out,target,normalize=True).mean()+.5*c['quality'][1](out,target,require_grad=True).mean()+.1383589248954108*F.l1_loss(out,target)
            a,b=c['transform'](previous,out);motion=F.l1_loss(c['raft'](a,b)[-1],data['GT_forward'][i].to(c['device']))
            rgb,edge,valid=temporal_errors(out,previous,target,gt(i-1),data['GT_backward'][i].to(c['device']),data['GT_valid'][i].to(c['device']))
            objective+=(spatial+.05*motion+lam*(rgb+.5*edge))/60
            previous=out
        gradients=torch.autograd.grad(objective,c['params'],allow_unused=True)
        for dst,g in zip(reference,gradients):
            if g is not None:dst.add_(g.detach())
        detach_history(c);previous=previous.detach();c['raft'].corr_block.corr_pyramid=[t.detach() for t in c['raft'].corr_block.corr_pyramid]
        del objective,gradients,out,spatial,motion,rgb,edge,a,b
    assert_reset(c);del data
absolute=gradient_norm([a-b for a,b in zip(production,reference)]);relative=absolute/max(gradient_norm(reference),1e-20)
assert relative<1e-3,('COMBINED_VJP_REFERENCE_MISMATCH',relative)
for p,g in zip(c['params'],production):p.grad=g.clone()
torch.nn.utils.clip_grad_norm_(c['params'],1.,error_if_nonfinite=True);clipped=[p.grad.clone() for p in c['params']]
for p,g in zip(c['params'],reference):p.grad=g.clone()
torch.nn.utils.clip_grad_norm_(c['params'],1.,error_if_nonfinite=True)
clip_error=gradient_norm([a-p.grad for a,p in zip(clipped,c['params'])])/max(gradient_norm(clipped),1e-20)
assert clip_error<1e-3
assert tensor_hash(c['named'])==before and len(c['optimizer'].state)==0;guard.full()
dump(ROOT/'audits/four_window_gradient_reference.json',dict(status='PASS',effective_batch=4,distinct_groups=len({r['source_group'] for r in rows}),sample_keys=sample_keys,
    production_norm=gradient_norm(production),combined_objective_reference_norm=gradient_norm(reference),absolute_L2=absolute,relative_L2=relative,post_clip_relative_L2=clip_error,tolerance=1e-3,
    normalization='15 supervised frames times4 windows =60; exactly once',clip_count_per_reference=1,optimizer_updates=0,weights_unchanged=True,frozen_exact_bytes_unchanged=True,
    driver_SHA256=sha(ROOT/'audits/four_window_gradient_reference.md'),physical_GPU=4,time=time.time()))
print('FOUR_WINDOW_COMBINED_VJP_REFERENCE_PASS',relative,clip_error,flush=True)
```
