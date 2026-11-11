"""Implementation checks only. No quality threshold, no latent/RMS diagnostics."""
from common_s2 import *
import copy


def main():
    env();import torch
    from adapter import CausalAdapter,decode_head
    from experiment_model import construct,reconstruct
    from model_context import assert_reset
    from cache_codec import cache_paths
    from cache_h import paths,ready
    from loss import load_training_context,window,norm,frame
    row=plan_rows()[0]
    while not ready(row) or not (ROOT/'checkpoints/common_adapter_step0.pt').exists():time.sleep(2)
    device=torch.device('cuda',0);seed(TRAIN_SEED);torch.cuda.set_device(0)
    branch=CausalAdapter().to(device)
    assert sum(p.numel() for p in branch.parameters())==21696
    # Give B nonzero weights so causality/cache tests cannot pass trivially at zero.
    with torch.no_grad():branch.B.weight.normal_(0,.01)
    inputs=[torch.randn(1,320,8,9,device=device) for _ in range(8)]
    def stream(xs,reset=True):
        if reset:branch.reset()
        return [branch(x).detach() for x in xs]
    with torch.no_grad():
        whole=stream(inputs)
        chunk=stream(inputs[:3])+stream(inputs[3:6],False)+stream(inputs[6:],False)
        assert all(torch.equal(a,b) for a,b in zip(whole,chunk))
        future=stream(inputs[:4]+[torch.randn_like(x)*10 for x in inputs[4:]])
        assert all(torch.equal(a,b) for a,b in zip(whole[:4],future[:4]))
        stream([torch.randn_like(inputs[0])*100]);reset=stream(inputs)
        assert all(torch.equal(a,b) for a,b in zip(whole,reset))
        assert all(torch.equal(a,b) for a,b in zip(branch.history,inputs[-2:]))
        branch.reset();first=branch(inputs[0]);reference=branch.from_three([inputs[0]]*3)
        assert torch.equal(first,reference)
    _,p,rp=cache_paths(row,row['external_qp'],True);assert sha(p)==load(rp)['cache_sha256']
    raw=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
    matches=[];provenance=[]
    for train_precision in (True,False):
        baseline=construct('B300',train_precision,0,with_adapter=False);before=tensor_hash(baseline['pm'].state_dict().items())
        outs=[]
        with torch.no_grad():
            for i in range(1,7):outs.append(reconstruct(baseline,raw['features'][i],raw['info']['actual_qps'][i]).cpu())
        assert tensor_hash(baseline['pm'].state_dict().items())==before
        provenance.append(baseline['provenance']);del baseline;torch.cuda.empty_cache()
        c=construct('B300',train_precision,0);gen=c['models']['generator'];head_calls=gen.head.lora.forward_calls
        with torch.no_grad():
            for i in range(1,7):
                out=reconstruct(c,raw['features'][i],raw['info']['actual_qps'][i]).cpu()
                assert torch.equal(out,outs[i-1]),('ZERO_BRANCH_MISMATCH',train_precision,i)
        assert c['adapter'].forward_calls==6 and gen.head.lora.forward_calls-head_calls==6
        assert len(c['adapter'].history)==2
        assert_reset(c);assert not c['adapter'].history
        assert all(not p.requires_grad for n,p in c['pm'].named_parameters() if 'causal_adapter.' not in n)
        matches.append(dict(precision='FP32 training' if train_precision else DEPLOYMENT,
            exact_equal_frames=6,adapter_calls=6,head_LoRA_calls=6,history_bytes_at_256x256=2*320*32*32*(4 if train_precision else 2)))
        del c;torch.cuda.empty_cache()
    c=load_training_context(device);p,rp=paths(row);d=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
    frozen=tensor_hash(c['head'].state_dict().items());params=list(c['adapter'].parameters())
    result,_=window(c,d,lambda_temp=1.)
    g0={n:norm([p.grad]) for n,p in c['adapter'].named_parameters()}
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert norm([p.grad for p in c['adapter'].B.parameters()])>0
    assert norm([p.grad for m in (c['adapter'].A,c['adapter'].temporal) for p in m.parameters()])==0
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
    opt.step();opt.zero_grad(set_to_none=True);window(c,d,lambda_temp=1.)
    g1={n:norm([p.grad]) for n,p in c['adapter'].named_parameters()}
    assert all(v>0 and math.isfinite(v) for v in g1.values())
    assert tensor_hash(c['head'].state_dict().items())==frozen
    assert all(p.grad is None and not p.requires_grad for p in c['head'].parameters())
    assert all(p.grad is None and not p.requires_grad for m in c['quality'] for p in m.parameters())
    # Explicitly check the frozen head input has a nonzero image-loss gradient.
    x=d['h'][2:3].to(device).clone().requires_grad_();image=decode_head(c['head'],x)
    image.square().mean().backward();assert torch.isfinite(x.grad).all() and x.grad.norm()>0
    rec=dict(status='PASS',parameters=21696,zero_branch=matches,causality=True,chunked_equal=True,
        reset_no_cross_video=True,raw_h_history_only=True,left_replication=True,
        loss_finite=True,gradients_at_zero=g0,gradients_after_disposable_B_update=g1,
        frozen_head_parameters_unchanged=True,frozen_perceptual_parameters=True,
        frozen_head_input_gradient_norm=float(x.grad.norm()),training_and_deployment_adapter_executed=True,
        head_LoRA_preserved_and_executed=True,disposable_updates_not_formal_training=True,
        provenance=provenance,independent_RMS_diagnosis=False,time=time.time())
    dump(ROOT/'audits/preflight.json',rec);event('preflight_pass',parameters=21696)

if __name__=='__main__':main()
