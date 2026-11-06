"""Synthetic CPU operator check only; never loads or trains an experiment model."""
from pathlib import Path
import hashlib,json,os
E=Path(__file__).resolve().parent
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(E/'tmp'),PYTHONDONTWRITEBYTECODE='1')
(E/'tmp').mkdir(exist_ok=True)
import torch
from torch import nn
from convlora import ParallelConvLoRA

def base_hash(modules):
    h=hashlib.sha256()
    for m in modules:
        for n,t in m.base.state_dict().items():h.update(n.encode());h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    torch.set_num_threads(2);torch.manual_seed(20261106)
    modules=[ParallelConvLoRA(nn.Conv2d(320,320,3,padding=1,groups=320)) for _ in range(4)]
    modules.append(ParallelConvLoRA(nn.Conv2d(320,192,3,padding=1)))
    params=[p for m in modules for p in m.lora.parameters()]
    assert sum(p.numel() for p in params)==59392
    xs=[torch.randn(1,320,8,8) for _ in modules]
    for m,x in zip(modules,xs):assert torch.equal(m(x),m.base(x))
    frozen=base_hash(modules);opt=torch.optim.AdamW(params,lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
    projections=[torch.randn_like(m(x)) for m,x in zip(modules,xs)]
    grads=[]
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        # Disposable synthetic linear VJP, not the formal experimental loss.
        vjp=sum((m(x)*p).sum() for m,x,p in zip(modules,xs,projections));vjp.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert all(p.grad is None for m in modules for p in m.base.parameters())
        down=[float(m.lora.down.weight.grad.norm()) for m in modules];up=[float(m.lora.up.weight.grad.norm()) for m in modules]
        assert all(v>0 for v in up)
        assert all(v==0 for v in down) if step==0 else all(v>0 for v in down)
        grads.append(dict(disposable_step=step+1,down_gradient_norm=down,up_gradient_norm=up))
        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
    assert base_hash(modules)==frozen and all(m.lora.up.weight.norm()>0 for m in modules)
    before=[[p.detach().clone() for p in m.lora.parameters()] for m in modules]
    for m in modules:m.half()
    assert all(p.dtype==torch.float32 for m in modules for p in m.lora.parameters())
    assert all(torch.equal(p,b) for m,bs in zip(modules,before) for p,b in zip(m.lora.parameters(),bs))
    result=dict(status='PASS',scope='Synthetic standalone CPU convolutions only; full model, CUDA dispatch, real loss and RANS not tested',
        rank=4,alpha_div_rank=1,operator_count=5,parameters=59392,zero_initialization_strict_equal=True,
        frozen_base_unchanged=True,base_gradients_absent=True,FP32_values_preserved_through_parent_half=True,
        disposable_optimizer_updates=3,formal_optimizer_updates=0,formal_model_loaded=False,gradients=grads)
    (E/'audits/cpu_lora_operator_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='gradients'},indent=2))

if __name__=='__main__':main()
