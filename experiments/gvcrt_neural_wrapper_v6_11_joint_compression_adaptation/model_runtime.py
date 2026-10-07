"""FP32 core masters, differentiable native-half compute, exact source optimizer."""
import random,math
import torch
from v68_io import *

def restore_rng(s):
    random.setstate(s['sample_rng_state']);torch.set_rng_state(s['torch_rng_state']);torch.cuda.set_rng_state(s['cuda_rng_state'])

def norm(gs):return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gs if g is not None))

def core_parameters(pm,models):
    excluded={id(p) for k in ('bridge','generator') for p in models[k].parameters()}
    return {n:p for n,p in pm.named_parameters() if id(p) not in excluded}

def setup(device,joint=False,quality=True):
    v4=module('v611_training_core',V4/'train.py');cfg=load(ROOT/'config.json')
    wrapper=v4.NeuralWrapper().to(device).float().train();im,pm=v4.load_models(device,force_zero_thres=.12)
    bridge=pm.recon_generation_net.mlp.float().train().requires_grad_(True)
    generator=pm.recon_generation_net.decoder.float().train().requires_grad_(True)
    models=dict(wrapper=wrapper,bridge=bridge,generator=generator)
    s=torch.load(SOURCE,map_location='cpu',weights_only=True)
    for k,m in models.items():m.load_state_dict(s[k],strict=True)
    oc=cfg['optimizer']
    opt=torch.optim.AdamW([dict(params=list(m.parameters()),lr=oc[k+'_lr'],name=k) for k,m in models.items()],weight_decay=oc['weight_decay'])
    opt.load_state_dict(s['optimizer']);assert state_hash(opt.state_dict())==state_hash(s['optimizer'])
    assert v4.compression_hash(im,pm)==cfg['compression_hash']
    core=core_parameters(pm,models)
    if joint:
        for p in core.values():p.data=p.data.float();p.requires_grad_(True)
        opts={k:v for k,v in opt.param_groups[0].items() if k not in ('params','name','lr','initial_lr')}
        opt.add_param_group(dict(params=list(core.values()),name='compression_core',lr=1e-6,**opts))
        assert all(p not in opt.state for p in core.values())
        original=v4.joint_ste_forward
        class Call(torch.nn.Module):
            def __init__(self):super().__init__();self.pm=pm
            def forward(self,proxy,qp):return original(self.pm,proxy,qp)
        call=Call()
        def adapted(model,proxy,qp):
            assert model is pm
            return torch.func.functional_call(call,{'pm.'+n:p.to(torch.float16) for n,p in core.items()},(proxy,qp),strict=False)
        v4.joint_ste_forward=adapted
    ids=[id(p) for g in opt.param_groups for p in g['params']];assert len(ids)==len(set(ids))
    assert all(not p.requires_grad for p in im.parameters())
    qm=v4.quality_models(device) if quality else None
    restore_rng(s)
    return v4,im,pm,models,opt,qm,s

def deployment_hash(im,pm,v4):
    return tensor_hash(dict([(f'i.{n}',p.half()) for n,p in im.named_parameters()]+[(f'p.{n}',p.half()) for n,p in pm.named_parameters() if not n.startswith('recon_generation_net.')]))

def precision_metadata(im,pm,models,v4):
    core=core_parameters(pm,models)
    return dict(compression_hash=deployment_hash(im,pm,v4),I_hash=v4.module_hash(im),
        master_core_hash=state_hash({n:p.detach() for n,p in core.items()}),
        deployment_core_hash=state_hash({n:p.detach().half() for n,p in core.items()}),
        master_dtypes=sorted(set(str(p.dtype) for p in core.values())),deployment_dtype='torch.float16')

def inference_state(im,pm,models,v4):
    return dict(**{k:{n:p.detach().cpu().clone() for n,p in m.state_dict().items()} for k,m in models.items()},
        full_p={n:p.detach().cpu().half().clone() for n,p in pm.state_dict().items()},
        precision=precision_metadata(im,pm,models,v4))

def inventory(im,pm,models,opt,quality,branch):
    groups={id(p):(g['name'],g['lr']) for g in opt.param_groups for p in g['params']}
    items=[('wrapper.'+n,p) for n,p in models['wrapper'].named_parameters()]+[('pm.'+n,p) for n,p in pm.named_parameters()]+[('im.'+n,p) for n,p in im.named_parameters()]
    items += [(f'quality{i}.'+n,p) for i,m in enumerate(quality or ()) for n,p in m.named_parameters()]
    return [dict(branch=branch,name=n,module='.'.join(n.split('.')[:2]),dtype=str(p.dtype),requires_grad=p.requires_grad,group=groups.get(id(p),('frozen',0))[0],lr=groups.get(id(p),('frozen',0))[1],numel=p.numel()) for n,p in items]
