"""Exact source restoration and independent native Original GVC-RT teacher."""
import random,math
from v68_io import *
def restore_rng(s):
    import torch
    random.setstate(s['sample_rng_state']);torch.set_rng_state(s['torch_rng_state']);torch.cuda.set_rng_state(s['cuda_rng_state'])
def setup(device,cp=SOURCE,quality=True):
    import torch
    v4=module('v68_training_core',V4/'train.py');cfg=load(ROOT/'config.json')
    wrapper=v4.NeuralWrapper().to(device).float().train();im,pm=v4.load_models(device,force_zero_thres=.12)
    bridge=pm.recon_generation_net.mlp.float().train().requires_grad_(True);generator=pm.recon_generation_net.decoder.float().train().requires_grad_(True)
    models=dict(wrapper=wrapper,bridge=bridge,generator=generator)
    s=torch.load(cp,map_location='cpu',weights_only=True)
    for k,m in models.items():m.load_state_dict(s[k],strict=True)
    oc=cfg['optimizer'];opt=torch.optim.AdamW([dict(params=list(m.parameters()),lr=oc[k+'_lr'],name=k) for k,m in models.items()],weight_decay=oc['weight_decay'])
    opt.load_state_dict(s['optimizer']);assert state_hash(opt.state_dict())==state_hash(s['optimizer'])
    assert v4.compression_hash(im,pm)==cfg['compression_hash'];ids={id(p) for m in models.values() for p in m.parameters()}
    assert all(not p.requires_grad for m in (im,pm) for p in m.parameters() if id(p) not in ids)
    qm=v4.quality_models(device) if quality else None;restore_rng(s)
    return v4,im,pm,models,opt,qm,s

def norm(gs):return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gs if g is not None))
