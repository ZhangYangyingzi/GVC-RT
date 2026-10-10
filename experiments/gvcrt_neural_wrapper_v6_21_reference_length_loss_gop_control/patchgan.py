"""70x70 spectral-normalized PatchGAN, independent RNG and hinge updates."""
from io21 import *
def create(device):
    import torch
    from torch import nn
    seed=load(ROOT/'config.json')['discriminator']['seed']
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        layers=[];channels=(3,64,128,256,512,1)
        for i,stride in enumerate((2,2,2,1,1)):
            layers.append(nn.utils.spectral_norm(nn.Conv2d(channels[i],channels[i+1],4,stride,1)))
            if i<4:layers.append(nn.LeakyReLU(.2))
        model=nn.Sequential(*layers).to(device).float().train()
        with torch.no_grad():
            probe=torch.rand(1,3,256,256,device=device)*2-1
            for _ in range(10):score=model(probe)
            assert score.shape==(1,1,30,30) and torch.isfinite(score).all()
        rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
    model.eval().requires_grad_(False)
    opt=torch.optim.Adam(model.parameters(),lr=2e-4,betas=(0.,.9),weight_decay=0)
    return model,opt,rng
def update(model,opt,real,fakes):
    import torch
    import torch.nn.functional as F
    assert len(real)==len(fakes)==15 and all(not f.requires_grad for f in fakes)
    model.train().requires_grad_(True);opt.zero_grad(set_to_none=True);loss=0.
    for r,f in zip(real,fakes):
        value=.5*(F.relu(1-model(r.detach()*2-1)).mean()+F.relu(1+model(f*2-1)).mean())
        assert torch.isfinite(value)
        (value/15).backward();loss+=float(value.detach())/15
    norm=sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None)**.5
    assert norm>0 and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    opt.step();assert all(torch.isfinite(p).all() for p in model.parameters())
    opt.zero_grad(set_to_none=True);model.eval().requires_grad_(False)
    return dict(D_loss=loss,D_grad_norm=norm,D_optimizer_updates=1)

