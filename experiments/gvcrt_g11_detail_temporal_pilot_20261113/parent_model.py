from util import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from alignment import grid,candidates,shared_valid

class Residual(nn.Module):
    def __init__(self):
        super().__init__();self.a=nn.Conv2d(128,128,3,padding=1,bias=False);self.b=nn.Conv2d(128,128,3,padding=1,bias=False)
    def forward(self,x):return x+self.b(F.silu(self.a(x)))
class Fusion(nn.Module):
    def __init__(self,channels):
        super().__init__();self.E=nn.Sequential(nn.Conv2d(256,128,1,bias=False),nn.SiLU(),Residual(),Residual())
        self.Q=nn.Conv2d(128,32,1,bias=False);self.K=nn.Conv2d(128,32,1,bias=False);self.V=nn.Conv2d(128,128,1,bias=False)
        self.gate=nn.Sequential(nn.Conv2d(388,64,3,padding=1,bias=False),nn.SiLU(),nn.Conv2d(64,1,1,bias=True))
        self.P_h=nn.Sequential(nn.Conv2d(channels,128,1,bias=False),nn.SiLU())
        self.fuse=nn.Sequential(nn.Conv2d(384,128,1,bias=False),nn.SiLU(),Residual(),Residual())
        self.output=nn.Conv2d(128,channels,1,bias=False);nn.init.zeros_(self.output.weight)
    def forward(self,f,past,h,meta,condition='T',intervention=None):
        current=self.E(f);source=self.E(past) if condition=='T' and intervention!='current' else current
        coords=meta['coords'] if condition=='T' else grid(*f.shape[-2:],f.device)
        keys=candidates(self.K(source),coords);values=candidates(self.V(source),coords)
        logits=(self.Q(current)[:,None]*keys).sum(2,keepdim=True)/(32**.5)
        mask=shared_valid(meta);valid=mask.any(1).float()
        # All-invalid sites have finite logits and safe masked normalization.
        weights=torch.softmax(logits.masked_fill(~mask,-1e4),dim=1)*mask
        weights=weights/(weights.sum(1,keepdim=True)+1e-12)
        a=(weights*values).sum(1)
        maximum=logits.masked_fill(~mask,-1e4).max(1).values*valid
        entropy=-(weights*weights.clamp_min(1e-12).log()).sum(1)
        gate=torch.sigmoid(self.gate(torch.cat((current,a,(current-a).abs(),maximum,entropy,meta['fb_error'],valid),1)))*valid
        if intervention=='off':gate=gate*0
        delta=self.output(self.fuse(torch.cat((self.P_h(h),current,gate*a),1)))
        stats=dict(gate_mean=gate.detach().mean(),gate_valid_mean=gate.detach().sum()/valid.sum().clamp_min(1),entropy=entropy.detach().mean(),valid_fraction=valid.mean(),delta_h_over_h_RMS=delta.detach().square().mean().sqrt()/h.detach().square().mean().sqrt().clamp_min(1e-12))
        return h+delta,stats

def restore_b300(fp32=True):
    from audited_b300 import restore_b300 as restore
    c=restore(fp32=fp32)
    # CUDA grid_sample has no bitwise deterministic backward. Keep deterministic
    # kernels where available, explicitly disclose its atomic accumulation.
    torch.use_deterministic_algorithms(True,warn_only=True)
    assert not any(p.requires_grad for p in c['pm'].parameters())
    return c

def tcg(module,current,previous):
    if previous is None:return current
    return current+module.fusion(torch.cat((module.previous_adapter(previous),module.current_adapter(current)),1))

class Receiver:
    """Pure suffix with explicit candidate state, safe for checkpoint recomputation."""
    def __init__(self,c,position):
        self.c=c;self.gen=c['models']['generator'];self.position=position
    def stage(self,module,x,q=None,checkpointing=True):
        def run(z):return module(z,q) if q is not None else module(z)
        return checkpoint(run,x,use_reentrant=False) if checkpointing and torch.is_grad_enabled() and x.requires_grad else run(x)
    def suffix(self,h,x,q,previous=None,checkpointing=True):
        g=self.gen
        if self.position=='stage3':h=self.stage(g.stage3,h,checkpointing=checkpointing);h=g.ada4(h,x)
        state=self.stage(g.stage4,h,q,checkpointing)
        y=tcg(g.tcg_out,state,previous)
        y=g.ada_final(y,x);y=g.head(y)
        return (F.pixel_shuffle(y,8).clamp(-1,1)+1)/2,state
    @torch.no_grad()
    def prefix(self,f,q):
        g=self.gen;b=self.c['models']['bridge'];x=F.pixel_unshuffle(f,2)
        x=b[2](b[1](b[0](x)));x=b[3](x*torch.sigmoid(x))
        h=g.stage1(g.ada1(g.conv_in(x),x));h=g.stage2(g.ada2(h,x));h=g._apply_tcg(g.tcg_low,h,True,True)
        h=g.upsample(g.ada3(h,x));h=g._apply_tcg(g.tcg_mid,h,True,True)
        if self.position=='stage4':h=g.ada4(g.stage3(h),x)
        return h,x,q
