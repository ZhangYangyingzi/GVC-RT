from util import *
import torch
from torch import nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self):
        super().__init__();self.a=nn.Conv2d(64,64,3,padding=1,bias=False);self.b=nn.Conv2d(64,64,3,padding=1,bias=False)
    def forward(self,u):return u+self.b(F.silu(self.a(u)))

class Detail(nn.Module):
    def __init__(self, scales):
        super().__init__()
        self.register_buffer('f_rms',scales['f_rms'].reshape(1,256,1,1).float())
        self.register_buffer('h_rms',scales['h_rms'].reshape(1,320,1,1).float())
        self.f_proj=nn.Conv2d(256,64,1,bias=False);self.h_proj=nn.Conv2d(320,64,1,bias=False)
        self.fuse=nn.Conv2d(258,64,1,bias=False);self.blocks=nn.Sequential(Residual(),Residual());self.output=nn.Conv2d(64,320,1,bias=False)
        nn.init.zeros_(self.output.weight)
    def forward(self,f,h,p1,p2,m1,m2):
        z=[F.silu(self.f_proj(x/self.f_rms)) for x in [f,p1,p2]]+[F.silu(self.h_proj(h/self.h_rms)),m1,m2]
        return self.output(self.blocks(F.silu(self.fuse(torch.cat(z,1)))))

def restore_b300(fp32=True):
    from model_context import build
    from common_s2 import B300,B300_SHA,INIT_SHA,TCG_PREFIXES
    # Use the audited loading chain but immediately discard its temporary optimizer.
    c=build(0,train=fp32,losses=False,group='A_control',fresh_optimizer=True)
    c['optimizer']=None
    assert sha(B300)==B300_SHA
    b=torch.load(B300,map_location='cpu',weights_only=True)
    assert b['experiment_group']=='B' and b['committed_updates']==300 and b['initial_A2000_sha256']==INIT_SHA
    keys={n for n,p in c['pm'].named_parameters() if '.lora.' in n or n.startswith('recon_generation_net.mlp.3.') or n.startswith(tuple('recon_generation_net.decoder.'+s for s in TCG_PREFIXES))}
    assert set(b['trainable_state'])==keys and len(keys)==52
    state=dict(c['pm'].state_dict());state.update(b['trainable_state']);c['pm'].load_state_dict(state,strict=True)
    for n,t in b['trainable_state'].items():assert torch.equal(c['pm'].state_dict()[n].cpu(),t.to(c['pm'].state_dict()[n].dtype))
    for m in [c['im'],c['pm']]:m.eval().requires_grad_(False)
    gen=c['models']['generator'];gen.set_temporal_cascade_grad(False)
    # The FP32 reference avoids custom fused decoder operators. Codec dispatch remains original.
    if fp32:
        import types
        for m in gen.modules():
            if m.__class__.__name__ in ['DepthConvBlock','StageBlock']:
                m.forward=types.MethodType(lambda self,*a,**kw:self.forward_torch(*a,**kw),m)
                if hasattr(m,'proxy'):m.proxy=None
    assert all(m.inference_gain==.5 for m in gen.modules() if hasattr(m,'inference_gain'))
    c['provenance'].update(B300_overlay_last=True,B300_keys=sorted(keys),B300_SHA256=B300_SHA,all_frozen=True,receiver_precision='FP32' if fp32 else 'FP16')
    c['named']=[];c['params']=[];c['trainable_parameter_names']=set()
    assert not any(p.requires_grad for p in c['pm'].parameters())
    return c

def make_head(device='cuda'):
    from convlora import ParallelConvLoRA
    head=ParallelConvLoRA(nn.Conv2d(320,192,3,padding=1),4)
    head.load_state_dict(torch.load(ROOT/'cache/head.pt',map_location='cpu',weights_only=True)['state'],strict=True)
    head.inference_gain=.5
    return head.to(device).float().eval().requires_grad_(False)

def rgb(head,h):return (F.pixel_shuffle(head(h),8).clamp(-1,1)+1)/2

def inputs(data,condition,device='cuda',crop=None):
    names=['f','h','p1','p2','m1','m2']
    xs=[]
    for n in names:
        t=data[n]
        if crop:
            x0,y0,x1,y1=crop;t=t[...,y0:y1,x0:x1]
        xs.append(t.to(device).float().clone())
    if condition=='S':xs[2]=xs[0];xs[3]=xs[0]
    return xs

def candidate(branch,head,data,condition,crop=None):
    if crop is None:
        xs=inputs(data,condition);d=branch(*xs);return rgb(head,xs[1]+d),d,xs
    x,y=crop;fx,fy=x//8,y//8;H,W=data['h'].shape[-2:]
    x0=max(0,fx-5);y0=max(0,fy-5);x1=min(W,fx+32+5);y1=min(H,fy+32+5)
    xs=inputs(data,condition,crop=(x0,y0,x1,y1));d=branch(*xs)
    out=rgb(head,xs[1]+d);sx=(fx-x0)*8;sy=(fy-y0)*8
    return out[...,sy:sy+256,sx:sx+256],d,xs
