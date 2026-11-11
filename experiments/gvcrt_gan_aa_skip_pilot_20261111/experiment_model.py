"""B300 receiver, additive native f_hat adapter and local native-convolution boundary."""
from common_s2 import *
import types, torch
from torch import nn
from b300_loader import construct as audited_b300, detach_history
from model_context import assert_reset
from filters import lowpass, replace_wsilus


class SkipAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        from src.layers.layers import WSiLU
        self.down=nn.Conv2d(256,32,1,bias=False)
        self.act1=WSiLU()
        self.depthwise=nn.Conv2d(32,32,3,padding=1,groups=32,bias=False)
        self.act2=WSiLU()
        self.proj=nn.Conv2d(32,512,1,bias=False)
        nn.init.zeros_(self.proj.weight)
    def forward(self,x):
        return self.proj(self.act2(self.depthwise(self.act1(self.down(x)))))


def force_reference(gen):
    # The common reference preserves every original arithmetic operation.
    for m in gen.modules():
        if m.__class__.__name__ in ['DepthConvBlock','StageBlock','SubpelConv2x']:
            m.forward=types.MethodType(lambda self,*a,**kw:self.forward_torch(*a,**kw),m)
            if hasattr(m,'proxy'):m.proxy=None
    if os.environ.get('GVC_GRAD_CHECKPOINT')=='1':
        from torch.utils.checkpoint import checkpoint
        def block_forward(self,x,quant_step=None,to_cat=None,cat_at_front=True):
            fn=lambda t:self.forward_torch(t,quant_step,to_cat,cat_at_front)
            return checkpoint(fn,x,use_reentrant=False) if torch.is_grad_enabled() and x.requires_grad else fn(x)
        for m in gen.modules():
            if m.__class__.__name__=='DepthConvBlock':m.forward=types.MethodType(block_forward,m)
    gen.forward=types.MethodType(lambda self,*a,**kw:self.forward_torch(*a,**kw),gen)


def construct(group='J0', train=True, rank=0, losses=None):
    assert group in [*GROUPS,'B300']
    # train=True is needed for lossless FP32 loading of inherited parameters.
    # It does not authorize the inherited loader's trainable boundary.
    c=audited_b300('B300',True,rank,losses=train if losses is None else losses)
    c['optimizer']=None;c['ctx']=None;c['initial_trainable']={}
    c['pm'].requires_grad_(False);gen=c['models']['generator']
    force_reference(gen);gen.eval()
    if losses is False or not train:
        c['quality']=None;c['raft']=None
    with torch.random.fork_rng(devices=[rank]):
        torch.manual_seed(TRAIN_SEED)
        if group!='B300':gen.skip_adapter=SkipAdapter().to(c['device'])
    aa=group in ['J2','J3']
    changed=[]
    if aa:
        changed=replace_wsilus(gen.stage3)
        changed+=['skip_adapter.'+n for n in replace_wsilus(gen.skip_adapter)]
    c.update(experiment_group=group,anti_alias=aa)
    named=[];sets={'Skip':[],'Native':[]}
    for name,p in c['pm'].named_parameters():
        prefix='recon_generation_net.decoder.'
        local=name.removeprefix(prefix)
        kind=None
        if group!='B300' and name.startswith(prefix):
            if local.startswith('skip_adapter.'):kind='Skip'
            elif (local.startswith('upsample.conv1.') or local.startswith('stage3.blocks.')) and '.lora.' not in local:
                parent=dict(gen.named_modules())[local.rsplit('.',1)[0]]
                if isinstance(parent,nn.Conv2d):kind='Native'
        p.requires_grad_(train and kind is not None)
        if kind:named.append((name,p));sets[kind].append(p)
    c.update(named=named,params=[p for n,p in named],parameter_sets=sets,trainable_parameter_names={n for n,p in named})
    if train and group!='B300':
        c['optimizer']=torch.optim.AdamW([dict(params=ps,lr=5e-5 if k=='Skip' else 2e-6,group_name=k) for k,ps in sets.items()],betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
    c['provenance'].update(experiment_group=group,common_FP32_reference=True,s_train=.5,s_eval=.5,trainable_names=[n for n,p in named],trainable_counts={k:sum(p.numel() for p in ps) for k,ps in sets.items()},filtered_WSiLU_names=changed)
    assert not any(p.requires_grad for n,p in c['pm'].named_parameters() if '.lora.' in n or '.tcg_' in n or '.norms.' in n)
    return c


def bridge(c,feature):
    m=c['models']['bridge'];x=torch.nn.functional.pixel_unshuffle(feature.detach().to(c['device']).float(),2)
    x=m[2](m[1](m[0](x)))
    return m[3](x*torch.sigmoid(x))


def decode(c,z,f,q,use_tcg=True,update=True):
    gen=c['models']['generator']
    if c['experiment_group']=='B300':return gen(z,q,use_tcg=use_tcg,update_buffer=update)
    out=gen.stage1(gen.ada1(gen.conv_in(z),z))
    out=gen.stage2(gen.ada2(out,z))
    out=gen._apply_tcg(gen.tcg_low,out,use_tcg,update)
    out=gen.upsample(gen.ada3(out,z))
    if c['anti_alias']:out=lowpass(out)
    skip=gen.skip_adapter(f.to(c['device']).float())
    c['last_RMS']={'main':float(out.detach().square().mean().sqrt()),'skip':float(skip.detach().square().mean().sqrt())}
    out=gen._apply_tcg(gen.tcg_mid,out+skip,use_tcg,update)
    out=gen.stage3(out)
    out=gen.stage4(gen.ada4(out,z),q)
    out=gen._apply_tcg(gen.tcg_out,out,use_tcg,update)
    return torch.nn.functional.pixel_shuffle(gen.head(gen.ada_final(out,z)),8).clamp(-1,1)


def reconstruct(c,feature,qp,retain=False,latent=False):
    gen=c['models']['generator'];gen.set_temporal_cascade_grad(retain)
    for name in ['tcg_low','tcg_mid','tcg_out']:getattr(gen,name).alignment=None
    z=bridge(c,feature);q=c['pm'].q_scale_recon[qp:qp+1].float()
    out=decode(c,z,feature,q)
    return (out,z) if latent else out


def trainable_state(c):return {n:p.detach().cpu().clone() for n,p in c['named']}


def restore(c,a,optimizer=True):
    assert a['B300_SHA256']==B300_SHA and a['experiment_group']==c['experiment_group']
    assert set(a['trainable_state'])==c['trainable_parameter_names']
    full=c['pm'].state_dict();full.update(a['trainable_state']);c['pm'].load_state_dict(full,strict=True)
    if optimizer:c['optimizer'].load_state_dict(a['optimizer'])
    assert_reset(c)
