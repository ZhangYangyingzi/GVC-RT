"""Restore the audited original->pilot->B1500->A2000 chain, then B300 LAST."""
from common_s2 import *
import types,torch
from model_context import build,assert_reset

def _block(self,x,quant_step=None,to_cat=None,cat_at_front=True):
    from torch.utils.checkpoint import checkpoint
    fn=lambda x:self.forward_torch(x,quant_step,to_cat,cat_at_front)
    return checkpoint(fn,x,use_reentrant=False) if torch.is_grad_enabled() else fn(x)

def _stage(self,x,quant_step=None):return self.forward_torch(x,quant_step)

def category(n):
    if n.startswith('recon_generation_net.mlp.'):return 'Bridge'
    if n.startswith('recon_generation_net.decoder.'):
        if '.lora.' in n:return 'LoRA'
        if n.startswith(tuple('recon_generation_net.decoder.'+p for p in TCG_PREFIXES)):return 'TCG'
        return 'Backbone'

def construct(group='E',train=True,rank=0,losses=None):
    assert group in ['E','F','B300','A2000']
    c=build(rank,train=train,losses=train if losses is None else losses,group='A_control',fresh_optimizer=True)
    c['optimizer']=None;c['experiment_group']=group;c['alignment_enabled']=False
    if group!='A2000':
        assert sha(B300)==B300_SHA
        b=torch.load(B300,map_location='cpu',weights_only=True)
        assert b['experiment_group']=='B' and b['committed_updates']==300 and b['initial_A2000_sha256']==INIT_SHA
        required={n for n,p in c['pm'].named_parameters() if category(n) in ['TCG','LoRA'] or n.startswith('recon_generation_net.mlp.3.')}
        assert set(b['trainable_state'])==required
        full=dict(c['pm'].state_dict());full.update(b['trainable_state']);c['pm'].load_state_dict(full,strict=True)
        assert all(torch.equal(c['pm'].state_dict()[n].cpu(),t.to(c['pm'].state_dict()[n].dtype)) for n,t in b['trainable_state'].items())
        c['provenance'].update(B300_path=str(B300),B300_SHA256=B300_SHA,B300_overlay_last=True,B300_keys=sorted(required),Bridge_mlp3_overlaid=True)
    if train:
        for m in c['pm'].recon_generation_net.modules():
            if m.__class__.__name__=='DepthConvBlock':m.forward=types.MethodType(_block,m);m.proxy=None
            if m.__class__.__name__=='StageBlock':m.forward=types.MethodType(_stage,m)
        torch.use_deterministic_algorithms(True,warn_only=True)
    for p in c['pm'].parameters():p.requires_grad_(False)
    named=[];sets={k:[] for k in ['Bridge','Backbone','TCG','LoRA']};seen=set()
    for n,p in c['pm'].named_parameters():
        k=category(n)
        if k is None:continue
        assert id(p) not in seen;seen.add(id(p));p.requires_grad_(train);named.append((n,p));sets[k].append(p)
    counts={k:sum(p.numel() for p in ps) for k,ps in sets.items()}
    assert counts==dict(Bridge=801124,Backbone=33776640,TCG=137760,LoRA=59392),counts
    if train:assert all(p.dtype==torch.float32 for n,p in named)
    rates=dict(Bridge=2e-6,Backbone=2e-6,TCG=1e-5,LoRA=5e-5)
    optimizer=torch.optim.AdamW([dict(params=ps,lr=rates[k],group_name=k) for k,ps in sets.items()],betas=(.9,.999),eps=1e-8,weight_decay=1e-4) if train else None
    c.update(named=named,params=[p for n,p in named],parameter_sets=sets,optimizer=optimizer,trainable_parameter_names={n for n,p in named})
    c['provenance'].update(parameter_counts=counts,trainable_names=[n for n,p in named],fresh_AdamW=True,optimizer_restored=False,s_train=.5,s_eval=.5,source_commit=SOURCE_COMMIT)
    return c

def detach_history(c):
    for n in ['tcg_low','tcg_mid','tcg_out']:
        m=getattr(c['models']['generator'],n)
        if m._previous is not None:m._previous=m._previous.detach()
    c['models']['generator'].set_temporal_cascade_grad(False)

def reconstruct(c,feature,qp,alignment=None,retain=False,update=True):
    gen=c['models']['generator'];gen.set_temporal_cascade_grad(retain)
    for n in ['tcg_low','tcg_mid','tcg_out']:getattr(gen,n).alignment=None
    pm=c['pm'];dtype=next(c['models']['bridge'].parameters()).dtype
    return pm.recon_generation_net(feature.detach().to(dtype),pm.q_scale_recon[qp:qp+1].to(dtype),use_tcg=True,update_buffer=update)

def state(c):return {n:t.detach().cpu().clone() for n,t in c['pm'].state_dict().items() if n in c['trainable_parameter_names']}

def restore(c,saved,optimizer=True):
    assert saved['B300_SHA256']==B300_SHA and saved['experiment_group']==c['experiment_group']
    assert set(saved['trainable_state'])==c['trainable_parameter_names']
    full=dict(c['pm'].state_dict());full.update(saved['trainable_state']);c['pm'].load_state_dict(full,strict=True)
    if optimizer and c['optimizer'] is not None:
        names={id(p):n for n,p in c['named']}
        assert saved['optimizer_name_groups']=={g['group_name']:[names[id(p)] for p in g['params']] for g in c['optimizer'].param_groups}
        c['optimizer'].load_state_dict(saved['optimizer'])
    assert_reset(c)
