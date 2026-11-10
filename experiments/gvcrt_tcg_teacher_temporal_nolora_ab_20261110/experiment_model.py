"""B300 restoration followed by physical unwrapping; only Bridge+TCG train."""
from common_s2 import *
import torch
from b300_loader import construct as restore_b300,detach_history
from model_context import assert_reset

def category(n):
    if n.startswith('recon_generation_net.mlp.'):return 'Bridge'
    if n.startswith(tuple('recon_generation_net.decoder.'+p for p in TCG_PREFIXES)):return 'TCG'

def strip_lora(c):
    gen=c['models']['generator'];removed=[]
    for n,m in list(gen.named_modules()):
        if m.__class__.__name__!='ParallelConvLoRA':continue
        parent,_,leaf=n.rpartition('.');container=gen.get_submodule(parent) if parent else gen
        before=tensor_hash(m.base.state_dict().items());base=m.base;groups=base.groups
        container._modules[leaf]=base
        assert tensor_hash(container._modules[leaf].state_dict().items())==before
        removed.append(dict(path=n,base_SHA256=before,groups=groups,shape=list(base.weight.shape)))
    assert len(removed)==5
    assert not any('lora' in n.lower() or 'router' in n.lower() for n,m in gen.named_modules())
    assert not any('lora' in n.lower() for n,p in c['pm'].named_parameters())
    assert all(r['groups']==320 for r in removed if '.dc.2' in r['path'])
    c['provenance'].update(LoRA_physically_removed=True,LoRA_merged=False,removed_wrappers=removed,Bridge_TCG_inherited_from_LoRA_trained_B300=True,safe_dispatch_preserved=True)
    return removed

def construct(group=GROUPS[0],train=True,rank=0,reference=None):
    assert group in [*GROUPS,'B300_noLoRA_step0','B300_original_reference']
    c=restore_b300('B300',train,rank)
    if reference is not None:
        import copy
        # Entropy coder C++ objects are not pickleable; clone the complete
        # torch reconstruction module and keep independently restored codec.
        c['pm'].recon_generation_net=copy.deepcopy(reference['pm'].recon_generation_net)
        c['models']['bridge']=c['pm'].recon_generation_net.mlp;c['models']['generator']=c['pm'].recon_generation_net.decoder
        c['provenance']['cloned_full_reconstruction_from_B300_reference']=True
    c['optimizer']=None;c['ctx']=None
    c['experiment_group']=group
    if group!='B300_original_reference':
        strip_lora(c)
        common=ROOT/'checkpoints/B300_noLoRA_step0.pt'
        if common.exists():
            saved=torch.load(common,map_location='cpu',weights_only=True);assert saved['B300_SHA256']==B300_SHA
            c['pm'].recon_generation_net.load_state_dict(saved['recon_state'],strict=True)
    for p in c['pm'].parameters():p.requires_grad_(False)
    named=[];sets={k:[] for k in ['Bridge','TCG']}
    if group!='B300_original_reference':
        for n,p in c['pm'].named_parameters():
            k=category(n)
            if k:p.requires_grad_(train);named.append((n,p));sets[k].append(p)
        assert {k:sum(p.numel() for p in ps) for k,ps in sets.items()}==dict(Bridge=801124,TCG=137760)
    rates=dict(Bridge=2e-6,TCG=1e-5)
    optimizer=torch.optim.AdamW([dict(params=ps,lr=rates[k],group_name=k) for k,ps in sets.items()],betas=(.9,.999),eps=1e-8,weight_decay=1e-4) if train and named else None
    if train:
        from quality import training_models
        if getattr(c['quality'][0],'pnet_type',None)!='vgg':c['quality']=training_models(c['device'],c['quality'][1])
    c.update(named=named,params=[p for n,p in named],parameter_sets=sets,optimizer=optimizer,trainable_parameter_names={n for n,p in named},alignment_enabled=False)
    c['provenance'].update(experiment_group=group,trainable_parameter_counts={k:sum(p.numel() for p in ps) for k,ps in sets.items()},fresh_AdamW=True,optimizer_restored=False,Teacher_deployed=False)
    if group!='B300_original_reference':c['provenance'].update(active_LoRA_branches=0,LoRA_configuration=None,parent_B300_LoRA_gain=.5,s_train=None,s_eval=None)
    return c

def bridge(c,feature):
    m=c['models']['bridge'];dtype=next(m.parameters()).dtype
    x=torch.nn.functional.pixel_unshuffle(feature.detach().to(c['device'],dtype),2)
    x=m[2](m[1](m[0](x)));return m[3](x*torch.sigmoid(x))

def reconstruct(c,feature,qp,retain=False,latent=False):
    gen=c['models']['generator'];gen.set_temporal_cascade_grad(retain)
    for n in ['tcg_low','tcg_mid','tcg_out']:getattr(gen,n).alignment=None
    z=bridge(c,feature);q=c['pm'].q_scale_recon[qp:qp+1].to(z.dtype)
    out=gen(z,q,use_tcg=True,update_buffer=True)
    return (out,z) if latent else out

def trainable_state(c):return {n:p.detach().cpu().clone() for n,p in c['named']}

def restore(c,saved,optimizer=True):
    assert saved['B300_SHA256']==B300_SHA and saved['experiment_group']==c['experiment_group']
    assert set(saved['trainable_state'])==c['trainable_parameter_names']
    full=c['pm'].state_dict();full.update(saved['trainable_state']);c['pm'].load_state_dict(full,strict=True)
    if optimizer and c['optimizer'] is not None:
        names={id(p):n for n,p in c['named']};assert saved['optimizer_name_groups']=={g['group_name']:[names[id(p)] for p in g['params']] for g in c['optimizer'].param_groups}
        c['optimizer'].load_state_dict(saved['optimizer'])
    assert_reset(c)
