"""Four independent new objectives; no historic optimizer state is reused."""
from common_s2 import *
import types
import torch
from model_context import build,component_hashes,assert_reset,frozen_items

def _torch_block(self,x,quant_step=None,to_cat=None,cat_at_front=True):
    return self.forward_torch(x,quant_step,to_cat,cat_at_front)

def _torch_stage(self,x,quant_step=None):return self.forward_torch(x,quant_step)

def construct(group='A',train=True,rank=0):
    assert group in GROUPS
    c=build(rank,train=train,losses=train,group='A_control',fresh_optimizer=True)
    if train:torch.use_deterministic_algorithms(True,warn_only=True)
    # Standard CUDA grid_sample input gradients use atomic accumulation. This
    # runtime fact is audited against repeat-relative-L2<1e-3, never bit equality.
    gen=c['models']['generator'];bridge=c['models']['bridge']
    c['experiment_group']=group;c['alignment_enabled']=group in 'CD'
    c['optimizer']=None
    # Training must permit input gradients through frozen layers. Evaluation
    # retains the audited legacy safe-dispatch arithmetic at unchanged weights.
    if train:
        for m in c['pm'].recon_generation_net.modules():
            if m.__class__.__name__=='DepthConvBlock':m.forward=types.MethodType(_torch_block,m)
            if m.__class__.__name__=='StageBlock':m.forward=types.MethodType(_torch_stage,m)
    for p in c['pm'].parameters():p.requires_grad_(False)
    named=[];sets={k:[] for k in ['TCG','LoRA','Bridge_mlp3']}
    for name,p in c['pm'].named_parameters():
        kind=None
        if name.startswith(tuple('recon_generation_net.decoder.'+x for x in TCG_PREFIXES)):kind='TCG'
        elif '.lora.' in name:kind='LoRA'
        elif group in 'BD' and name.startswith('recon_generation_net.mlp.3.'):kind='Bridge_mlp3'
        if kind:
            p.requires_grad_(train);named.append((name,p));sets[kind].append(p)
    assert sum(p.numel() for p in sets['TCG'])==137760
    assert sum(p.numel() for p in sets['LoRA'])==59392
    assert bool(sets['Bridge_mlp3'])==(group in 'BD')
    if train:assert all(p.dtype==torch.float32 for n,p in named)
    rates=dict(TCG=1e-5,LoRA=5e-5,Bridge_mlp3=1e-6)
    optimizer=torch.optim.AdamW([dict(params=v,lr=rates[k],group_name=k) for k,v in sets.items() if v],betas=(.9,.999),eps=1e-8,weight_decay=1e-4) if train else None
    c.update(named=named,params=[p for n,p in named],optimizer=optimizer,parameter_sets=sets,trainable_parameter_names={n for n,p in named},
        initial_trainable={n:p.detach().clone() for n,p in named})
    c['provenance'].update(experiment_group=group,s_train=.5,s_eval=.5,parent_optimizer_restored=False,new_optimizer_empty=optimizer is None or len(optimizer.state)==0,
        optimizer_restored=False,optimizer_absolute_step_at_load=0,optimizer_name_order=[n for n,p in named],optimizer=dict(type='AdamW',state_entries=0 if optimizer is not None else None,param_groups=[dict(name=k,lr=rates[k],parameter_count=sum(p.numel() for p in v)) for k,v in sets.items() if v]),
        initialization_LoRA_step=2000,initialization_TCG_step=6500,trainable_module_counts={k:sum(p.numel() for p in v) for k,v in sets.items()},
        trainable_parameters=sum(p.numel() for n,p in named) if train else 0,trainable_names=[n for n,p in named],runtime_versions=dict(Python=sys.version,Torch=torch.__version__,CUDA=torch.version.cuda,cuDNN=torch.backends.cudnn.version(),intraop_threads=torch.get_num_threads(),deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled()),Bridge_mlp3_type=bridge[3].__class__.__name__,frozen_codec_reference='Independent original codec trajectory; candidate recon never committed to codec DPB')
    return c

def detach_history(c):
    for name in ['tcg_low','tcg_mid','tcg_out']:
        m=getattr(c['models']['generator'],name)
        if m._previous is not None:m._previous=m._previous.detach()
    c['models']['generator'].set_temporal_cascade_grad(False)

def reconstruct(c,feature,qp,alignment=None,retain=False,update=True):
    gen=c['models']['generator'];gen.set_temporal_cascade_grad(retain)
    for name in ['tcg_low','tcg_mid','tcg_out']:getattr(gen,name).alignment=alignment if c['alignment_enabled'] else None
    pm=c['pm'];dtype=next(c['models']['bridge'].parameters()).dtype
    # Q-scale is immutable codec state. Incoming feature is the independent
    # decoder's detached pm.dec output, not a candidate-generated reference.
    quant=pm.q_scale_recon[qp:qp+1]
    return pm.recon_generation_net(feature.detach().to(dtype),quant.to(dtype),use_tcg=True,update_buffer=update)

def state(c):
    names=c['trainable_parameter_names']
    return {n:t.detach().cpu().clone() for n,t in c['pm'].state_dict().items() if n in names}

def restore(c,payload):
    assert payload['initial_A2000_sha256']==INIT_SHA
    assert payload['experiment_group']==c['experiment_group']
    expected=set(c['trainable_parameter_names']);assert set(payload['trainable_state'])==expected
    full=dict(c['pm'].state_dict());full.update(payload['trainable_state']);c['pm'].load_state_dict(full,strict=True)
    if c['optimizer'] is not None:
        if 'optimizer_name_groups' in payload:
            lookup={id(p):n for n,p in c['named']}
            actual={g['group_name']:[lookup[id(p)] for p in g['params']] for g in c['optimizer'].param_groups}
            assert payload['optimizer_name_groups']==actual,'OPTIMIZER_PARAMETER_NAME_ORDER_MISMATCH'
        else:assert payload.get('source_manifest_SHA256') is None,'ONLY_LEGACY_DISPOSABLE_SMOKE_CAN_LACK_PARAMETER_NAMES'
        c['optimizer'].load_state_dict(payload['optimizer'])
        assert all(int(x['step'])==payload['committed_updates'] for x in c['optimizer'].state.values())
    assert_reset(c)
