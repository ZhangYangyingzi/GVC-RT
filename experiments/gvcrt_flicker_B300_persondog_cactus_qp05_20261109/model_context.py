"""Strict full reconstruction from original I/P, pilot receiver, and B+1500 TCG."""
from common_s2 import *

def backend(reference=False,rank=0):
    env()
    if os.environ.get('MODEL_KIND')=='original':
        from original_context import original_backend
        return original_backend(rank)
    tree=SOURCE_PILOT_EXPERIMENT/('reference_tree' if reference else 'runtime_tree')
    for p in [tree,tree/'experiments/gvcrt_neural_wrapper_v2_joint',tree/'experiments/gvcrt_neural_wrapper_v4_progressive_joint',
              tree/'experiments/gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100',
              tree/'experiments/gvcrt_neural_wrapper_v6_2_objective_calibration',
              tree/'expericent_generation_input/expericent_interface_causal_controls_v9/src']:
        sys.path.insert(0,str(p))
    sys.path.append(str(HIST.parents[1]/'experiments/gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies'))
    import torch
    torch.cuda.set_device(rank)
    from src.utils.common import set_torch_env
    set_torch_env();torch.set_num_threads(2)
    def no_download(*a,**kw):raise RuntimeError('LOCAL_PRETRAINED_WEIGHTS_REQUIRED; downloads prohibited')
    torch.hub.download_url_to_file=no_download
    import core,training32,fullqp_common
    assert Path(sys.modules['src.models.video_model_gvcrt'].__file__).resolve().is_relative_to(tree)
    eco=module('s2_eval_core',tree/'experiments/gvcrt_neural_wrapper_v4_progressive_joint/eval_core.py')
    return torch,core,eco,training32,fullqp_common,tree

def build(rank=0,train=True,losses=True,lora_checkpoint=None,baseline=False,reference=False,legacy_dispatch=False,group='A_control',fresh_optimizer=False):
    if os.environ.get('MODEL_KIND')=='original':
        assert not train and not losses and lora_checkpoint is None
        from original_context import build_original
        return build_original(rank)
    assert group=='A_control'
    torch,core,eco,tr,qp,tree=backend(reference,rank)
    device=torch.device('cuda',rank)
    assert sha(BASELINE)==BASELINE_SHA and sha(PILOT)==PILOT_SHA
    for name in ['GVC-RT_I.pt','GVC-RT_P.pt']:
        h=expected_source_sha(PROJECT/'checkpoints'/name)
        assert sha(tree/'checkpoints'/name)==h
    im,pm=core.load_models(device,force_zero_thres=.12)
    load_audit={}
    for label,model,name in [('I',im,'GVC-RT_I.pt'),('P',pm,'GVC-RT_P.pt')]:
        raw=torch.load(tree/'checkpoints'/name,map_location='cpu',weights_only=True)
        raw=raw.get('student_ema',raw.get('student',raw.get('state_dict',raw)))
        actual=model.state_dict()
        unexpected=set(raw)-set(actual)
        missing=set(actual)-set(raw)
        assert not unexpected,('ORIGINAL_CODEC_UNEXPECTED_KEYS',label,sorted(unexpected))
        if label=='I':assert not missing,('ORIGINAL_I_MISSING_KEYS',sorted(missing))
        else:assert all(n.startswith(tuple('recon_generation_net.decoder.'+p for p in TCG_PREFIXES)) for n in missing),('ORIGINAL_P_CRITICAL_MISSING_KEYS',sorted(missing))
        for n,t in raw.items():assert torch.equal(actual[n].detach().cpu(),t.to(dtype=actual[n].dtype)),('ORIGINAL_CODEC_TENSOR_MISMATCH',label,n)
        load_audit[label]=dict(verified_loaded_keys=len(raw),explicit_legacy_TCG_missing_keys=sorted(missing),unexpected_keys=[])
    gen=pm.recon_generation_net.decoder
    bridge=pm.recon_generation_net.mlp
    if train:gen.float();bridge.float()
    # Promote the three pretrained TCGs BEFORE loading B+1500: no FP16 round-trip.
    for name in ['tcg_low','tcg_mid','tcg_out']:getattr(gen,name).float()
    full=torch.load(PILOT,map_location='cpu',weights_only=True)
    state=torch.load(BASELINE,map_location='cpu',weights_only=True)
    assert state['absolute_optimizer_step']==6500 and state['additional_step']==1500
    tcg=state['tcg_model_state']
    expected_tcg={n for n in gen.state_dict() if n.startswith(TCG_PREFIXES)}
    assert set(tcg)==expected_tcg,'PARTIAL_TCG_KEYS_NOT_EXACT'
    merged=dict(full['models']['generator'])
    assert set(merged)==set(gen.state_dict()),'PILOT_GENERATOR_KEYS_NOT_EXACT'
    merged.update(tcg)
    gen.load_state_dict(merged,strict=True)
    bridge.load_state_dict(full['models']['bridge'],strict=True)
    identity=torch.nn.Identity().to(device)
    identity.load_state_dict(full['models']['wrapper'],strict=True)
    quality=core.quality_models(device) if losses else None
    buffer_models=dict(wrapper=identity,bridge=bridge,generator=gen,im=im,pm=pm)
    if quality is not None:buffer_models.update(lp=quality[0],dists=quality[1])
    # Restore exact pilot persistent buffers before attaching any LoRA module.
    for key,m in buffer_models.items():
        assert key in full['buffers']
        for n,t in m.named_buffers():
            assert n in full['buffers'][key],('PILOT_BUFFER_MISSING',key,n)
            with torch.no_grad():t.copy_(full['buffers'][key][n])
    for n,t in gen.state_dict().items():assert torch.equal(t.detach().cpu(),merged[n].to(dtype=t.dtype)),('GENERATOR_LOAD_MISMATCH',n)
    for n,t in gen.state_dict().items():
        if n.startswith(TCG_PREFIXES):assert t.dtype==torch.float32 and torch.equal(t.detach().cpu(),tcg[n]),('TCG_FP32_B1500_LOAD_MISMATCH',n)
    for name in ['tcg_low','tcg_mid','tcg_out']:
        assert getattr(gen,name).fusion[-1].weight.detach().norm()>0,('TRAINED_FUSION_HEAD_MUST_NOT_BE_REZEROED',name)
    for n,t in bridge.state_dict().items():assert torch.equal(t.detach().cpu(),full['models']['bridge'][n].to(dtype=t.dtype))
    for m in [im,pm,*(() if quality is None else quality),identity]:m.eval().requires_grad_(False)
    if train:gen.train()
    gen.set_temporal_cascade_grad(False)

    from convlora import attach_generator_convlora,configure_safe_dispatch
    original_generator_keys=set(gen.state_dict())
    if baseline:
        assert lora_checkpoint is None
        if not legacy_dispatch:configure_safe_dispatch(gen)
        lora_configuration=None
    else:
        assert not legacy_dispatch
        initial_path=SOURCE/'common_initialization/untrained_common.pt'
        common_state=None if not initial_path.exists() else torch.load(initial_path,map_location='cpu',weights_only=True)['lora_state']
        lora_configuration=attach_generator_convlora(gen,SEED,expanded=group=='B_expanded',common_state=common_state)
        for name,p in gen.named_parameters():p.requires_grad_(train and '.lora.' in name)
    named=[('generator.'+n,p) for n,p in gen.named_parameters() if '.lora.' in n]
    params=[p for n,p in named]
    assert len(named)==(0 if baseline else 12 if group=='B_expanded' else 10)
    assert sum(p.numel() for p in params)==(0 if baseline else lora_configuration['parameter_count'])
    for n,p in pm.named_parameters():assert p.requires_grad==(train and not baseline and '.lora.' in n)
    assert not any('router' in n.lower() for n,m in pm.named_modules()),'ROUTER_MUST_NOT_BE_INSTANTIATED'
    assert all(p.dtype==torch.float32 for p in params)
    for name in ['tcg_low','tcg_mid','tcg_out']:
        assert not any(p.requires_grad for p in getattr(gen,name).parameters())
    raft=transform=None
    if losses:
        from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
        assert sha(RAFT)==RAFT_SHA
        raft=raft_large(weights=None,progress=False)
        raft.load_state_dict(torch.load(RAFT,map_location='cpu',weights_only=True),strict=True)
        raft=raft.to(device).eval().requires_grad_(False)
        transform=Raft_Large_Weights.C_T_SKHT_V2.transforms()
    optimizer=torch.optim.AdamW([dict(params=params,lr=5e-5,group_name='LoRA')],betas=(.9,.999),eps=1e-8,weight_decay=1e-4) if train and not baseline else None
    models=dict(wrapper=identity,bridge=bridge,generator=gen)
    c=dict(torch=torch,core=core,eco=eco,tr=tr,qp=qp,im=im,pm=pm,models=models,quality=quality,raft=raft,transform=transform,
        params=params,named=named,optimizer=optimizer,device=device,group=group,train=train,reference=reference,baseline=baseline,tree=tree,
        trainable_parameter_names={n for n,p in pm.named_parameters() if '.lora.' in n},
        initial_trainable={n:p.detach().clone() for n,p in named},ctx=(core,im,pm,models,quality,qp,params,None))
    assert set(canonical_generator_state(c))==original_generator_keys
    original_frozen_hash=component_hashes(c)
    c['provenance']=dict(baseline_cumulative_tcg_step=6500,
        baseline_checkpoint=dict(path=str(BASELINE),sha256=BASELINE_SHA),receiver_initialization=dict(path=str(PILOT),sha256=PILOT_SHA),
        original_codec_key_audit=load_audit,group=group,TCG_exact_FP32_overlay=True,TCG_fusion_preserved=True,
        full_generator_keys_verified=len(merged),Bridge_loaded_strictly=True,persistent_buffers_restored=True,
        Router_instantiated=False,Router_checkpoint_loaded=False,historical_optimizer_loaded=False,
        LoRA_configuration=lora_configuration,trainable_parameters=sum(p.numel() for p in params),
        optimizer=None if optimizer is None else dict(type='AdamW',lr=5e-5,betas=[.9,.999],eps=1e-8,weight_decay=1e-4,state_entries=len(optimizer.state)),
        precision=dict(codec='FP16',Bridge_and_nonTCG_generator='FP32' if train else 'FP16',TCG='FP32 parameters/internal operations; incoming output dtype',LoRA='FP32 parameters/internal operations; incoming output dtype',RGB_scoring='matched_rgb_fp32_v2'),
        components=original_frozen_hash,dispatch='legacy' if legacy_dispatch else 'LoRA-safe PyTorch stage3/head dispatch shared with baseline',
        backend=dict(customized_cuda=bool(__import__('src.layers.cuda_inference',fromlist=['CUSTOMIZED_CUDA_INFERENCE']).CUSTOMIZED_CUDA_INFERENCE),
            deterministic=torch.are_deterministic_algorithms_enabled(),matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32))
    if not baseline:
        assert sha(INIT)==INIT_SHA
        parent=torch.load(INIT,map_location='cpu',weights_only=True)
        load_lora_state(c,parent)
        names=[n for n,p in named];expected=['generator.'+n for n in parent['lora_state']]
        assert names==expected and len(names)==10 and sum(p.numel() for p in params)==59392
        assert all(torch.equal(t.detach().cpu(),parent['lora_state'][n]) for n,t in lora_state(c).items())
        if train and not fresh_optimizer:
            old_group=parent['optimizer']['param_groups'];assert len(old_group)==1 and old_group[0]['lr']==5e-5
            assert old_group[0]['betas']==(.9,.999) and old_group[0]['eps']==1e-8 and old_group[0]['weight_decay']==1e-4
            assert old_group[0]['params']==list(range(10))
            for i,(name,p) in enumerate(named):
                assert tuple(parent['optimizer']['state'][i]['exp_avg'].shape)==tuple(p.shape)
                assert tuple(parent['optimizer']['state'][i]['exp_avg_sq'].shape)==tuple(p.shape)
                assert int(parent['optimizer']['state'][i]['step'])==2000
            optimizer.load_state_dict(parent['optimizer'])
            assert all(int(s['step'])==2000 for s in optimizer.state.values())
            assert tensor_hash([(str(i)+k,t) for i,s in optimizer.state_dict()['state'].items() for k,t in s.items() if torch.is_tensor(t)])==tensor_hash([(str(i)+k,t) for i,s in parent['optimizer']['state'].items() for k,t in s.items() if torch.is_tensor(t)])
        from convlora import set_inference_gain
        set_inference_gain(gen,.5)
        c['initial_trainable']={n:p.detach().clone() for n,p in named}
        c['provenance'].update(parent_A2000_SHA256=INIT_SHA,parent_LoRA_step=2000,optimizer_restored=bool(train),optimizer_name_order=names,s_train=1.0,s_eval=.5,
            optimizer_absolute_step_at_load=2000 if train else None)
    if lora_checkpoint is not None:load_lora_state(c,lora_checkpoint)
    return c


def lora_state(c):return {n:t for n,t in c['models']['generator'].state_dict().items() if '.lora.' in n}

def load_lora_state(c,state):
    assert state['group']==c['group'],'CHECKPOINT_GROUP_MISMATCH'
    expected=set(lora_state(c));assert expected and set(state['lora_state'])==expected
    assert state['baseline_sha256']==BASELINE_SHA and state['pilot_sha256']==PILOT_SHA
    full=dict(c['models']['generator'].state_dict());full.update(state['lora_state'])
    c['models']['generator'].load_state_dict(full,strict=True)
    assert all(p.dtype==c['torch'].float32 for p in c['params'])

def trainable_hash(c):return tensor_hash(c['named'])

def canonical_generator_state(c):
    result={}
    for name,t in c['models']['generator'].state_dict().items():
        if '.lora.' in name:continue
        key=name.replace('.dc.2.base.','.dc.2.').replace('upsample.conv1.base.','upsample.conv1.')
        if key.startswith('head.base.'):key='head.'+key[len('head.base.'):]
        assert key not in result;result[key]=t
    return result

def component_hashes(c):
    gen=canonical_generator_state(c)
    items={'I_model':c['im'].state_dict().items(),
        'P_codec':[(n,t) for n,t in c['pm'].state_dict().items() if not n.startswith('recon_generation_net.')],
        'Bridge':c['models']['bridge'].state_dict().items(),
        'generator_original_nonTCG':[(n,t) for n,t in gen.items() if not n.startswith(TCG_PREFIXES)],
        'TCG':[(n,t) for n,t in gen.items() if n.startswith(TCG_PREFIXES)]}
    if c['quality'] is not None:items.update(LPIPS=c['quality'][0].state_dict().items(),DISTS=c['quality'][1].state_dict().items())
    if c['raft'] is not None:items['RAFT']=c['raft'].state_dict().items()
    return {k:dict(sha256=tensor_hash(list(v))) for k,v in items.items()}

def frozen_items(c):
    items=[('I',c['im']),('P',c['pm'])]
    if c['quality'] is not None:items += [('LPIPS',c['quality'][0]),('DISTS',c['quality'][1])]
    if c['raft'] is not None:items += [('RAFT',c['raft'])]
    for prefix,model in items:
        for name,t in model.state_dict().items():
            if prefix=='P' and name in c['trainable_parameter_names']:continue
            yield prefix+'.'+name,t

def history_snapshot(c):
    pm=c['pm'];gen=c['models']['generator']
    return dict(TCG={n:None if getattr(gen,n)._previous is None else tensor_sha(getattr(gen,n)._previous) for n in ['tcg_low','tcg_mid','tcg_out']},
        DPB=[dict(feature=None if x.feature is None else tensor_sha(x.feature),frame=None if x.frame is None else tensor_sha(x.frame),poc=x.poc) for x in pm.dpb],curr_poc=pm.curr_poc)

def assert_reset(c):
    c['pm'].clear_dpb();c['pm'].set_curr_poc(0)
    assert not c['pm'].dpb
    if c.get('original'):assert not any('tcg' in n.lower() for n,p in c['pm'].named_parameters())
    else:assert all(getattr(c['models']['generator'],n)._previous is None for n in ['tcg_low','tcg_mid','tcg_out'])
