"""Strict full reconstruction from original I/P, pilot receiver, and B+1500 TCG."""
from common_s2 import *

def backend(reference=False,rank=0):
    env()
    tree=ROOT/('reference_tree' if reference else 'runtime_tree')
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

def build(rank=0,train=True,mode='enabled',reference=False,losses=True,router_state=None):
    torch,core,eco,tr,qp,tree=backend(reference,rank)
    device=torch.device('cuda',rank)
    assert sha(BASELINE)==BASELINE_SHA and sha(PILOT)==PILOT_SHA
    for name,h in [('GVC-RT_I.pt','c8a840415cc028307131c66f68076b7e9f6fcc4db7fed423c62e8b202f83a22d'),('GVC-RT_P.pt','7a89bd342a1484b3da7dfc3f03834ff81e0e4180e02f8a286dec0d1025c99f4e')]:
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
    # The pilot's exact persistent buffers are restored before Router attachment.
    for key,m in buffer_models.items():
        assert key in full['buffers']
        for n,t in m.named_buffers():
            assert n in full['buffers'][key],('PILOT_BUFFER_MISSING',key,n)
            with torch.no_grad():t.copy_(full['buffers'][key][n])
    for n,t in gen.state_dict().items():assert torch.equal(t.detach().cpu(),merged[n].to(dtype=t.dtype)),('GENERATOR_LOAD_MISMATCH',n)
    for n,t in bridge.state_dict().items():assert torch.equal(t.detach().cpu(),full['models']['bridge'][n].to(dtype=t.dtype))
    for m in [im,pm,*(() if quality is None else quality),identity]:m.eval().requires_grad_(False)
    if train:gen.train()
    gen.set_temporal_cascade_grad(False)
    router=None
    if not reference:
        from src.models.spatial_tcg_router import attach_router
        router=attach_router(pm,mode)
        if router_state is not None:router.load_state_dict(router_state,strict=True)
        router.train(train).requires_grad_(train)
    raft=transform=None
    if losses:
        from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
        assert sha(RAFT)==RAFT_SHA
        raft=raft_large(weights=None,progress=False)
        raft.load_state_dict(torch.load(RAFT,map_location='cpu',weights_only=True),strict=True)
        raft=raft.to(device).eval().requires_grad_(False)
        transform=Raft_Large_Weights.C_T_SKHT_V2.transforms()
    params=[] if router is None else list(router.parameters())
    named=[] if router is None else [('router.'+n,p) for n,p in router.named_parameters()]
    if router is not None:
        assert sum(p.numel() for p in params)==33889
        assert all(p.dtype==torch.float32 and p.requires_grad==train for p in params)
    optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=1e-4) if train and router is not None else None
    models=dict(wrapper=identity,bridge=bridge,generator=gen)
    c=dict(torch=torch,core=core,eco=eco,tr=tr,qp=qp,im=im,pm=pm,models=models,quality=quality,raft=raft,transform=transform,
           router=router,params=params,named=named,optimizer=optimizer,device=device,train=train,reference=reference,tree=tree,
           ctx=(core,im,pm,models,quality,qp,params,None))
    for n,p in pm.named_parameters():assert p.requires_grad==(train and n.startswith('recon_generation_net.router.'))
    c['provenance']=dict(baseline_cumulative_tcg_step=6500,baseline_checkpoint=dict(filename=BASELINE.name,path=str(BASELINE),sha256=BASELINE_SHA),
        receiver_initialization=dict(path=str(PILOT),sha256=PILOT_SHA,keys_loaded_strictly=['wrapper','bridge','generator']),
        codec_files=[dict(filename=n,path=str(PROJECT/'checkpoints'/n),sha256=sha(PROJECT/'checkpoints'/n)) for n in ['GVC-RT_I.pt','GVC-RT_P.pt']],
        TCG_partial_keys_verified=sorted(tcg),TCG_partial_parameter_count=sum(t.numel() for t in tcg.values()),
        generator_full_keys_verified=len(merged),original_codec_key_audit=load_audit,baseline_optimizer_loaded=False,new_router_optimizer_state_entries=0 if optimizer is None else len(optimizer.state),
        precision=dict(codec='FP16',bridge_generator='FP32' if train else 'FP16',router='FP32',quality='FP32',RAFT='FP32'),
        components=component_hashes(c),backend=dict(customized_cuda=bool(__import__('src.layers.cuda_inference',fromlist=['CUSTOMIZED_CUDA_INFERENCE']).CUSTOMIZED_CUDA_INFERENCE),
                deterministic=torch.are_deterministic_algorithms_enabled(),matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32),
        model_source_root=str(tree),router_new_initialization=None if router is None else dict(seed=SEED,conv_input_channels=router.channels*3,decoded_feature_channels=router.channels,
             trainable_parameters=33889,last_weight_zero=True,last_bias=math.log(.01/.99),M_mean_at_zero_weight=.01,mode=mode))
    return c

def component_hashes(c):
    items={
      'I_model':c['im'].state_dict().items(),
      'P_codec':[(n,t) for n,t in c['pm'].state_dict().items() if not n.startswith('recon_generation_net.')],
      'Bridge':c['models']['bridge'].state_dict().items(),
      'generator_non_TCG':[(n,t) for n,t in c['models']['generator'].state_dict().items() if not n.startswith(TCG_PREFIXES)],
      'TCG':[(n,t) for n,t in c['models']['generator'].state_dict().items() if n.startswith(TCG_PREFIXES)]}
    if c['quality'] is not None:items.update(LPIPS=c['quality'][0].state_dict().items(),DISTS=c['quality'][1].state_dict().items())
    if c['raft'] is not None:items['RAFT']=c['raft'].state_dict().items()
    return {k:dict(sha256=tensor_hash(list(v))) for k,v in items.items()}

def frozen_items(c):
    for prefix,model in [('I',c['im']),('P',c['pm']),*([('LPIPS',c['quality'][0]),('DISTS',c['quality'][1])] if c['quality'] is not None else []),*([('RAFT',c['raft'])] if c['raft'] is not None else [])]:
        for name,t in model.state_dict().items():
            if prefix=='P' and name.startswith('recon_generation_net.router.'):continue
            yield prefix+'.'+name,t

def history_snapshot(c):
    pm=c['pm'];gen=c['models']['generator'];r=c['router']
    return dict(TCG={n:None if getattr(gen,n)._previous is None else tensor_sha(getattr(gen,n)._previous) for n in ['tcg_low','tcg_mid','tcg_out']},
                router_previous=None if r is None or r._previous is None else tensor_sha(r._previous),
                DPB=[dict(feature=None if x.feature is None else tensor_sha(x.feature),frame=None if x.frame is None else tensor_sha(x.frame),poc=x.poc) for x in pm.dpb],curr_poc=pm.curr_poc)

def assert_reset(c):
    c['pm'].clear_dpb();c['pm'].set_curr_poc(0)
    assert not c['pm'].dpb
    assert all(getattr(c['models']['generator'],n)._previous is None for n in ['tcg_low','tcg_mid','tcg_out'])
    assert c['router'] is None or c['router']._previous is None
