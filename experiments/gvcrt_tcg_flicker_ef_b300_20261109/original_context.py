"""Pinned b2a0895 original codec in a separate worker namespace, no adapted receiver."""
from common_s2 import *
import types
_RT={}

def runtime(rank):
    if rank not in _RT:
        import torch
        torch.cuda.set_device(rank)
        import original_runtime
        rt=original_runtime.initialize('cuda:'+str(rank))
        protocol=load(ORIGINAL_EVAL/'protocol.json')
        assert protocol['original_commit']==ORIGINAL_COMMIT
        for name,h in protocol['original_source_sha256'].items():assert sha(ORIGINAL/name)==h
        for name,h in protocol['checkpoint_sha256'].items():assert sha(PROJECT/'checkpoints'/name)==h
        core=types.SimpleNamespace(unit=rt['unit'],quality_models=rt['quality_models'],module_hash=original_runtime.model_sha)
        eco=types.SimpleNamespace(frame_metrics=rt['frame_metrics']);rt.update(core=core,eco=eco)
        _RT[rank]=rt
    return _RT[rank]

def original_backend(rank):
    rt=runtime(rank);return rt['torch'],rt['core'],rt['eco'],None,None,ORIGINAL

def build_original(rank):
    from original_runtime import load_models,verify_codec_imports
    rt=runtime(rank);im,pm=load_models(rt);torch=rt['torch']
    assert not any('tcg' in n.lower() or 'lora' in n.lower() or 'router' in n.lower() for n,p in pm.named_parameters())
    return dict(torch=torch,im=im,pm=pm,core=rt['core'],eco=rt['eco'],models=dict(generator=pm.recon_generation_net.decoder,bridge=pm.recon_generation_net.mlp),
        quality=None,raft=None,params=[],named=[],optimizer=None,device=torch.device('cuda',rank),original=True,train=False,baseline=False,
        trainable_parameter_names=set(),provenance=dict(model='Original GVC-RT',original_commit=ORIGINAL_COMMIT,original_codec_only=True,
            pilot_loaded=False,TCG_loaded=False,LoRA_loaded=False,Router_instantiated=False,source_imports=verify_codec_imports(),codec_dtype='FP16'))
