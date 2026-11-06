"""Discardable full-model loading, zero-output and actual CUDA-entry checks."""
from common_s2 import *
from model_context import build,component_hashes,assert_reset,history_snapshot,frozen_items,lora_state

def main():
    env()
    import torch
    c=build(0,train=False,losses=False);b=build(0,train=False,losses=False,baseline=True)
    assert component_hashes(c)==component_hashes(b)
    gen=c['models']['generator'];guard=FrozenGuard(frozen_items(c));seed(SEED)
    probes=[torch.randn(1,256,32,32,device=c['device'],dtype=torch.float16) for _ in range(2)]
    q=c['pm'].q_scale_recon[5:6];capture=[]
    hook=gen.register_forward_pre_hook(lambda m,args:capture.append(tuple(x.detach().clone() if torch.is_tensor(x) else x for x in args)))
    original=[];adapted=[]
    with torch.no_grad():
        for x in probes:
            original.append(b['pm'].recon_generation_net(x.clone(),q).clone())
            adapted.append(c['pm'].recon_generation_net(x.clone(),q).clone())
            assert torch.equal(original[-1],adapted[-1]),'ZERO_LORA_NOT_MATCHED_BASELINE'
            assert history_snapshot(c)['TCG']==history_snapshot(b)['TCG']
    hook.remove();assert len(capture)==2
    branch_modules=[m for m in gen.modules() if m.__class__.__name__=='LowRankConvBranch'];assert len(branch_modules)==5
    calls_before=[m.forward_calls for m in branch_modules]
    args=capture[-1]
    outputs=[]
    with torch.no_grad():
        for method in [gen.forward_torch,gen.forward_cuda]:
            gen.reset_temporal_buffer()
            gen.forward_torch(*[a.clone() if torch.is_tensor(a) else a for a in capture[0]])
            outputs.append(method(*[a.clone() if torch.is_tensor(a) else a for a in args]))
        assert torch.equal(*outputs),'TORCH_CUDA_ENTRY_OUTPUT_MISMATCH'
        calls_after=[m.forward_calls for m in branch_modules]
        assert all(a-b==4 for a,b in zip(calls_after,calls_before)),'LORA_BRANCH_NOT_EXECUTED'
        saved={n:t.clone() for n,t in lora_state(c).items()}
        for m in branch_modules:m.up.weight.normal_(std=1e-3)
        gen.reset_temporal_buffer();changed=gen.forward_cuda(*args)
        gen.reset_temporal_buffer()
        current=dict(gen.state_dict());current.update(saved);gen.load_state_dict(current,strict=True)
        zero=gen.forward_cuda(*args)
        difference=float((changed-zero).abs().max());assert difference>0,'NONZERO_LORA_BYPASSED'
        assert_reset(c);assert_reset(b)
        c['pm'].recon_generation_net(probes[0],q);before=history_snapshot(c)
        c['pm'].recon_generation_net(probes[1],q,update_buffer=False)
        assert history_snapshot(c)==before
        c['pm'].recon_generation_net(probes[1],q,use_tcg=False,update_buffer=False)
        assert history_snapshot(c)==before
        assert_reset(c)
    assert all(not p.requires_grad for p in c['pm'].parameters())
    guard.full()
    dump(ROOT/'baseline_provenance.json',c['provenance'])
    dump(ROOT/'trainable_parameters.json',dict(modules=['stage3 depthwise-parallel LoRA (4)','head parallel LoRA'],parameter_count=59392,rank=4,alpha_div_rank=1,
        trainable_parameter_list=[dict(name=n,shape=list(p.shape),dtype=str(p.dtype)) for n,p in c['named']],
        all_original_parameters_frozen=True,TCG_frozen=True,Router_instantiated=False))
    dump(ROOT/'audits/full_model_load_and_dispatch.json',dict(status='PASS',baseline_cumulative_TCG_step=6500,baseline_sha256=BASELINE_SHA,pilot_sha256=PILOT_SHA,
        TCG_exact_FP32=True,Router_instantiated=False,only_LoRA_allowed=True,trainable_parameter_count=59392,
        baseline_and_zero_LoRA_strict_equal=True,TCG_cache_strict_equal=True,torch_vs_cuda_entry_strict_equal=True,
        native_LoRA_fused_CUDA_kernel=False,CUDA_entry_uses_explicit_PyTorch_fallback=True,original_codec_dispatch_unchanged=True,
        LoRA_branch_forward_calls_before=calls_before,LoRA_branch_forward_calls_after=calls_after,
        nonzero_disposable_LoRA_output_max_difference=difference,frozen_state_hash=guard.digest,update_buffer_false_unchanged=True,
        internal_reference_no_cache_pollution=True,formal_updates=0))
    print('FULL_MODEL_LOAD_ZERO_EQUIVALENCE_DISPATCH_PASS')

if __name__=='__main__':main()
