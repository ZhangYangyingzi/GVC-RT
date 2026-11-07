"""Fresh common initialization and actual A/B coverage, precision and cache checks."""
from common_s2 import *
from model_context import build,lora_state,component_hashes,frozen_items,history_snapshot,assert_reset

def main():
    env()
    import torch
    a=build(0,train=False,losses=False,group='A_control')
    initial={n:t.detach().cpu().clone() for n,t in lora_state(a).items()}
    assert len(initial)==10 and sum(t.numel() for t in initial.values())==59392
    assert all(not t.any() for n,t in initial.items() if '.up.' in n)
    path=ROOT/'common_initialization/untrained_common.pt'
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=True)
        assert old['untrained'] and all(torch.equal(t,old['lora_state'][n]) for n,t in initial.items())
    else:atomic_torch(path,dict(untrained=True,seed=SEED,baseline_sha256=BASELINE_SHA,pilot_sha256=PILOT_SHA,lora_state=initial))
    b=build(0,train=False,losses=False,group='B_expanded');base=build(0,train=False,losses=False,baseline=True)
    assert component_hashes(a)==component_hashes(b)==component_hashes(base)
    bs=lora_state(b)
    assert all(torch.equal(t,bs[n].cpu()) for n,t in initial.items())
    config=b['provenance']['LoRA_configuration'];extra=next(t for t in config['targets'] if t['path']=='upsample.conv1')
    assert sum(p.numel() for p in b['params'])==59392+4*extra['in_channels']+36*extra['out_channels']
    seed(SEED);xs=[torch.randn(1,256,32,32,device=a['device'],dtype=torch.float16) for _ in range(3)]
    q=a['pm'].q_scale_recon[5:6];shapes={};hooks=[];captured=[]
    gen=b['models']['generator']
    for name in ['tcg_low','tcg_mid','tcg_out']:
        hooks.append(getattr(gen,name).register_forward_pre_hook(lambda m,args,n=name:shapes.update({n:list(args[0].shape)})))
    hooks.append(gen.upsample.conv1.register_forward_hook(lambda m,args,out:shapes.update(upsample_input=list(args[0].shape),upsample_conv_output=list(out.shape))))
    hooks.append(gen.register_forward_pre_hook(lambda m,args:captured.append(tuple(x.detach().clone() if torch.is_tensor(x) else x for x in args))))
    guards=[FrozenGuard(frozen_items(c)) for c in [a,b,base]]
    with torch.no_grad():
        for x in xs:
            values=[c['pm'].recon_generation_net(x.clone(),q) for c in [base,a,b]]
            assert torch.equal(values[0],values[1]) and torch.equal(values[0],values[2]),'STEP0_OUTPUT_MISMATCH'
            assert history_snapshot(base)['TCG']==history_snapshot(a)['TCG']==history_snapshot(b)['TCG']
        for h in hooks:h.remove()
        dispatcher=[]
        for c in [a,b]:
            g=c['models']['generator'];out=[]
            for fn in [g.forward_torch,g.forward_cuda]:
                g.reset_temporal_buffer();g.forward_torch(*captured[0]);out.append(fn(*captured[1]))
            assert torch.equal(*out)
            branches=[m for m in g.modules() if m.__class__.__name__=='LowRankConvBranch']
            assert len(branches)==(5 if c is a else 6) and all(m.forward_calls>0 for m in branches)
            assert_reset(c);c['pm'].recon_generation_net(xs[0],q);before=history_snapshot(c)
            c['pm'].recon_generation_net(xs[1],q,update_buffer=False);assert history_snapshot(c)==before
            c['pm'].recon_generation_net(xs[1],q,use_tcg=False,update_buffer=False);assert history_snapshot(c)==before
            dispatcher.append(dict(group=c['group'],branches=len(branches),CUDA_entry_matches_PyTorch=True,actual_forward_calls=[m.forward_calls for m in branches]))
        g=b['models']['generator'];up=g.upsample.conv1.lora.up.weight
        g.reset_temporal_buffer();zero=g.forward_cuda(*captured[1]);up.normal_(std=1e-3)
        g.reset_temporal_buffer();changed=g.forward_cuda(*captured[1]);difference=float((changed-zero).abs().max())
        assert difference>0,'UPSAMPLE_LORA_BYPASSED';up.zero_()
        for c,guard in zip([a,b,base],guards):assert_reset(c);guard.full()
    dump(ROOT/'baseline_provenance.json',dict(baseline=base['provenance'],A_control=a['provenance'],B_expanded=b['provenance'],
        common_untrained_state_sha256=sha(path),shared_initial_tensors_equal=True,previous_trained_LoRA_checkpoint_loaded=False))
    dump(ROOT/'audits/initialization_and_dispatch.json',dict(status='PASS',A_parameters=59392,B_parameters=config['parameter_count'],
        extra_target=extra,shared_initialization_tensors_equal=True,common_state_sha256=sha(path),zero_baseline_A_B_outputs_strict_equal=True,
        TCG_cache_strict_equal=True,actual_TCG_feature_shapes=shapes,representative_codec_feature_shape=[1,256,32,32],
        three_distinct_TCG_resolutions_claimed=False,dispatch=dispatcher,nonzero_upsample_output_difference=difference,
        original_components_equal_and_unchanged=True,update_buffer_false_unchanged=True,formal_updates=0))
    print('A_B_COMMON_INITIALIZATION_AND_DISPATCH_PASS',config['parameter_count'],extra)

if __name__=='__main__':main()
