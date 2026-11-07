"""Real inherited-loss VJPs, frozen-state checks and disposable LoRA updates."""
from common_s2 import *
from model_context import build,assert_reset,frozen_items,lora_state
from pipeline import Frames,raw,Pipeline,optimized_loss

def norm(gs):return math.sqrt(sum(float(g.double().square().sum()) for g in gs))

def component_function(original):
    import inspect
    s=inspect.getsource(original)
    a='("LPIPS","weighted_reconstruction","weighted_DISTS","rate_proxy")'
    assert s.count(a)==1;s=s.replace(a,'("LPIPS","weighted_reconstruction","weighted_DISTS","rate_proxy","weighted_motion")')
    b='("rate_proxy",(beta*rb+.01*pl)/15,1,1)'
    assert s.count(b)==1;s=s.replace(b,b+',("weighted_motion",.01*motion/15,1,1)')
    namespace=dict(original.__globals__);exec(compile(s,str(ROOT/'gradient_probe.py'),'exec'),namespace)
    return namespace['basic_crop']

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--group',choices=GROUPS,required=True);group=p.parse_args().group
    import torch,loss
    env();c=build(0,train=True,losses=True,group=group);guard=FrozenGuard(frozen_items(c));seed(SEED)
    plan=plan_rows();rows=[next(r for r in plan if r['domain']==d) for d in ['uvg','ulong']]
    audits=[]
    for row in rows:
        frames=Frames(raw(row),c['device']);saved=rng_state()
        g,a=component_function(loss.basic_crop)(c['ctx'],frames,row['external_qp'],audit=True,raft=c['raft'],raft_transform=c['transform']);frames.close();assert_reset(c)
        comps=a.pop('audit_component_tensors');component_norms={k:norm(v) for k,v in comps.items()}
        assert all(component_norms[k]>0 for k in ['LPIPS','weighted_reconstruction','weighted_DISTS','weighted_motion'])
        assert component_norms['rate_proxy']==0,'RATE_OR_IDENTITY_PROXY_MUST_BE_CONSTANT_FOR_LORA'
        # Same model/RNG and sample: original implementation repeat error and auxiliary-GT equivalence.
        restore_rng(saved);frames=Frames(raw(row),c['device'])
        repeat,_=loss.basic_crop(c['ctx'],frames,row['external_qp'],raft=c['raft'],raft_transform=c['transform']);frames.close();assert_reset(c)
        repeat_error=norm([a-b for a,b in zip(g,repeat)])/max(norm(g),1e-30)
        restore_rng(saved);pipe=Pipeline([row],0);frames=pipe.next(row)
        optimized,diag=optimized_loss(loss.basic_crop)(c['ctx'],frames,row['external_qp'],raft=c['raft'],raft_transform=c['transform']);frames.close();pipe.close();assert_reset(c)
        error=norm([a-b for a,b in zip(g,optimized)])/max(norm(g),1e-30)
        assert error<1e-3,('LORA_AUXILIARY_EQUIVALENCE_FAILED',error)
        assert max(abs(a['losses'][k]-diag['losses'][k]) for k in diag['losses'])<1e-5
        guard.check()
        audits.append(dict(domain=row['domain'],video=row['video'],sample_key=digest(row),component_gradient_norms=component_norms,
            actual_repeated_reference_relative_L2=repeat_error,auxiliary_relative_L2=error,tolerance=1e-3,
            rate_proxy_constant_monitoring_only=True,deep_DISTS_gradient_checks=a['deep_gradient_checks'],losses=a['losses']))
    row=rows[0];pipe=Pipeline([row]*3,0);temporary=[]
    for step in range(1,4):
        frames=pipe.next(row);g,a=optimized_loss(loss.basic_crop)(c['ctx'],frames,row['external_qp'],raft=c['raft'],raft_transform=c['transform']);frames.close()
        assert all(torch.isfinite(x).all() for x in g) and norm(g)>0
        grads={n:float(v.norm()) for (n,p),v in zip(c['named'],g)}
        assert all(v>0 for n,v in grads.items() if '.up.' in n)
        if step==1:assert all(v==0 for n,v in grads.items() if '.down.' in n)
        else:assert all(v>0 for n,v in grads.items() if '.down.' in n)
        c['optimizer'].zero_grad(set_to_none=True)
        for p,v in zip(c['params'],g):p.grad=v
        torch.nn.utils.clip_grad_norm_(c['params'],1.,error_if_nonfinite=True);c['optimizer'].step();guard.check();assert_reset(c)
        assert all(p.grad is None for n,p in c['pm'].named_parameters() if '.lora.' not in n)
        temporary.append(dict(disposable_step=step,gradient_norms=grads,losses=a['losses']))
    pipe.close();guard.full()
    assert all(not torch.equal(p,c['initial_trainable'][n]) for n,p in c['named'])
    state=dict(lora_state={n:t.detach().cpu() for n,t in lora_state(c).items()},baseline_sha256=BASELINE_SHA,pilot_sha256=PILOT_SHA,group=group,
        disposable_engineering_only=True,formal_committed_updates=0)
    atomic_torch(ROOT/f'large_artifacts/engineering/{group}/disposable_lora.pt',state)
    dump(ROOT/f'audits/real_loss_gradients_{group}.json',dict(status='PASS',sample_audits=audits,temporary_updates=temporary,formal_updates=0,
        only_LoRA_changed=True,all_original_parameters_gradient_none=True,frozen_state_hash=guard.digest,TCG_frozen=True,
        reconstruction_VJP_scale=4096,unscale_count=1,loss_source_sha256=sha(ROOT/'loss.py'),
        rate_gradient_role='constant / monitoring, not training LoRA',proxy='Identity, zero',optimizer_fresh=True))
    print('REAL_LOSS_LORA_GRADIENT_FROZEN_AND_AUXILIARY_PASS')

if __name__=='__main__':main()
