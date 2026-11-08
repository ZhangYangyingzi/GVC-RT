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
    import torch,loss
    env();c=build(0,train=True,losses=True,group='A_control');guard=FrozenGuard(frozen_items(c))
    rows=plan_rows()[:4];assert all(r['domain']=='ulong' and r['logical_sample'] for r in rows)
    records=[]
    for row in rows[:2]:
        restore_rng(torch.load(INIT,map_location='cpu',weights_only=True)['all_logical_rank_RNG'][str(row['logical_rank'])])
        frames=Frames(raw(row),c['device'])
        gradients,diag=component_function(loss.basic_crop)(c['ctx'],frames,row['external_qp'],audit=True,raft=c['raft'],raft_transform=c['transform'])
        frames.close();components=diag.pop('audit_component_tensors');norms={k:norm(v) for k,v in components.items()}
        assert all(norms[k]>0 for k in ['LPIPS','weighted_reconstruction','weighted_DISTS','weighted_motion'])
        assert norms['rate_proxy']==0 and all(torch.isfinite(g).all() for g in gradients)
        assert all(p.grad is None for n,p in c['pm'].named_parameters() if '.lora.' not in n)
        guard.check();assert_reset(c)
        records.append(dict(sample_key=row['sample_key'],video=row['video'],domain='ulong',start=row['canonical_indices'][0],component_gradient_norms=norms,
            deep_DISTS_gradient_checks=diag['deep_gradient_checks'],losses=diag['losses'],new_window_read_audit_SHA256=sha(ROOT/f"audits/window_reads/{row['sample_key']}.json.gz")))
    assert all(int(s['step'])==2000 for s in c['optimizer'].state.values())
    for p,g in zip(c['params'],gradients):p.grad=g
    before=[p.detach().clone() for p in c['params']];torch.nn.utils.clip_grad_norm_(c['params'],1.,error_if_nonfinite=True);c['optimizer'].step()
    assert all(int(s['step'])==2001 for s in c['optimizer'].state.values()) and norm([p-b for p,b in zip(c['params'],before)])>0
    guard.full();assert_reset(c)
    dump(ROOT/'audits/loss_components.json',dict(status='PASS',samples=records,disposable_updates=1,formal_updates=0,absolute_optimizer_step_after_probe=2001,
        restored_parent_optimizer=True,only_LoRA_changed=True,TCG_and_original_parameters_frozen=True,rate_proxy_constant=True,VJP_scale4096=True,unscale_once=True,loss_SHA256=sha(ROOT/'loss.py')))
    print('ULONG_NEW_WINDOW_LOSS_VJP_AND_RESTORED_OPTIMIZER_PASS')

if __name__=='__main__':main()
