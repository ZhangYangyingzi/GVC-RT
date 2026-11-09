"""Verify all four parameter groups and optimizer/RNG survive a saved smoke state."""
from common_s2 import *
import argparse

def main(group,rank):
    import torch
    from experiment_model import construct,restore,state,reconstruct
    from model_context import assert_reset
    from cache_codec import cache_paths
    env();cp=ROOT/group/'engineering/latest.pt';saved=torch.load(cp,map_location='cpu',weights_only=True)
    assert saved['committed_updates']==2 and saved['next_cursor']==8
    equivalence=load(ROOT/f'audits/gradient_equivalence_{group}.json')
    objective=load(ROOT/group/'engineering/receipts/0001.json')['loss']
    assert abs(objective-equivalence['reference_objective'])<1e-6,('OBJECTIVE_EQUIVALENCE',objective,equivalence)
    seed(PLAN_SEED);c=construct(group,True,rank);restore(c,saved);restore_rng(saved['RNG_by_rank'][0])
    assert all(torch.equal(t,saved['trainable_state'][n]) for n,t in state(c).items())
    assert sum(t.numel() for t in state(c).values())==34774916
    assert all(int(s['step'])==2 for s in c['optimizer'].state.values())
    r=plan_rows()[0];_,path,_=cache_paths(r,r['external_qp'],True);d=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    with torch.no_grad():
        assert_reset(c);out=reconstruct(c,d['features'][1].to(rank),d['info']['actual_qps'][1]);h=tensor_sha(out)
        for n,p in c['named']:p.add_(.0001)
        restore(c,saved);out2=reconstruct(c,d['features'][1].to(rank),d['info']['actual_qps'][1]);assert tensor_sha(out2)==h
    dump(ROOT/f'audits/checkpoint_reload_{group}.json',dict(status='PASS',groups=['Bridge','Backbone','TCG','LoRA'],parameters=34774916,objective_equivalence_absolute_error=abs(objective-equivalence['reference_objective']),exact_trainable_roundtrip=True,mutated_then_restored_forward_hash_equal=True,optimizer_steps=2,RNG_and_cursor_present=True,SHA256=sha(cp),time=time.time()))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--group',required=True);a.add_argument('--rank',type=int,required=True);v=a.parse_args();main(v.group,v.rank)
