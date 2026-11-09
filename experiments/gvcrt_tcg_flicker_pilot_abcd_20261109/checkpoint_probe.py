from common_s2 import *
from experiment_model import construct,restore,state
import argparse

def main(group,rank):
    import torch
    cp=ROOT/group/'smoke/latest.pt';saved=torch.load(cp,map_location='cpu',weights_only=True);assert saved['committed_updates']==20 and saved['next_cursor']==80
    c=construct(group,True,rank);dump(ROOT/f'audits/formal_initial_provenance_{group}.json',c['provenance']);restore(c,saved);restore_rng(saved['RNG'])
    assert all(torch.equal(t,saved['trainable_state'][n]) for n,t in state(c).items())
    items=lambda o:[(str(i)+k,t) for i,s in o.state_dict()['state'].items() for k,t in s.items() if torch.is_tensor(t)]
    current=tensor_hash(items(c['optimizer']));expected=tensor_hash([(str(i)+k,t) for i,s in saved['optimizer']['state'].items() for k,t in s.items() if torch.is_tensor(t)])
    assert current==expected
    dump(ROOT/f'audits/checkpoint_reload_{group}.json',dict(status='PASS',checkpoint_SHA256=sha(cp),disposable_updates=20,cursor=80,parameters_equal=True,optimizer_state_equal=True,RNG_restored=True,
        formal_training_will_reinitialize_from_original_A2000=True,parent_A2000_sha256=INIT_SHA,time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--rank',type=int,required=True);a=p.parse_args();main(a.group,a.rank)
