"""Recover an isolated copy of an actual milestone, with zero optimizer updates."""
from common_s2 import *


def main():
    env();import torch
    import resume as recovery
    from adapter import CausalAdapter
    fixture=ROOT/'tmp/recovery_fixture';fixture.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'protocol.json',fixture/'protocol.json');records=[]
    for group in GROUPS:
        source=ROOT/group/'training/checkpoints/step_0075.pt'
        saved=torch.load(source,map_location='cpu',weights_only=True)
        target=fixture/group/'training/latest.pt';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        recovery.ROOT=fixture;recovery.repair_commit_metadata(group);recovery.repair_commit_metadata(group)
        cp=fixture/group/'training/checkpoints/step_0075.pt';a=torch.load(cp,map_location='cpu',weights_only=True)
        assert a['committed_updates']==75 and a['next_cursor']==300
        assert load(fixture/group/'training/progress.json')['committed_updates']==75
        assert load(fixture/group/'training/receipts/0075.json')['sample_keys']==saved['receipt']['sample_keys']
        branch=CausalAdapter();branch.load_state_dict(a['adapter'],strict=True)
        optimizer=torch.optim.AdamW(branch.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
        optimizer.load_state_dict(a['optimizer']);assert all(int(v['step'])==75 for v in optimizer.state.values())
        def ohash(state):return tensor_hash([(str(i)+k,v) for i,s in state['state'].items() for k,v in s.items() if torch.is_tensor(v)])
        assert ohash(optimizer.state_dict())==ohash(saved['optimizer'])
        restore_rng(a['RNG']);restored=rng_state()
        assert restored['python']==a['RNG']['python']
        assert torch.equal(restored['torch'],a['RNG']['torch']) and torch.equal(restored['cuda'],a['RNG']['cuda'])
        assert restored['numpy'][0]==a['RNG']['numpy'][0] and torch.equal(restored['numpy'][1],a['RNG']['numpy'][1]) and restored['numpy'][2:]==a['RNG']['numpy'][2:]
        records.append(dict(group=group,source=str(source),source_SHA256=sha(source),fixture_checkpoint_SHA256=sha(cp),
            committed_updates=75,next_cursor=300,optimizer_state_exact=True,RNG_exact=True,idempotent_repair=True))
    dump(ROOT/'audits/recovery_check.json',dict(status='PASS',records=records,optimizer_updates=0,
        production_files_modified=False,scenario='interruption after atomic latest but before milestone/receipt/progress publication',time=time.time()))
    print('RECOVERY_PASS')

if __name__=='__main__':main()
