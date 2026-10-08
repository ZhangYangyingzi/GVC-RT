"""Additional completion checks before publication; no training or selection."""
import math,random
from mixed_io import *

def main():
    import torch
    from PIL import Image
    command([PYTHON,'-B',str(Path(__file__).resolve())]);frozen()
    final=load(ROOT/'final_integrity.json');assert final['status']=='PASS'
    assert final['completed_RD']==900 and final['completed_validation']==788
    assert not final['missing'] and not final['failed']
    assert inventory()==load(ROOT/'audits/continuation_inventory.json'),'historical files changed during continuation'
    rows=[json.loads(x) for x in (ROOT/'training_logs/mixed.jsonl').read_text().splitlines()]
    assert len(rows)==5000
    ulong=load(ROOT/'train_manifest_ulong.json')['videos'];uvg=load(ROOT/'train_manifest.json')['videos']
    lookup={'ulong':{v['filename']:v for v in ulong},'uvg':{v['name']:v for v in uvg}}
    qp=load(ROOT/'qp_schedule.json')['external_qp'];schedule=load(ROOT/'qp_semantics_audit.json')['actual_qps']
    rng=random.Random(load(ROOT/'config.json')['domain_seed'])
    for step,r in enumerate(rows,1):
        assert r['adaptation_step']==r['optimizer_update_count']==step and r['source_step']==1000
        assert r['domain']==('ulong' if rng.random()<.5 else 'uvg')
        v=lookup[r['domain']][r['video']]
        assert r['source_path']==v['path' if r['domain']=='ulong' else 'source_path']
        assert r['external_qp']==qp[step-1] and r['actual_qps']==schedule[str(qp[step-1])][:4]
        assert r['canonical_indices']==list(range(r['canonical_indices'][0],r['canonical_indices'][0]+4))
        expected=r['canonical_indices'] if r['domain']=='ulong' else [v['source_frame_indices'][i] for i in r['canonical_indices']]
        assert r['source_frame_indices']==expected and r['crop_size']==256
        assert r['crop_x']>=0 and r['crop_y']>=0 and r['all_finite']
        assert all(math.isfinite(x) for x in r.values() if isinstance(x,float))
        assert r['learning_rates']==dict(wrapper=5e-5,bridge=3e-6,generator=5e-7)
    records=[];index=load(ROOT/'checkpoint_index.json')
    assert set(index)==set(map(str,STEPS))
    assert load(ROOT/'frozen_checkpoint_index.json')['checkpoint_index_sha256']==sha(ROOT/'checkpoint_index.json')
    for step in STEPS:
        cp=index[str(step)];assert sha(cp['path'])==cp['sha256']
        state=torch.load(cp['path'],map_location='cpu',weights_only=True)
        assert state['adaptation_step']==step and state['source_step']==1000
        assert state['compression_hash']==evalcfg()['compression_hash']
        assert {k:tensor_hash(state[k]) for k in ('wrapper','bridge','generator')}==cp['module_hashes']
        assert state['config_sha256']==sha(ROOT/'config.json')
        if step:
            assert all(state['code_hashes'][n]==sha(ROOT/n) for n in ('train.py','data.py','mixed_io.py','v614_io.py'))
            assert state['optimizer']['state']
            assert all(int(s['step'])==step for s in state['optimizer']['state'].values())
        assert all(k in state for k in ('domain_rng_state','sampling_rng_state','python_rng_state','torch_rng_state','cuda_rng_state'))
        inf=cp['inference'];assert sha(inf['path'])==inf['sha256'];s=torch.load(inf['path'],map_location='cpu',weights_only=True)
        assert {k:tensor_hash(s[k]) for k in ('wrapper','bridge','generator')}==cp['module_hashes']
        records.append(dict(adaptation_step=step,sha256=cp['sha256'],full_resume_state=True));del state,s
    pngs=list((ROOT/'results').glob('*/Rate_*.png'));assert len(pngs)==12
    for p in pngs:
        with Image.open(p) as im:im.verify()
    for p,h in final['output_hashes'].items():assert sha(ROOT/p)==h
    dump(ROOT/'audits/completion_checks.json',dict(status='PASS',training_updates=5000,QP_replay_exact=True,domain_RNG_replay_exact=True,all_training_sources_in_frozen_train_manifests=True,checkpoints=records,RD_points=900,validation_points=788,PNG_decode_checks=12,historical_inventory_unchanged=True,finished_unix=time.time()))
    print('COMPLETION CHECKS PASS',flush=True)

if __name__=='__main__':main()
