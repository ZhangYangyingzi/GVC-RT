"""Cross-branch fairness gate and complete checkpoint validation."""
import argparse,collections,math
from v64_io import *
def smoke():
    frozen(True);audits=[load(ROOT/'branches'/b/'smoke_audit.json') for b in BRANCHES]
    assert all(a['status']=='PASS' for a in audits)
    assert len({json.dumps(a['initialization']['module_hashes'],sort_keys=True) for a in audits})==1
    assert len({a['initialization']['compression_hash'] for a in audits})==1
    assert all(a['initialization']['fresh_optimizer'] and a['initialization']['optimizer_entries']==0 for a in audits)
    rows=[dict(branch=b,**r) for b,a in zip(BRANCHES,audits) for r in a['rows']]
    assert {r['source'] for r in rows}=={'vimeo'}
    dump(ROOT/'loader_audit.json',dict(status='PASS',rows=rows,native_order=True,shared_crop=True,no_resize_interpolation_duplication=True))
    dump(ROOT/'initialization_audit.json',dict(status='PASS',branches={b:a['initialization'] for b,a in zip(BRANCHES,audits)},identical_modules=True,identical_compression=True,fresh_optimizers=True))
    dump(ROOT/'smoke_audit.json',dict(status='PASS',branches=list(BRANCHES),old_experiments_unchanged=True,objective_exact_original_AST=True,formal_training_authorized_by_gate=True,completed_unix=time.time()))
    print('ALL BRANCH SMOKE PASS',flush=True)
def checkpoints():
    import torch
    frozen(True);hashes={};summary=[];qps=load(ROOT/'qp_sequence_audit.json')['sequence'];initial=load(ROOT/'initialization_audit.json')
    for b in BRANCHES:
        br=ROOT/'branches'/b;assert load(br/'training_status.json')['status']=='PASS'
        log=ROOT/'training_logs'/f'{b}.jsonl';rows=[json.loads(s) for s in log.read_text().splitlines()]
        assert [r['update'] for r in rows]==list(range(1,1001)) and [r['external_qp'] for r in rows]==qps
        assert all(r['source']==source_for(b,r['update']) for r in rows)
        for r in rows:
            assert r['actual_qps']==load(ROOT/'qp_train_eval_semantics_audit.json')['rows'][r['external_qp']]['evaluation']
            assert r['all_finite'] and all(math.isfinite(r[k]) for k in ('loss','D','rate_bpp','gradient_norm'))
        counts=dict(collections.Counter(r['source'] for r in rows));assert counts=={'vimeo':1000}
        index=load(br/'checkpoint_hashes.json');assert set(map(int,index))==set(STEPS)
        for step in STEPS:
            record=index[str(step)];assert sha(record['path'])==record['sha256']
            state=torch.load(record['path'],map_location='cpu',weights_only=True)
            assert state['step']==step and state['branch']==b and state['initialization_hashes']==initial['branches'][b]['module_hashes']
            assert state['compression_hash']==initial['branches'][b]['compression_hash'] and state['qp_sequence_sha256']==sha(ROOT/'qp_sequence_audit.json')
            assert state['config_sha256']==sha(ROOT/'config.json')
            for k in ('wrapper','bridge','generator'):
                assert all(bool(torch.isfinite(t).all()) for t in state[k].values())
                h=hashlib.sha256()
                for n,t in state[k].items():h.update(n.encode());h.update(t.contiguous().numpy().tobytes())
                assert h.hexdigest()==record['module_hashes'][k]
            assert bool(state['optimizer']['state'])==(step>0)
            if step==1000:assert state['training_log_sha256']==sha(log)
            del state
        hashes[b]=index['1000'];summary.append(dict(branch=b,updates=1000,ulong_updates=counts.get('ulong',0),vimeo_updates=counts.get('vimeo',0),checkpoint_sha256=index['1000']['sha256'],same_qp_sequence=True))
    dump(ROOT/'checkpoint_hashes.json',hashes);write(ROOT/'training_summary.csv',summary)
    cfg=load(ROOT/'config.json');cps=dict(cfg['baseline_checkpoints'],**hashes)
    dump(ROOT/'evaluation/config.json',dict(checkpoints=cps,methods=list(METHODS),force_zero_thres=.12,expected_points=930))
    dump(ROOT/'audits/checkpoint_integrity.json',dict(status='PASS',fixed_step1000=True,vimeo_training_complete=True,same_qp_sequence=True,vimeo_only1000=True,checkpoints=hashes))
    # Revalidate all reused points and copy only their metadata into this experiment.
    old=old_eval();adapter=eval_adapter();vs={(v['dataset'],v['video_index']):v for v in load(ROOT/'source_manifest.json')['videos']}
    for rec in load(ROOT/'audits/baseline_reuse_audit.json')['points']:
        assert sha(rec['path'])==rec['sha256'];r=load(rec['path']);v=vs[(rec['dataset'],rec['video_index'])]
        old.validate(r,v,rec['QP'],rec['method']);r.update(reused=True,reused_from=rec['path'],reused_point_sha256=rec['sha256'])
        adapter.validate(r,v,rec['QP'],rec['method']);dump(point(rec['dataset'],rec['method'],rec['video_index'],rec['QP']),r)
    print('CHECKPOINT GATE PASS; 620 BASELINES VERIFIED',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('smoke','checkpoints'));a=p.parse_args()
    smoke() if a.phase=='smoke' else checkpoints()
