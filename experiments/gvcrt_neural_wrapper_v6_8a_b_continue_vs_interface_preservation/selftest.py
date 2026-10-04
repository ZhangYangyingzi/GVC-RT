"""CPU-only schema, frozen-cohort, PCHIP and adapter checks."""
import ast,math
from v68_io import *
def main():
    for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    plans=load(ROOT/'audits/continuation_training_plans.json')['plans'];assert len(plans)==500
    assert [p['absolute_step'] for p in plans]==list(range(1001,1501))
    train=set(load(V64/'manifests/vimeo_official_train.json')['sequences']);held=load(ROOT/'audits/vimeo_heldout_manifest.json');assert not train&{v['sequence'] for v in held['clips']}
    for p in plans:assert p['sample_id'] in train and p['actual_qps']==[p['external_qp']+j for j in (0,2,0,1)] and p['native_image_indices']==[x+1 for x in p['temporal_indices']]
    assert len(sources())==31 and sum(len(sources(d))*10 for d in DATASETS)==310
    for d in DATASETS:
        cache=load(ROOT/'audits/GT_feature_cache.json')['datasets'][d];assert sha(cache['path'])==cache['sha256']
        assert load(ROOT/'audits'/f'fid_bootstrap_indices_{d}.json')==load(V67/'audits'/f'fid_bootstrap_indices_{d}.json')
    sys.path.insert(0,str(V62B));rd=module('v68_selftest_rd',V62B/'report.py')
    own=[load(point('uvg','B1000',0,q)) for q in range(10)]
    for r in own:validate_point(r,sources('uvg')[0],r['QP'],'B1000')
    for metric in METRICS:
        result,_=rd.compare_rows(own,own,metric,'B1000','B1000');assert result['status']=='valid' and abs(result['mean_equal_rate_delta'])<1e-12
    for name in ('fid_worker','heldout'):
        m=adapted_script(name);assert m.ROOT==ROOT and m.METHODS==METHODS and m.B_METHODS==METHODS
        assert 'B1000' in m.load(ROOT/'config.json')['checkpoints']
    dump(ROOT/'audits/selftest.json',dict(status='PASS',syntax=True,plans=500,train_test_disjoint=True,formal_sequences=31,raw_target=1550,FID_target=200,heldout_target=480,identical_curve_PCHIP_zero_delta=True,GT_cache_hashes_match=True,bootstrap_indices_identical=True,read_only_worker_adapters=True))
    print('SELFTEST PASS',flush=True)
if __name__=='__main__':main()
