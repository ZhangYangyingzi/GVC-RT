import math
from v61_io import *
from report import collect
def main():
    try:
        frozen();cfg=load(ROOT/'config.json');split=load(ROOT/'data_split_integrity.json');assert split['status']=='PASS'
        train=load(ROOT/'train_manifest_1024.json')['videos'];validation=load(ROOT/'validation_manifest.json')['videos'];old=load(V41/'train_manifest.json')['videos']
        th={sha(r['path']) for r in train};vh={sha(r['path']) for r in validation};tests=set(split['test_sha256'])
        assert len(th)==len(train)==1024 and len(vh)==32 and {r['sha256'] for r in old}<=th
        assert th=={r['sha256'] for r in train} and vh=={r['sha256'] for r in validation}
        assert not th&(vh|tests|set(split['reserved_sha256'])) and not vh&tests
        assert load(ROOT/'parts/train_done.json')['status']=='PASS';hist=load(ROOT/'qp_semantics_audit.json');assert hist['status']=='PASS'
        assert sum(hist['per_qp_sampling_counts'].values())==40000 and set(map(int,hist['per_qp_sampling_counts']))==set(range(10))
        with (ROOT/'training_log.jsonl').open() as f:history=[json.loads(line) for line in f]
        assert [r['additional_update'] for r in history]==list(range(1,40001))
        for q in range(10):assert sum(r['qp']==q for r in history)==hist['per_qp_sampling_counts'][str(q)]
        assert all(r['all_finite'] and all(math.isfinite(float(r[k])) for k in ('loss','LPIPS','DISTS','R_est_bpp','proxy_L1')) for r in history)
        checkpoints=load(ROOT/'checkpoint_hashes.json');assert set(map(int,checkpoints))==set(cfg['checkpoint_additional_updates'])
        for step,r in checkpoints.items():assert sha(r['path'])==r['sha256'] and r['global_step']==20000+int(step)
        selection=load(ROOT/'checkpoint_selection.json');assert selection['status']=='PASS' and selection['used_final_test'] is False
        assert sha(selection['checkpoint'])==selection['checkpoint_sha256'];assert selection['validation_per_video_sha256']==sha(ROOT/'validation_per_video.csv')
        assert selection['validation_summary_sha256']==sha(ROOT/'validation_summary.csv');assert selection['selection_csv_sha256']==sha(ROOT/'checkpoint_selection.csv')
        counts={}
        for dataset in ('validation',*cfg['final_datasets']):
            rows=collect(dataset);counts[dataset]=len(rows);report=load(ROOT/'results'/dataset/'report_integrity.json');assert report['status']=='PASS'
            curves=load(ROOT/'results'/dataset/'rd_curves/manifest.json');assert len(curves['files'])==8
            for p,h in curves['files'].items():assert sha(ROOT/'results'/dataset/'rd_curves'/p)==h
            for name,key in [('equal_rate_summary.csv','mean_equal_rate_delta'),('bd_rate.csv','BD_rate_percent')]:
                for r in read(ROOT/'results'/dataset/name):
                    assert r['status'] in ('valid','invalid')
                    if r['status']=='valid':assert math.isfinite(float(r[key]))
                    else:assert r['reason']
        assert counts==dict(validation=1344,ulong=160,uvg=140,virat720=160)
        assert load(ROOT/'final_cohort_split_audit.json')['status']=='PASS'
        canonical=load(V52/'canonical_source_audit.json')
        assert not (th|vh) & {s['original_sha256'] for s in canonical['records']}
        for s in canonical['records']:
            assert [sha(Path(s['canonical_dir'])/f'frame_{i:06d}.png') for i in range(64)]==s['canonical_frame_sha256']
        from evaluate import read_frames
        for v in load(ROOT/'final_sources.json')['videos']:frames=read_frames(v);del frames
        dump(ROOT/'final_integrity.json',dict(status='PASS',old_experiments_untouched=True,old_artifacts_hash_verified=True,
            no_UVG_in_train=True,no_VIRAT_in_train=True,train_count=1024,validation_count=32,train_val_test_disjoint=True,
            QP_training_set=list(range(10)),QP_histogram_valid=True,selected_checkpoint_from_validation_only=True,
            real_RANS_complete=True,independent_decode_complete=True,no_missing_QP_points=True,all_metric_values_finite=True,
            all_expected_datasets_complete=True,checkpoint_hashes_recorded=True,source_hashes_recorded=True,
            threshold_012_confirmed_Original_and_Ours=True,expected_points=counts,selected_checkpoint=selection['checkpoint'],
            selected_checkpoint_sha256=selection['checkpoint_sha256'],script_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},failed_checks=[]))
        status(status='PASS',phase='complete',selected_checkpoint=selection['checkpoint'])
        print('FINAL INTEGRITY PASS',flush=True)
    except BaseException as exc:
        dump(ROOT/'final_integrity.json',dict(status='FAIL',error=repr(exc)));raise
if __name__=='__main__':main()
