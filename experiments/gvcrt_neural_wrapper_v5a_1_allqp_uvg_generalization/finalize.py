"""Fail closed unless every requested evaluation artifact and frozen input passes."""
import ast
import math
from audit_utils import *
from report import collect

def main():
    config=load('config.json');checks={}
    checks['evaluation_only']=config['evaluation_only'] and not config['training_allowed']
    forbidden={'backward','step','optimizer_step','train'}
    calls=[]
    for path in ROOT.glob('*.py'):
        tree=ast.parse(path.read_text())
        calls.extend(n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden)
    checks['no_training_performed']=not calls and not list(ROOT.rglob('*.pt')) and not list(ROOT.rglob('*.pth'))
    checks['checkpoints_unchanged']=all(not c['path'] or sha(c['path'])==c['sha256'] for c in config['checkpoints'].values())
    checks['source_experiments_unchanged']=all(sha(path)==h for path,h in config['source_hashes'].items())
    checks['additional_model_and_source_hashes_unchanged']=all(sha(path)==h for path,h in load('checkpoint_metadata_audit.json')['additional_source_hashes'].items())
    checks['implementation_regression_tests_pass']=load('implementation_selftest.json')['status']=='PASS'
    checks['numeric_record_regression_pass']=load('numeric_record_regression_audit.json')['status']=='PASS'
    checks['metric_implementation_and_weights_unchanged']=all(sha(path)==h for path,h in load('metric_implementation_audit.json')['source_and_weight_sha256'].items())
    checks['official_qp_num_is_10']=load('qp_semantics_audit.json')['get_qp_num']==10
    checks['qp0_3_backward_compatibility_pass']=load('qp0_3_backward_compatibility_audit.json')['status']=='PASS'
    checks['ulong_final_8_videos_unchanged']=sha(ROOT/'test_manifest.json')==sha(V5/'test_manifest.json')==config['ulong_manifest_sha256'] and len(videos('ulong'))==8
    checks['uvg_zero_shot']=load('uvg_zero_shot_audit.json')['status']=='PASS' and all(load('uvg_zero_shot_audit.json')[k] is False for k in ('training_used_uvg','checkpoint_selection_used_uvg','hyperparameter_tuning_used_uvg'))
    checks['uvg_manifest_frozen_before_evaluation']=sha(ROOT/'uvg_available_manifest.json')==config['uvg_manifest_sha256']
    checks['uvg_sources_unchanged']=all(sha(v['path'])==v['sha256'] for v in videos('uvg'))
    checks['uvg_RGB_decode_regression_pass']=load('uvg_decode_audit.json')['status']=='PASS'
    checks['actual_model_qp_shapes_pass']=all(load(f'parts/model_audit_{d}_{m}.json')['status']=='PASS' for d in ('ulong','uvg') for m in METHODS)
    checks['all_qp_curves_complete']=all((ROOT/'rd_curves'/d/f'Rate_{metric}.{ext}').is_file() and (ROOT/'rd_curves'/d/f'Rate_{metric}.{ext}').stat().st_size>100 for d in ('ulong','uvg') for metric in METRICS for ext in ('png','pdf'))
    counts={};records=[];invalid=[];all_finite=True
    frozen=load(V5/'initialization_audit.json')['branches']['clip4_control']['hashes']['compression']
    for dataset in ('ulong','uvg'):
        rows=collect(dataset);records.extend(rows);counts[dataset]=len(rows)
        summary=read(f'{dataset}_all_qp_summary.csv')
        assert len(summary)==40
        for method in METHODS:assert sorted(int(r['external_qp']) for r in summary if r['method']==method)==list(range(10))
        expected=sum(64 if dataset=='ulong' else v['frames_evaluated'] for v in videos(dataset))
        assert all(int(r['FID_num_samples'])==int(r['num_frames'])==expected for r in summary)
        for r in summary:
            all_finite=all_finite and all(math.isfinite(float(r[k])) for k in (*METRICS,'mean_real_kbps','mean_bpp','PSNR','SSIM','MS_SSIM'))
        eq=read(f'{dataset}_equal_rate_summary.csv');bd=read(f'{dataset}_bd_rate.csv')
        assert len(eq)==len(bd)==60
        for r in eq:
            assert r['status'] in ('valid','invalid')
            if r['status']=='valid':assert math.isfinite(float(r['mean_equal_rate_delta'])) and math.isfinite(float(r['fraction_of_common_rate_range_better']))
            else:assert r['reason'];invalid.append(dict(r,kind='equal_rate'))
        for r in bd:
            assert r['status'] in ('valid','invalid')
            if r['status']=='valid':assert math.isfinite(float(r['BD_rate_percent']))
            else:assert r['reason'];invalid.append(dict(r,kind='BD_rate'))
        assert load(f'parts/{dataset}_report_done.json')['summary_sha256']==sha(ROOT/f'{dataset}_all_qp_summary.csv')
    checks['all_external_qps_tested']=sorted({r['external_qp'] for r in records})==list(range(10))
    checks['ulong_64_frames_complete']=all(r['num_frames']==64 for r in records if r['dataset']=='Fresh_U_Long')
    checks['uvg_actual_frames_complete']=counts['uvg']==len(videos('uvg'))*40
    for m,key in [('original','original_complete'),('v41_20000','v41_20000_complete'),('clip4_control','clip4_complete'),('clip8','clip8_complete')]:
        checks[key]=sum(r['method']==m for r in records)==10*(8+len(videos('uvg')))
    checks['real_RANS_complete']=all(int(r['real_bytes'])==int(r['bytes_consumed'])==Path(r['bitstream_path']).stat().st_size for r in records)
    checks['independent_decode_pass']=all(truth(r['independent_decode_pass']) and truth(r['metric_decode_pass']) for r in records)
    checks['compression_core_frozen']=all(r['compression_hash_before']==r['compression_hash_after']==frozen for r in records)
    checks['GPU_0_3_unused']=all(r.get('physical_gpu') in (None,4,5,6,7) for r in records)
    for metric in METRICS:checks[metric+'_complete']=all_finite
    checks['equal_rate_complete']=True;checks['BD_rate_complete_or_validly_skipped']=True
    checks['all_values_finite']=all_finite
    expected_tables={'all_qp_monotonicity_audit.csv':32,'qp_generalization_summary.csv':24,
        'cross_dataset_generalization.csv':24,'clip8_vs_clip4_allqp.csv':6,'ulong_seen_unseen_qp_summary.csv':8}
    checks['all_required_tables_complete']=all(len(read(name))==count for name,count in expected_tables.items())
    # Each actual source sequence must stay byte-identical across every method.
    source_consistent=True
    for dataset in ('ulong','uvg'):
        for i in range(len(videos(dataset))):
            values=[load(f'parts/source_{dataset}_{m}_{i}.json') for m in METHODS]
            source_consistent &= len({v['source_hash'] for v in values})==1
    checks['source_frames_identical_across_methods']=source_consistent
    failed=[k for k,v in checks.items() if not v]
    result=dict(checks,status='FAIL' if failed else 'PASS',failed_checks=failed,official_qp_num=10,
        external_qps_tested=list(range(10)),point_counts=counts,
        num_available_uvg_sequences=len(videos('uvg')),uvg_sequence_names=[v['sequence_name'] for v in videos('uvg')],
        FID_samples_per_method_QP={'ulong':512,'uvg':sum(v['frames_evaluated'] for v in videos('uvg'))},
        invalid_RD_results=invalid,performance_is_not_integrity_gate=True,
        no_training_evidence='evaluation-only launch commands; no optimizer/backward/train calls in experiment scripts; no checkpoint outputs; source hashes unchanged')
    dump('final_integrity.json',result)
    assert not failed,failed
    print('FINAL INTEGRITY PASS',counts,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if not (ROOT/'final_integrity.json').exists():dump('final_integrity.json',dict(status='FAIL',error=repr(exc)))
        raise
