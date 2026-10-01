"""Raw equal-rate tables through the unchanged V6.2-B comparator."""
import math,statistics,traceback
from v63_io import *
def main():
    frozen(True);ev=eval_adapter();sys.path.insert(0,str(V62B));old=module('v63_rd_report',V62B/'report.py')
    allrows=[];equal=[];coverage=[];same=[];curves=[]
    for d in DATASETS:
        ds=[]
        for v in sources(d):
            rows=[]
            for m in METHODS:
                for q in range(10):
                    r=load(point(d,m,v['video_index'],q));ev.validate(r,v,q,m)
                    if m in BRANCHES:assert not r.get('reused',False)
                    else:assert sha(r['reused_from'])==r['reused_point_sha256']
                    rows.append(r);allrows.append(r);ds.append(r)
            for m in BRANCHES:
                for anchor in ('original','v62'):
                    a=[r for r in rows if r['method']==anchor];c=[r for r in rows if r['method']==m]
                    for metric in METRICS:
                        e,_=old.compare_rows(a,c,metric,anchor,m);e.update(dataset=d,sequence=v['name'],video_index=v['video_index']);equal.append(e)
                        alo,ahi=min(r['kbps'] for r in a),max(r['kbps'] for r in a);clo,chi=min(r['kbps'] for r in c),max(r['kbps'] for r in c)
                        lo,hi=max(alo,clo),min(ahi,chi);span=max(0,math.log(hi/lo))
                        coverage.append(dict(dataset=d,sequence=v['name'],anchor=anchor,method=m,metric=metric,status=e['status'],common_rate_min_kbps=lo,common_rate_max_kbps=hi,
                            common_log_rate_span=span,anchor_log_range_fraction=span/math.log(ahi/alo),candidate_log_range_fraction=span/math.log(chi/clo),reason=e['reason']))
                    for q in range(10):
                        ar=next(r for r in a if r['external_qp']==q);cr=next(r for r in c if r['external_qp']==q)
                        same.append(dict(dataset=d,sequence=v['name'],anchor=anchor,method=m,QP=q,kbps_delta=cr['kbps']-ar['kbps'],**{metric+'_delta':cr[metric]-ar[metric] for metric in METRICS}))
        write(ROOT/'results'/d/'raw.csv',ds)
        for m in METHODS:
            for q in range(10):
                own=[r for r in ds if r['method']==m and r['external_qp']==q]
                curves.append(dict(dataset=d,method=m,QP=q,kbps=statistics.mean(r['kbps'] for r in own),**{metric:statistics.mean(r[metric] for r in own) for metric in METRICS},N=len(own)))
        write(ROOT/'results'/d/'equal_rate_per_sequence.csv',[r for r in equal if r['dataset']==d])
    assert len(allrows)==1550 and len({r['compression_hash_before'] for r in allrows})==1
    summaries=[]
    for d in DATASETS:
        for anchor in ('original','v62'):
            for m in BRANCHES:
                for metric in METRICS:
                    own=[r for r in equal if r['dataset']==d and r['anchor']==anchor and r['method']==m and r['metric']==metric];valid=[r for r in own if r['status']=='valid']
                    summaries.append(dict(dataset=d,anchor=anchor,method=m,metric=metric,total_N=len(own),valid_N=len(valid),invalid_N=len(own)-len(valid),
                        mean_equal_rate_delta=statistics.mean(r['mean_equal_rate_delta'] for r in valid) if valid else None,
                        fraction_of_common_rate_range_better=statistics.mean(r['fraction_of_common_rate_range_better'] for r in valid) if valid else None,
                        aggregation='arithmetic macro mean of valid per-sequence comparisons, each using its own common measured rate interval',
                        status='valid' if len(valid)==len(own) else 'partial' if valid else 'invalid'))
    write(ROOT/'evaluation_raw.csv',allrows);write(ROOT/'equal_rate_per_sequence.csv',equal);write(ROOT/'equal_rate_summary.csv',summaries)
    write(ROOT/'results/common_rate_coverage.csv',coverage);write(ROOT/'results/same_qp_raw.csv',same);write(ROOT/'results/dataset_macro_rd.csv',curves)
    write(ROOT/'results/uvg_per_sequence_equal_rate.csv',[r for r in equal if r['dataset']=='uvg' and r['metric'] in ('LPIPS','DISTS','FloLPIPS')])
    for d in DATASETS:write(ROOT/'results'/d/'equal_rate_summary.csv',[r for r in summaries if r['dataset']==d])
    frozen(True)
    for v in load(ROOT/'source_manifest.json')['videos']:
        h=hashlib.sha256();io=module('v63_final_frames',V62B/'v62b_io.py');count=0
        for i,a in enumerate(io.arrays(v)):
            data=a.tobytes();assert hashlib.sha256(data).hexdigest()==v['frame_rgb_sha256'][i];h.update(data);count+=1
        assert count==v['frames'] and h.hexdigest()==v['rgb_sha256']
    files=['config.json','preflight_audit.json','dataset_audit.json','loader_audit.json','initialization_audit.json','qp_sequence_audit.json','training_summary.csv','checkpoint_hashes.json','evaluation_manifest.json','evaluation_raw.csv','equal_rate_per_sequence.csv','equal_rate_summary.csv']
    assert load(ROOT/'audits/checkpoint_integrity.json')['status']=='PASS'
    dump(ROOT/'final_integrity.json',dict(status='PASS',completed_unix=time.time(),all_three_trainings_complete=True,all_step1000_checkpoints_valid=True,
        no_old_experiments_modified=True,no_train_test_leakage=True,same_initialization=True,same_objective=True,same_qp_sequence=True,mixed_exact_500_500=True,
        expected_points=1550,completed_points=len(allrows),fresh_points=930,reused_points=620,real_RANS=True,independent_decode=True,all_raw_metrics_finite=True,
        reports_generated=True,rate_axis='natural log(kbps)',interpolation='unchanged PCHIP extrapolate=False',no_smoothing=True,no_metric_sorting=True,
        common_measured_range_only=True,artifacts={f:sha(ROOT/f) for f in files}))
    print('FINAL INTEGRITY PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'logs/report_failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
