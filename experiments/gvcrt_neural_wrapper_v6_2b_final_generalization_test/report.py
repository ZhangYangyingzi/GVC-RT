"""Numerical comparisons only. No training, checkpoint search, or conclusions."""
import argparse
import math
import statistics
from v62b_io import *
from evaluate import validate

def collect(dataset):
    rows=[]
    for v in sources(dataset):
        for method in METHODS:
            for q in range(10):
                r=load(point(dataset,method,v['video_index'],q));validate(r,v,q,method);rows.append(r)
    assert len(rows)==len(sources(dataset))*30
    assert len({r['compression_hash_before'] for r in rows})==1
    return rows

def same_qp(rows,sequence=False):
    output=[]
    for r in rows:
        anchor=next(a for a in rows if a['method']=='original' and a['external_qp']==r['external_qp'] and (not sequence or a['video_index']==r['video_index']))
        out=dict(r,same_qp_bitrate_delta_percent=100*(float(r['kbps'])/float(anchor['kbps'])-1))
        for k in METRICS:out['same_qp_'+k+'_delta']=float(r[k])-float(anchor[k])
        output.append(out)
    return output

def compare_rows(anchor,candidate,metric,anchor_name,method):
    sys.path.insert(0,str(V52));from rd_analysis import compare
    if metric in HIGHER:
        aa=[dict(r,**{metric:-float(r[metric])}) for r in anchor];cc=[dict(r,**{metric:-float(r[metric])}) for r in candidate]
    else:aa,cc=anchor,candidate
    e,b=compare(aa,cc,metric,anchor_name,method)
    if metric in HIGHER and e['status']=='valid':e['mean_equal_rate_delta']=-e['mean_equal_rate_delta']
    for r in (e,b):r.update(metric_direction='higher_is_better' if metric in HIGHER else 'lower_is_better',
        higher_metric_adaptation='negative metric passed to unchanged lower-distortion comparator; reported equal-rate delta restored to original units' if metric in HIGHER else 'none')
    return e,b

def pooled(rows):
    import numpy as np
    fid=module('v62b_fid_report',FID_SOURCE);out=[];references={}
    for method in METHODS:
        for q in range(10):
            own=[r for r in rows if r['method']==method and r['external_qp']==q];real=[];recon=[]
            for r in own:
                with np.load(r['feature_path']) as f:
                    assert f['real'].shape==f['reconstruction'].shape==(r['frames'],2048)
                    assert np.isfinite(f['real']).all() and np.isfinite(f['reconstruction']).all()
                    k=r['video_index']
                    if k in references:assert np.allclose(f['real'],references[k],rtol=1e-5,atol=1e-6),('Reference feature mismatch',method,q,k)
                    else:references[k]=f['real'].copy()
                    real.append(f['real']);recon.append(f['reconstruction'])
            ra,rc=np.concatenate(real),np.concatenate(recon);value=float(fid.low_rank_fid(ra,rc));assert math.isfinite(value)
            out.append(dict(dataset=own[0]['dataset'],method=method,external_qp=q,QP=q,
                kbps=statistics.mean(r['kbps'] for r in own),bpp=statistics.mean(r['bpp'] for r in own),
                real_bytes=sum(r['real_bytes'] for r in own),total_bits=sum(r['total_bits'] for r in own),
                bits_per_frame=sum(r['total_bits'] for r in own)/sum(r['frames'] for r in own),
                rate_accounting_fps=own[0]['rate_accounting_fps'],
                **{k:statistics.mean(r[k] for r in own) for k in SPATIAL},
                FloLPIPS=sum(r['FloLPIPS']*r['num_transitions'] for r in own)/sum(r['num_transitions'] for r in own),FID=value,
                num_sequences=len(own),num_frames=sum(r['frames'] for r in own),num_transitions=sum(r['num_transitions'] for r in own),FID_num_samples=len(ra),
                aggregate_rule='kbps/bpp/spatial metrics: arithmetic sequence mean; FloLPIPS: transition-weighted; FID: pooled frame features; bytes/bits: sum'))
    return out

def report_dataset(dataset):
    check_frozen();rows=collect(dataset);summary=same_qp(pooled(rows));perseq=same_qp(rows,True)
    out=ROOT/'results'/dataset;write(out/'all_qp_summary.csv',summary);write(out/'per_sequence_all_qp.csv',perseq)
    equal=[];bd=[]
    for anchor,method in PAIRS:
        a=[r for r in summary if r['method']==anchor];c=[r for r in summary if r['method']==method]
        for metric in METRICS:
            e,b=compare_rows(a,c,metric,anchor,method)
            for r in (e,b):r['dataset']=dataset
            equal.append(e);bd.append(b)
    write(out/'equal_rate_summary.csv',equal);write(out/'bd_rate.csv',bd)
    improvement=[]
    for m in METRICS:
        a=next(r for r in equal if r['anchor']=='original' and r['method']=='v41' and r['metric']==m)
        b=next(r for r in equal if r['anchor']=='original' and r['method']=='v62' and r['metric']==m)
        direct=next(r for r in equal if r['anchor']=='v41' and r['method']=='v62' and r['metric']==m)
        valid=a['status']==b['status']=='valid'
        improvement.append(dict(dataset=dataset,metric=m,V41_equal_rate_delta_vs_Original=a['mean_equal_rate_delta'],
            V62_equal_rate_delta_vs_Original=b['mean_equal_rate_delta'],V62_minus_V41_delta_difference=b['mean_equal_rate_delta']-a['mean_equal_rate_delta'] if valid else '',
            V41_better_range_fraction=a['fraction_of_common_rate_range_better'],V62_better_range_fraction=b['fraction_of_common_rate_range_better'],
            V41_status=a['status'],V62_status=b['status'],V41_common_rate_min_kbps=a['common_rate_min_kbps'],V41_common_rate_max_kbps=a['common_rate_max_kbps'],
            V62_common_rate_min_kbps=b['common_rate_min_kbps'],V62_common_rate_max_kbps=b['common_rate_max_kbps'],
            direct_V62_vs_V41_equal_rate_delta=direct['mean_equal_rate_delta'],direct_status=direct['status'],
            difference_definition='Difference of separate deltas versus Original on each comparison own measured overlap; direct comparison reported separately'))
    write(out/'v62_vs_v41_improvement.csv',improvement)
    if dataset=='uvg':
        write(ROOT/'uvg_per_sequence_comparison.csv',perseq);sequence_equal=[]
        for v in sources('uvg'):
            for anchor,method in PAIRS:
                aa=[r for r in perseq if r['method']==anchor and r['video_index']==v['video_index']]
                cc=[r for r in perseq if r['method']==method and r['video_index']==v['video_index']]
                for m in ('LPIPS','DISTS','FloLPIPS'):
                    e,_=compare_rows(aa,cc,m,anchor,method);sequence_equal.append(dict(dataset='uvg',sequence=v['name'],**e))
        write(ROOT/'uvg_per_sequence_equal_rate.csv',sequence_equal);write(out/'per_sequence_equal_rate.csv',sequence_equal)
    manifest=[]
    for r in rows:
        if r['visualization_dir']:manifest.append({k:r[k] for k in ('dataset','method','sequence','external_qp','frames','visualization_dir','visualization_sha256')})
    dump(ROOT/'visualizations'/f'{dataset}_manifest.json',dict(status='PASS',records=manifest))
    outputs={p.name:sha(p) for p in out.glob('*.csv')}
    dump(out/'report_integrity.json',dict(status='PASS',dataset=dataset,points=len(rows),summary_rows=len(summary),equal_rate_rows=len(equal),bd_rate_rows=len(bd),
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),files=outputs,completed_unix=time.time()))
    consolidate();print('REPORT PASS',dataset,flush=True)

def consolidate():
    rows=[]
    for d in DATASETS:
        if (ROOT/'results'/d/'report_integrity.json').exists():rows.extend(read(ROOT/'results'/d/'v62_vs_v41_improvement.csv'))
    if rows:write(ROOT/'v62_vs_v41_improvement.csv',rows)
    a=ROOT/'results/virat720/all_qp_summary.csv';b=ROOT/'results/virat480/all_qp_summary.csv'
    if all((ROOT/'results'/d/'report_integrity.json').exists() for d in ('virat720','virat480')):
        aa=read(a);bb=read(b);output=[]
        for method in METHODS:
            for q in range(10):
                row=dict(method=method,QP=q,rate_accounting_fps=20.0)
                for name,table in (('720p',aa),('480p',bb)):
                    r=next(x for x in table if x['method']==method and int(x['QP'])==q);assert float(r['rate_accounting_fps'])==20.0
                    row.update({name+'_'+k:r[k] for k in ('kbps','bpp',*METRICS)})
                output.append(row)
        write(ROOT/'virat_resolution_comparison.csv',output)

def finalize():
    cfg=check_frozen();allrows=[]
    for d in DATASETS:
        out=ROOT/'results'/d;r=load(out/'report_integrity.json');assert r['status']=='PASS'
        for f,h in r['files'].items():assert sha(out/f)==h
        allrows.extend(collect(d))
    assert len(allrows)==cfg['expected_points']==930
    assert len({r['compression_hash_before'] for r in allrows})==1
    for v in load(ROOT/'source_manifest.json')['videos']:
        hashes=[];h=hashlib.sha256()
        for a in arrays(v):
            data=a.tobytes();hashes.append(hashlib.sha256(data).hexdigest());h.update(data)
        assert hashes==v['frame_rgb_sha256'] and h.hexdigest()==v['rgb_sha256']
    for path,h in load(ROOT/'source_manifest.json')['origin_hashes'].items():assert sha(path)==h
    before=load(ROOT/'old_inventory_before.json');after=old_inventory()
    changed={p:dict(before=v,after=after.get(p)) for p,v in before.items() if after.get(p)!=v};added=sorted(set(after)-set(before))
    dump(ROOT/'old_experiments_untouched_audit.json',dict(status='PASS' if not changed and not added else 'FAIL',changed=changed,added=added,
        method='All old artifact sizes and nanosecond mtimes; consumed sources and checkpoints also SHA256-verified'))
    assert not changed and not added,'Old experiment inventory changed'
    consolidate()
    files=['frozen_candidate_audit.json','baseline_reuse_audit.json','virat_rate_accounting_audit.json','source_manifest.json','qp_semantics_audit.json',
        'v62_vs_v41_improvement.csv','uvg_per_sequence_comparison.csv','uvg_per_sequence_equal_rate.csv','virat_resolution_comparison.csv']
    for f in files:assert (ROOT/f).exists()
    dump(ROOT/'final_integrity.json',dict(status='PASS',completed_unix=time.time(),expected_points=930,completed_points=len(allrows),
        candidate_SHA256_correct=True,legacy_SHA256_correct=True,no_retraining=True,no_final_test_based_selection=True,
        ulong_frozen_cohort_exact=True,uvg_canonical_cohort_exact=True,virat_frozen_cohort_exact=True,
        virat720_resolution=[1280,720],virat480_resolution=[854,480],virat_temporal_sequences_unchanged=True,
        virat_rate_accounting_fps=20.0,virat_fps_applies_both_resolutions=True,no_virat_resampling=True,
        external_QP0_9_complete=True,formal_inference_QP_semantics=True,force_zero_thres=.12,
        shared_compression_hash=allrows[0]['compression_hash_before'],real_RANS_complete=True,independent_decode_complete=True,
        all_metrics_finite=True,source_checkpoint_hashes_recorded=True,common_rate_range_only=True,extrapolation=False,metric_smoothing=False,
        invalid_BD_rates_retained_with_reasons=True,old_experiments_untouched=True,
        artifacts={f:sha(ROOT/f) for f in files}))
    print('FINAL INTEGRITY PASS',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS);p.add_argument('--finalize',action='store_true');a=p.parse_args()
    if a.dataset:report_dataset(a.dataset)
    if a.finalize:finalize()
if __name__=='__main__':main()
