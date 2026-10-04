"""Validate artifacts and save numerical equal-rate and attribution tables."""
import math,statistics,traceback
from v67_io import *
def main():
    frozen(True);ev=eval_adapter();sys.path.insert(0,str(V62B));rd=module('v67_rd_comparator',V62B/'report.py');raw=[];equal=[];rate=[];curves=[]
    pairs=[(A,B),(A,D),(A,E),(B,D),(B,E)]
    for d in DATASETS:
        ds=[]
        for v in sources(d):
            rows=[]
            for m in METHODS:
                for q in range(10):
                    r=load(point(d,m,v['video_index'],q));ev.validate(r,v,q,m)
                    if m in BRANCHES:assert not r.get('reused',False)
                    else:assert sha(r['reused_from'])==r['reused_point_sha256']
                    rows.append(r)
                    if m in REPORT_METHODS:raw.append(r);ds.append(r)
            for anchor,m in pairs:
                for metric in METRICS:
                    e,_=rd.compare_rows([r for r in rows if r['method']==anchor],[r for r in rows if r['method']==m],metric,anchor,m);e.update(dataset=d,sequence=v['name'],video_index=v['video_index']);equal.append(e)
            for r in rows:
                if r['method']=='original':continue
                ref=next(x for x in rows if x['method']=='original' and x['external_qp']==r['external_qp'])
                rate.append({**{k:r[k] for k in ('dataset','sequence','video_index','method','external_qp','real_bytes','bpp','kbps')},'original_real_bytes':ref['real_bytes'],'same_QP_byte_saving_fraction_vs_original':1-r['real_bytes']/ref['real_bytes'],'accounting':'same QP rate only; equal-rate quality is reported separately'})
        write(ROOT/'results'/d/'raw_rd.csv',ds)
        for m in REPORT_METHODS:
            for q in range(10):
                selected=[r for r in ds if r['method']==m and r['external_qp']==q]
                curves.append(dict(dataset=d,method=m,QP=q,N=len(selected),kbps=statistics.mean(r['kbps'] for r in selected),**{k:statistics.mean(r[k] for r in selected) for k in METRICS}))
    assert len(raw)==1240 and len({r['compression_hash_before'] for r in raw})==1
    summary=[]
    for d in DATASETS:
        for anchor,m in pairs:
            for metric in METRICS:
                rows=[r for r in equal if r['dataset']==d and r['anchor']==anchor and r['method']==m and r['metric']==metric];valid=[r for r in rows if r['status']=='valid']
                summary.append(dict(dataset=d,anchor=anchor,method=m,metric=metric,metric_direction='lower_is_better' if metric in ('LPIPS','DISTS','FloLPIPS') else 'higher_is_better',total_N=len(rows),valid_N=len(valid),invalid_N=len(rows)-len(valid),mean_equal_rate_delta=statistics.mean(r['mean_equal_rate_delta'] for r in valid) if valid else None,fraction_of_common_rate_range_better=statistics.mean(r['fraction_of_common_rate_range_better'] for r in valid) if valid else None,status='valid' if len(valid)==len(rows) else 'partial' if valid else 'invalid',aggregation='macro average of per-sequence common measured log-rate ranges; no extrapolation'))
    write(ROOT/'results/raw_rd.csv',raw);write(ROOT/'results/rate_behavior.csv',rate);write(ROOT/'results/equal_rate_summary.csv',summary);write(ROOT/'results/equal_rate_per_sequence.csv',equal);write(ROOT/'results/dataset_macro_rd.csv',curves)
    for d in DATASETS:
        write(ROOT/'results'/f'{d}_per_sequence_equal_rate.csv',[r for r in equal if r['dataset']==d]);write(ROOT/'results'/d/'equal_rate_summary.csv',[r for r in summary if r['dataset']==d])
    proxy=[];pervideo=[];attribution=[]
    keys=('L1','LPIPS','DISTS','SSIM','MS_SSIM','edge_L1','edge_relative_energy_change','HF_ratio','Laplacian_ratio')
    for d in ('uvg','ulong'):
        for v in sources(d):
            for m in REPORT_METHODS:
                p=load(ROOT/'parts/proxy'/f'{d}_{v["video_index"]:02d}_{m}.json');assert p['status']=='PASS' and sha(p['csv'])==p['csv_sha256'] and p['source_rgb_sha256']==v['rgb_sha256'];assert p['checkpoint_sha256']==load(ROOT/'checkpoint_hashes.json')[m]['sha256']
                rows=read(p['csv']);assert len(rows)==64;proxy.extend(rows)
                for k in keys:
                    vals=[float(r[k]) for r in rows if r[k]!=''];assert all(math.isfinite(x) for x in vals)
                    pervideo.append(dict(dataset=d,sequence=v['name'],video_index=v['video_index'],method=m,metric=k,mean=statistics.mean(vals) if vals else None,valid_frames=len(vals)))
            p=load(ROOT/'parts'/f'attribution_done_{d}_{v["video_index"]}.json');assert p['status']=='PASS' and sha(p['csv'])==p['sha256'];attribution.extend(read(p['csv']))
    assert len(proxy)==3840 and len(attribution)==810
    ps=[]
    for d in ('uvg','ulong'):
        for m in REPORT_METHODS:
            for k in keys:
                vals=[r['mean'] for r in pervideo if r['dataset']==d and r['method']==m and r['metric']==k and r['mean'] is not None]
                ps.append(dict(dataset=d,method=m,metric=k,mean=statistics.mean(vals) if vals else None,median=statistics.median(vals) if vals else None,std=statistics.pstdev(vals) if vals else None,valid_videos=len(vals),aggregation='per-video mean then dataset macro statistics'))
    write(ROOT/'results/proxy_behavior.csv',proxy);write(ROOT/'results/proxy_behavior_per_sequence.csv',pervideo);write(ROOT/'results/proxy_behavior_summary.csv',ps);write(ROOT/'results/recoverability_attribution_BDE.csv',attribution)
    # Read canonical frames again; validate every RGB hash without touching old files.
    io=module('v67_integrity_frames',V62B/'v62b_io.py')
    for v in sources():
        h=hashlib.sha256();n=0
        for n,arr in enumerate(io.arrays(v),1):
            data=arr.tobytes();assert hashlib.sha256(data).hexdigest()==v['frame_rgb_sha256'][n-1];h.update(data)
        assert n==v['frames'] and h.hexdigest()==v['rgb_sha256']
    assert load(ROOT/'audits/training_integrity.json')['status']=='PASS'
    assert load(ROOT/'audits/training_sequence_equivalence.json')['formal_logs_pass']
    assert load(ROOT/'audits/guard_gradient_calibration.json')['status']=='PASS'
    assert load(ROOT/'audits/controlled_difference_audit.json')['status']=='PASS'
    for b in BRANCHES:assert load(ROOT/'audits'/f'parameter_update_audit_{b[0]}.json')['status']=='PASS' and load(ROOT/'branches'/b/'smoke_audit.json')['status']=='PASS'
    frozen(True)
    mechanism=[]
    for dataset in ('uvg','ulong'):
        for method in REPORT_METHODS:
            row=dict(dataset=dataset,method=method)
            for metric in ('L1','DISTS','MS_SSIM','edge_relative_energy_change','HF_ratio','Laplacian_ratio'):
                selected=next(r for r in ps if r['dataset']==dataset and r['method']==method and r['metric']==metric)
                row['proxy_'+metric]=selected['mean']
            row['mean_same_QP_saving_vs_original']=statistics.mean(r['same_QP_byte_saving_fraction_vs_original'] for r in rate if r['dataset']==dataset and r['method']==method)
            for metric in METRICS:
                if method==A:
                    row[metric+'_equal_rate_delta_vs_A']=0.0
                    row[metric+'_equal_rate_status']='identity_reference'
                else:
                    r=next(r for r in summary if r['dataset']==dataset and r['method']==method and r['anchor']==A and r['metric']==metric)
                    row[metric+'_equal_rate_delta_vs_A']=r['mean_equal_rate_delta'];row[metric+'_equal_rate_status']=r['status']
            mechanism.append(row)
    write(ROOT/'results/guard_mechanism_summary.csv',mechanism)
    for r in attribution:
        assert all(math.isfinite(float(r[k])) for k in ('P_effect_originalBG','BG_effect_learnedP','Full_gap'))
    audits=[]
    for dataset in ('uvg','ulong'):
        for v in sources(dataset):audits.extend(read(ROOT/'audits'/f'attribution_bitstream_{dataset}_{v["video_index"]}.csv'))
    assert len(audits)==135 and all(r['status']=='PASS' for r in audits)
    write(ROOT/'audits/attribution_bitstream_equality.csv',audits)
    dump(ROOT/'final_integrity.json',dict(status='PASS',no_old_experiments_modified=True,write_scope_audit=str(ROOT/'audits/write_scope.json'),
        A_reused_not_retrained=True,B_reused_not_retrained=True,same_source_checkpoint=True,same_training_data=True,same_sample_sequence=True,same_qp_sequence=True,same_total_updates=True,same_compression_core=True,
        guard_gradient_budget_matched=True,D_only_extra_proxy_L1=True,E_only_extra_output_MS_SSIM=True,real_RANS=True,independent_decode=True,
        all_expected_evaluations_complete=True,all_metrics_finite=True,raw_rd_points=1240,fresh_full_points=620,reused_AB_points=620,additional_original_rate_reference_points=310,
        attribution_P_only_points=135,attribution_fresh_P_only_points=90,proxy_frames=3840,interpolation='unchanged PCHIP on ln(kbps), no extrapolation',
        attribution_bitstreams_equal=True,finished_unix=time.time()))
    print('FINAL PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'final_integrity.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
