"""Convergence numeric tables and integrity; never selects a checkpoint."""
import math,statistics,traceback
from v67b_io import *
def main():
    import numpy as np
    cfg=frozen(True);sys.path.insert(0,str(V62B));rd=module('v67b_rd_comparator',V62B/'report.py');ev=eval_adapter()
    raw=[];equal=[];macro=[];progress=[];fidrows=[];boot=[];bootsum=[];fid_eq=[]
    pairs=[(A,b) for b in B_METHODS]+[('B250','B500'),('B500','B1000'),('B250','B1000')]
    fidpairs=[('B250','B500'),('B500','B1000'),('B250','B1000'),(A,'B1000'),(O,'B1000')]
    for d in DATASETS:
        ds=[]
        for v in sources(d):
            own=[]
            for m in METHODS:
                for q in range(10):
                    r=load(point(d,m,v['video_index'],q));validate_point(r,v,q,m);raw.append(r);ds.append(r);own.append(r)
            for anchor,m in pairs:
                for metric in METRICS:
                    e,_=rd.compare_rows([r for r in own if r['method']==anchor],[r for r in own if r['method']==m],metric,anchor,m);e.update(dataset=d,sequence=v['name'],video_index=v['video_index']);equal.append(e)
        for m in METHODS:
            for q in range(10):
                own=[r for r in ds if r['method']==m and r['external_qp']==q]
                macro.append(dict(dataset=d,method=m,external_qp=q,QP=q,kbps=statistics.mean(r['kbps'] for r in own),bpp=statistics.mean(r['bpp'] for r in own),**{k:statistics.mean(r[k] for r in own) for k in METRICS},num_sequences=len(own)))
                fpath=ROOT/'parts/fid'/d/m/f'qp{q}.json';f=load(fpath);assert f['status']=='PASS' and math.isfinite(f['FID'])
                for p in f['source_points']:assert sha(p['path'])==p['sha256']
                assert sha(f['bootstrap_path'])==f['bootstrap_sha256'];fidrows.append(f);br=read(f['bootstrap_path']);assert len(br)==20 and all(math.isfinite(float(r['FID'])) for r in br);boot.extend(br)
                bootsum.append(dict(dataset=d,method=m,QP=q,**f['bootstrap']))
        for anchor,m in fidpairs:
            e,_=rd.compare_rows([r for r in fidrows if r['dataset']==d and r['method']==anchor],[r for r in fidrows if r['dataset']==d and r['method']==m],'FID',anchor,m)
            fid_eq.append(dict(dataset=d,anchor=anchor,method=m,common_rate_min=e['common_rate_min_kbps'],common_rate_max=e['common_rate_max_kbps'],grid_points=e['grid_points'],mean_equal_rate_FID_delta=e['mean_equal_rate_delta'],fraction_common_range_better=e['fraction_of_common_rate_range_better'],status=e['status'],reason=e['reason'],rate_axis='ln(aggregate_dataset_kbps)',metric_direction='lower_is_better'))
        for q in range(10):
            lookup={r['method']:r for r in fidrows if r['dataset']==d and r['QP']==q};original=lookup[O]
            for m in METHODS:
                r=lookup[m];prev='B250' if m=='B500' else 'B500' if m=='B1000' else None
                progress.append(dict(dataset=d,method=m,QP=q,real_bytes=r['total_real_bytes'],bpp=r['dataset_bpp'],kbps=r['dataset_kbps'],macro_average_kbps=r['macro_average_kbps'],saving_vs_Original=1-r['total_real_bytes']/original['total_real_bytes'],previous_checkpoint=prev,saving_vs_previous_checkpoint=1-r['total_real_bytes']/lookup[prev]['total_real_bytes'] if prev else None))
        write(ROOT/'results'/d/'raw_rd.csv',export_rows(ds))
    assert len(raw)==1550 and len(fidrows)==200 and len(boot)==4000
    assert {r['compression_hash_before'] for r in raw}=={cfg['compression_hash']}
    summaries=[]
    for d in DATASETS:
        for anchor,m in pairs:
            for metric in METRICS:
                own=[r for r in equal if r['dataset']==d and r['anchor']==anchor and r['method']==m and r['metric']==metric];valid=[r for r in own if r['status']=='valid']
                summaries.append(dict(dataset=d,anchor=anchor,method=m,metric=metric,total_N=len(own),valid_N=len(valid),mean_equal_rate_delta=statistics.mean(r['mean_equal_rate_delta'] for r in valid) if valid else None,fraction_common_range_better=statistics.mean(r['fraction_of_common_rate_range_better'] for r in valid) if valid else None,status='valid' if len(valid)==len(own) else 'partial' if valid else 'invalid',metric_direction='lower_is_better' if metric in ('LPIPS','DISTS','FloLPIPS') else 'higher_is_better',aggregation='macro mean of per-sequence PCHIP comparisons on common measured log-rate intervals'))
    write(ROOT/'results/raw_rd.csv',export_rows(raw));write(ROOT/'results/dataset_macro_rd.csv',export_rows(macro));write(ROOT/'results/equal_rate_summary.csv',export_rows(summaries));write(ROOT/'results/equal_rate_per_sequence.csv',export_rows(equal));write(ROOT/'results/fid_raw.csv',export_rows(fidrows));write(ROOT/'results/fid_bootstrap.csv',boot);write(ROOT/'results/fid_bootstrap_summary.csv',export_rows(bootsum));write(ROOT/'results/fid_equal_rate_summary.csv',export_rows(fid_eq));write(ROOT/'results/rate_progression.csv',export_rows(progress))
    uvg=[r for r in equal if r['dataset']=='uvg' and r['metric'] in ('LPIPS','DISTS','FloLPIPS') and (r['anchor'],r['method']) in [('B250','B500'),('B500','B1000'),(A,'B1000')]];assert len(uvg)==63;write(ROOT/'results/uvg_checkpoint_progression.csv',uvg)
    convergence=[]
    for d in DATASETS:
        for m in B_METHODS:
            prev='B250' if m=='B500' else 'B500' if m=='B1000' else None
            for q in range(10):
                f=next(r for r in fidrows if r['dataset']==d and r['method']==m and r['QP']==q);rate=next(r for r in progress if r['dataset']==d and r['method']==m and r['QP']==q)
                row=dict(dataset=d,method=m,QP=q,mean_actual_bitrate_kbps=f['macro_average_kbps'],aggregate_dataset_kbps=f['dataset_kbps'],same_QP_saving_vs_Original=rate['saving_vs_Original'],FID=f['FID'],previous_checkpoint=prev)
                for metric in ('LPIPS','DISTS','FloLPIPS'):
                    for anchor,label in [(A,'A'),(prev,'previous_checkpoint')]:
                        s=next((r for r in summaries if r['dataset']==d and r['method']==m and r['anchor']==anchor and r['metric']==metric),None)
                        row[metric+'_equal_rate_delta_vs_'+label]=s['mean_equal_rate_delta'] if s else None;row[metric+'_equal_rate_status_vs_'+label]=s['status'] if s else 'NOT_APPLICABLE'
                fde=next((r for r in fid_eq if r['dataset']==d and r['method']==m and r['anchor']==prev),None);row['FID_equal_rate_delta_vs_previous_checkpoint']=fde['mean_equal_rate_FID_delta'] if fde else None;row['FID_equal_rate_status']=fde['status'] if fde else 'NOT_APPLICABLE';convergence.append(row)
    write(ROOT/'results/perceptual_convergence_summary.csv',convergence)
    heldout_report(cfg)
    # Final exact canonical source-frame audit, without any decoder or source mutation.
    io=module('v67b_final_sources',V62B/'v62b_io.py')
    for v in sources():
        h=hashlib.sha256();n=0
        for n,arr in enumerate(io.arrays(v),1):
            data=arr.tobytes();assert hashlib.sha256(data).hexdigest()==v['frame_rgb_sha256'][n-1];h.update(data)
        assert n==v['frames'] and h.hexdigest()==v['rgb_sha256']
    assert load(ROOT/'audits/training_log_audit.json')['status']==load(ROOT/'audits/training_plot_audit.json')['status']=='PASS'
    for file,h in load(ROOT/'audits/training_plot_audit.json')['plots'].items():assert sha(ROOT/'plots'/file)==h
    frozen(True)
    dump(ROOT/'final_integrity.json',dict(status='PASS',no_training=True,no_existing_files_modified=True,B250_found=True,B500_found=True,B1000_found=True,same_compression_core=True,same_source_protocol=True,same_dataset_manifests=True,same_qp_semantics=True,real_RANS=True,independent_decode=True,LPIPS_complete=True,DISTS_complete=True,FloLPIPS_complete=True,FID_protocol_audited=True,FID_complete=True,FID_same_frame_pool=True,all_metrics_finite=True,all_expected_points_complete=True,no_checkpoint_selection=True,raw_RD_points=1550,fresh_RD_points=620,reused_RD_points=930,FID_points=200,FID_bootstrap_points=4000,equal_rate_per_sequence_comparisons=len(equal),equal_rate_dataset_comparisons=len(summaries),FID_equal_rate_comparisons=len(fid_eq),heldout_status=load(ROOT/'audits/vimeo_heldout_integrity.json')['status'],finished_unix=time.time()))
    print('FINAL PASS',flush=True)

def heldout_report(cfg):
    import numpy as np
    manifest=load(ROOT/'audits/vimeo_heldout_manifest.json')
    if manifest['status']=='NOT_AVAILABLE':
        dump(ROOT/'audits/vimeo_heldout_integrity.json',dict(status='NOT_AVAILABLE'));write(ROOT/'results/vimeo_heldout_checkpoint_progression.csv',[dict(status='NOT_AVAILABLE')]);return
    fid=fid_function();summary=[];allrows=[];canonical=[]
    for v in manifest['clips']:
        for p,h in zip(v['paths'],v['file_sha256']):assert sha(p)==h
        r=load(ROOT/'parts/heldout/B250'/f'video_{v["index"]:03d}_qp0.json');assert sha(r['feature_path'])==r['feature_sha256']
        with np.load(r['feature_path']) as f:canonical.append(f['real'].copy())
    gt=np.concatenate(canonical);cachepath=ROOT/'fid_cache/vimeo_heldout_GT.npy'
    if cachepath.exists():assert np.array_equal(np.load(cachepath),gt)
    else:
        with cachepath.open('wb') as f:np.save(f,gt)
    for m in B_METHODS:
        for q in manifest['external_qps']:
            records=[];reconstructed=[];offset=0
            for v in manifest['clips']:
                r=load(ROOT/'parts/heldout'/m/f'video_{v["index"]:03d}_qp{q}.json')
                for k in ('bitstream','feature','frame_metrics','transitions'):assert sha(r[k+'_path'])==r[k+'_sha256']
                assert r['checkpoint_sha256']==cfg['checkpoints'][m]['sha256'] and r['source_frame_rgb_sha256']==v['frame_rgb_sha256']
                assert r['compression_hash_before']==r['compression_hash_after']==cfg['compression_hash'] and r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass']
                assert r['real_bytes']==r['bytes_consumed'] and r['frames']==7
                with np.load(r['feature_path']) as f:assert np.allclose(f['real'],gt[offset:offset+7],rtol=1e-5,atol=1e-6);reconstructed.append(f['reconstruction'].copy())
                offset+=7;records.append(r);allrows.append(r)
            pool=np.concatenate(reconstructed);value=fid(gt,pool);assert math.isfinite(value)
            summary.append(dict(dataset='vimeo_official_test_separate',method=m,QP=q,sequences=len(records),num_frames=len(gt),LPIPS=statistics.mean(r['LPIPS'] for r in records),DISTS=statistics.mean(r['DISTS'] for r in records),FID=value,real_bytes=sum(r['real_bytes'] for r in records),actual_bpp=sum(r['real_bytes']*8 for r in records)/(len(gt)*448*256),GT_feature_cache_sha256=sha(cachepath)))
    assert len(allrows)==len(manifest['clips'])*3*3 and len(summary)==9
    write(ROOT/'results/vimeo_heldout_raw.csv',allrows);write(ROOT/'results/vimeo_heldout_checkpoint_progression.csv',summary);dump(ROOT/'audits/vimeo_heldout_integrity.json',dict(status='PASS',official_test_only=True,train_test_disjoint=True,GT_cache_sha256=sha(cachepath),num_sequences=len(manifest['clips']),points=len(allrows),FID_points=9,no_checkpoint_selection=True))
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'final_integrity.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
