"""Validate completeness and export raw measurements and side-by-side tables."""
import math,statistics
from v68_io import *

def main():
    cfg=frozen(True);raw=[];fid=[];bootstrap=[];macro=[];heldout=[]
    source_counts={d:len(sources(d)) for d in DATASETS}
    for dataset in DATASETS:
        dataset_rows=[]
        for method in METHODS:
            for video in sources(dataset):
                own=[]
                for qp in range(10):
                    record=load(point(dataset,method,video['video_index'],qp))
                    validate_point(record,video,qp,method);own.append(record)
                dataset_rows.extend(own)
                write(ROOT/'results'/dataset/'per_video'/f'{method}_video_{video["video_index"]:02d}.csv',own)
            for qp in range(10):
                own=[r for r in dataset_rows if r['method']==method and r['QP']==qp]
                assert len(own)==source_counts[dataset]
                macro.append(dict(dataset=dataset,method=method,QP=qp,external_qp=qp,
                    sequences=len(own),**{k:statistics.mean(r[k] for r in own) for k in ('kbps','bpp',*METRICS)}))
                fp=ROOT/'parts/fid'/dataset/method/f'qp{qp}.json';record=load(fp)
                assert record['status']=='PASS' and math.isfinite(record['FID'])
                assert sha(record['bootstrap_path'])==record['bootstrap_sha256']
                for pt in record['source_points']:assert sha(pt['path'])==pt['sha256']
                assert record['total_real_bytes']==sum(r['real_bytes'] for r in own)
                assert record['num_frames']==sum(r['frames'] for r in own)
                draws=read(record['bootstrap_path']);assert len(draws)==20
                assert all(math.isfinite(float(row['FID'])) for row in draws)
                fid.append(record);bootstrap.extend(draws)
        raw.extend(dataset_rows);write(ROOT/'results'/dataset/'raw_rd.csv',dataset_rows)
    assert len(raw)==2790 and len(fid)==360 and len(bootstrap)==7200
    write(ROOT/'results/raw_rd.csv',raw)
    write(ROOT/'results/ablation_raw_rd.csv',[r for r in raw if r['method'] in FRESH])
    write(ROOT/'results/baseline_raw_rd.csv',[r for r in raw if r['method'] in REUSED])
    write(ROOT/'results/dataset_macro_rd.csv',macro)
    write(ROOT/'results/fid_raw.csv',fid);write(ROOT/'results/fid_bootstrap.csv',bootstrap)
    write(ROOT/'results/fid_bootstrap_summary.csv',[dict(dataset=r['dataset'],method=r['method'],QP=r['QP'],**r['bootstrap']) for r in fid])
    raw_index={(r['dataset'],r['method'],r['video_index'],r['QP']):r for r in raw}
    fid_index={(r['dataset'],r['method'],r['QP']):r for r in fid}
    macro_index={(r['dataset'],r['method'],r['QP']):r for r in macro}
    comparisons=[];fid_comparisons=[];macro_comparisons=[]
    def paired(candidate,baseline,fields):
        assert candidate['QP']==baseline['QP'] and candidate['dataset']==baseline['dataset']
        result=dict(dataset=candidate['dataset'],QP=candidate['QP'],
                    method=candidate['method'],baseline_method=baseline['method'],
                    matching='same external QP and source protocol; raw values only')
        for key in fields:
            result['baseline_'+key]=baseline[key];result['ablation_'+key]=candidate[key]
        return result
    for candidate in [r for r in raw if r['method'] in FRESH]:
        method='G'+candidate['method'][1:]
        baseline=raw_index[candidate['dataset'],method,candidate['video_index'],candidate['QP']]
        assert baseline['source_rgb_sha256']==candidate['source_rgb_sha256']
        assert baseline['actual_qps']==candidate['actual_qps']
        row=paired(candidate,baseline,('real_bytes','kbps','bpp',*METRICS,'FID'))
        row.update(sequence=candidate['sequence'],video_index=candidate['video_index'],
                   baseline_checkpoint_sha256=baseline['checkpoint_sha256'],
                   ablation_checkpoint_sha256=candidate['checkpoint_sha256'])
        comparisons.append(row)
    for dataset in DATASETS:
        for method in FRESH:
            baseline='G'+method[1:]
            for qp in range(10):
                fid_comparisons.append(paired(fid_index[dataset,method,qp],fid_index[dataset,baseline,qp],
                    ('total_real_bytes','dataset_kbps','dataset_bpp','num_frames','FID')))
                macro_comparisons.append(paired(macro_index[dataset,method,qp],macro_index[dataset,baseline,qp],
                    ('kbps','bpp',*METRICS)))
    write(ROOT/'results/comparison_per_video_same_qp.csv',comparisons)
    write(ROOT/'results/comparison_dataset_same_qp.csv',macro_comparisons)
    write(ROOT/'results/comparison_fid_same_qp.csv',fid_comparisons)
    manifest=load(ROOT/'audits/vimeo_heldout_manifest.json')
    for method in METHODS:
        for video in manifest['clips']:
            for qp in manifest['external_qps']:
                record=load(ROOT/'parts/heldout'/method/f'video_{video["index"]:03d}_qp{qp}.json')
                assert record['status']=='PASS' and record['checkpoint_sha256']==cfg['checkpoints'][method]['sha256']
                assert record['source_frame_rgb_sha256']==video['frame_rgb_sha256']
                assert record['compression_hash_before']==record['compression_hash_after']==cfg['compression_hash']
                assert record['real_RANS'] and record['independent_decode_pass'] and record['state_sync_pass']
                assert record['real_bytes']==record['bytes_consumed']
                for key in ('bitstream','feature','frame_metrics','transitions'):
                    assert sha(record[key+'_path'])==record[key+'_sha256']
                assert all(math.isfinite(record[k]) for k in (*METRICS,'kbps','bpp'))
                heldout.append(record)
    assert len(heldout)==864
    write(ROOT/'results/vimeo_heldout_raw.csv',heldout)
    hi={(r['method'],r['video_index'],r['QP']):r for r in heldout}
    hc=[]
    for candidate in [r for r in heldout if r['method'] in FRESH]:
        baseline=hi['G'+candidate['method'][1:],candidate['video_index'],candidate['QP']]
        row=paired(candidate,baseline,('real_bytes','kbps','bpp',*METRICS))
        row.update(sequence=candidate['sequence'],video_index=candidate['video_index']);hc.append(row)
    write(ROOT/'results/comparison_vimeo_heldout_same_qp.csv',hc)
    for branch in BRANCHES:
        rows=[json.loads(line) for line in (ROOT/'training_logs'/f'{branch}.jsonl').read_text().splitlines()]
        assert [r['absolute_step'] for r in rows]==list(range(1001,1501))
        assert not any('alignment_loss' in key or 'interface_cosine' in key for row in rows for key in row)
        assert load(ROOT/'branches'/branch/'training_status.json')['status']=='PASS'
        for step in (1250,1500):
            meta=load(ROOT/'branches'/branch/'checkpoint_hashes.json')[str(step)]
            assert sha(meta['path'])==meta['sha256'] and meta['lambda_align']==0
        write(ROOT/'results'/f'training_progression_{tag(branch)}.csv',rows)
    frozen(True)
    dump(ROOT/'final_integrity.json',dict(status='PASS',training_updates_per_branch=500,
        checkpoints_per_branch=[1250,1500],alignment_loss_present=False,
        raw_RD_points=len(raw),fresh_RD_points=1240,baseline_RD_points=1550,
        FID_points=len(fid),fresh_FID_points=160,FID_bootstrap_points=len(bootstrap),
        heldout_points=len(heldout),fresh_heldout_points=384,
        per_video_comparison_rows=len(comparisons),dataset_comparison_rows=len(macro_comparisons),
        FID_comparison_rows=len(fid_comparisons),heldout_comparison_rows=len(hc),
        all_required_artifacts_validated=True,compression_core_frozen=True,
        baseline_files_unchanged=True,research_interpretation_written=False,finished_unix=time.time()))
    print('RAW EXPORT PASS',flush=True)

if __name__=='__main__':main()
