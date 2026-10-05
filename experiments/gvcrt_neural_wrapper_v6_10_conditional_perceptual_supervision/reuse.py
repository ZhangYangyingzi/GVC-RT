"""Reuse only matching Original/B1000 measurements and validate referenced files."""
from v68_io import *
def main():
    assert (ROOT/'audits/evaluation_protocol_resolution.json').exists(),'Evaluation frame-pool choice required'
    cfg=frozen();reused=[];missing=[]
    for method in REUSED:
        for dataset in DATASETS:
            for v in sources(dataset):
                for qp in range(10):
                    target=point(dataset,method,v['video_index'],qp)
                    if target.exists():validate_point(load(target),v,qp,method);continue
                    errors=[]
                    for source in (V69,V67,V68A,V62B):
                        p=source/'parts'/dataset/method/target.name
                        if not p.exists():continue
                        try:
                            record=load(p);validate_point(record,v,qp,method)
                            record.update(reused=True,reused_from=str(p),reused_point_sha256=sha(p))
                            dump(target,record);reused.append(dict(path=str(target),source=str(p),source_sha256=sha(p)));break
                        except (AssertionError,KeyError,FileNotFoundError) as e:errors.append(dict(path=str(p),reason=str(e)))
                    else:missing.append(dict(dataset=dataset,method=method,video=v['video_index'],QP=qp,failed_candidates=errors))
            for qp in range(10):
                target=ROOT/'parts/fid'/dataset/method/f'qp{qp}.json'
                if target.exists():continue
                for source in (V69,V67,V68A):
                    p=source/'parts/fid'/dataset/method/target.name
                    if not p.exists():continue
                    try:
                        record=load(p);assert record['status']=='PASS'
                        assert record['GT_cache_sha256']==load(ROOT/'audits/GT_feature_cache.json')['datasets'][dataset]['sha256']
                        assert sha(record['bootstrap_path'])==record['bootstrap_sha256']
                        assert load(source/'audits'/f'fid_bootstrap_indices_{dataset}.json')==load(ROOT/'audits'/f'fid_bootstrap_indices_{dataset}.json')
                        for pt in record['source_points']:assert sha(pt['path'])==pt['sha256']
                        rs=[load(point(dataset,method,v['video_index'],qp)) for v in sources(dataset)]
                        assert len(rs)==len(record['source_points'])
                        for r,pt in zip(rs,record['source_points']):
                            old=load(pt['path'])
                            assert all(r[k]==old[k] for k in ('feature_sha256','bitstream_sha256','real_bytes','kbps','source_rgb_sha256','checkpoint_sha256'))
                        assert sum(r['real_bytes'] for r in rs)==record['real_bytes']
                        dump(target,record);break
                    except (AssertionError,KeyError,FileNotFoundError):continue
            if all((ROOT/'parts/fid'/dataset/method/f'qp{q}.json').exists() for q in range(10)):
                dump(ROOT/'parts'/f'fid_done_{dataset}_{method}.json',dict(status='PASS',reused=True))
        manifest=load(ROOT/'audits/vimeo_heldout_manifest.json')
        for v in manifest['clips']:
            for qp in manifest['external_qps']:
                target=ROOT/'parts/heldout'/method/f'video_{v["index"]:03d}_qp{qp}.json'
                if target.exists():continue
                for source in (V69,V67,V68A):
                    p=source/'parts/heldout'/method/target.name
                    if not p.exists():continue
                    try:
                        r=load(p)
                        assert r['source_frame_rgb_sha256']==v['frame_rgb_sha256'] and r['frames']==7
                        assert r['checkpoint_sha256']==('' if method=='original' else cfg['checkpoints'][method]['sha256'])
                        assert r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass'] and r['bytes_consumed']==r['real_bytes']
                        for k in ('bitstream','feature','frame_metrics','transitions'):assert sha(r[k+'_path'])==r[k+'_sha256']
                        dump(target,r);break
                    except (AssertionError,KeyError,FileNotFoundError):continue
        if len(list((ROOT/'parts/heldout'/method).glob('*.json')))==96:
            dump(ROOT/'parts'/f'heldout_done_{method}.json',dict(status='PASS',reused=True))
    dump(ROOT/'audits/baseline_reuse.json',dict(status='PASS',verified_reused_points=reused,points_to_evaluate=missing))
    print('BASELINE REUSE',len(reused),'new verified RD points; missing',len(missing),flush=True)
if __name__=='__main__':main()
