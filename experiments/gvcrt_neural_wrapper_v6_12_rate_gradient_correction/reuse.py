"""Verified, provenance-preserving method mapping; never touch source files."""
from v68_io import *
def main():
    cfg=frozen();assert load(V611/'final_integrity.json')['status']=='PASS'
    assert sha(ROOT/'source_manifest.json')==sha(V611/'source_manifest.json')
    assert sha(ROOT/'qp_semantics_audit.json')==sha(V611/'qp_semantics_audit.json')
    for name in ('GT_feature_cache.json','fid_protocol_audit.json','vimeo_heldout_manifest.json',*[f'fid_bootstrap_indices_{d}.json' for d in DATASETS]):
        assert sha(ROOT/'audits'/name)==sha(V611/'audits'/name)
    mapping={'original':'original','B1000':'B1000'}
    mapping={m:m for m in REUSED}
    reuse_f=False
    reused=[];missing=[]
    def provenance(p,old,new):return dict(reused=True,reused_from=str(p),reused_point_sha256=sha(p),method_mapping={old:new})
    for method,old in mapping.items():
        for dataset in DATASETS:
            for v in sources(dataset):
                for qp in range(10):
                    target=point(dataset,method,v['video_index'],qp);p=V611/'parts'/dataset/old/target.name
                    try:
                        r=load(p);assert r['method']==old;r['method']=method;r.update(provenance(p,old,method))
                        validate_point(r,v,qp,method);dump(target,r)
                        reused.append(dict(path=str(target),source=str(p),source_sha256=sha(p),mapping={old:method}))
                    except (AssertionError,KeyError,FileNotFoundError) as e:missing.append(dict(path=str(target),source=str(p),reason=repr(e)))
            for qp in range(10):
                target=ROOT/'parts/fid'/dataset/method/f'qp{qp}.json';p=V611/'parts/fid'/dataset/old/target.name
                try:
                    r=load(p);assert r['status']=='PASS' and r['method']==old
                    assert r['GT_cache_sha256']==load(ROOT/'audits/GT_feature_cache.json')['datasets'][dataset]['sha256']
                    assert sha(r['bootstrap_path'])==r['bootstrap_sha256']
                    rs=[load(point(dataset,method,v['video_index'],qp)) for v in sources(dataset)]
                    assert len(rs)==len(r['source_points'])
                    for new,pt in zip(rs,r['source_points']):
                        assert sha(pt['path'])==pt['sha256'];src=load(pt['path'])
                        assert all(new[k]==src[k] for k in ('feature_sha256','bitstream_sha256','real_bytes','kbps','source_rgb_sha256','checkpoint_sha256'))
                    assert sum(x['real_bytes'] for x in rs)==r['real_bytes']
                    bp=r['bootstrap_path'];draws=read(bp);assert len(draws)==20
                    for x in draws:assert x['method']==old;x['method']=method
                    local=ROOT/'results/bootstrap_reuse'/dataset/method/f'qp{qp}.csv';write(local,draws)
                    r.update(method=method,bootstrap_reused_from=bp,bootstrap_original_sha256=r['bootstrap_sha256'],bootstrap_path=str(local),bootstrap_sha256=sha(local),**provenance(p,old,method));dump(target,r)
                except (AssertionError,KeyError,FileNotFoundError) as e:missing.append(dict(path=str(target),source=str(p),reason=repr(e)))
            if all((ROOT/'parts/fid'/dataset/method/f'qp{q}.json').exists() for q in range(10)):
                dump(ROOT/'parts'/f'fid_done_{dataset}_{method}.json',dict(status='PASS',reused=True))
        manifest=load(ROOT/'audits/vimeo_heldout_manifest.json')
        for v in manifest['clips']:
            for qp in manifest['external_qps']:
                target=ROOT/'parts/heldout'/method/f'video_{v["index"]:03d}_qp{qp}.json';p=V611/'parts/heldout'/old/target.name
                try:
                    r=load(p);assert r['method']==old
                    assert r['source_frame_rgb_sha256']==v['frame_rgb_sha256'] and r['frames']==7
                    assert r['checkpoint_sha256']==('' if method=='original' else cfg['checkpoints'][method]['sha256'])
                    assert r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass'] and r['bytes_consumed']==r['real_bytes']
                    assert r['compression_hash_before']==r['compression_hash_after']==cfg['core_hashes'][method]
                    for k in ('bitstream','feature','frame_metrics','transitions'):assert sha(r[k+'_path'])==r[k+'_sha256']
                    r.update(method=method,source_frame_indices=list(range(1,8)),**provenance(p,old,method));dump(target,r)
                except (AssertionError,KeyError,FileNotFoundError) as e:missing.append(dict(path=str(target),source=str(p),reason=repr(e)))
        if len(list((ROOT/'parts/heldout'/method).glob('*.json')))==96:
            dump(ROOT/'parts'/f'heldout_done_{method}.json',dict(status='PASS',reused=True))
    dump(ROOT/'audits/baseline_reuse_audit.json',dict(status='PASS',method_mapping=mapping,verified_points=reused,missing_to_recompute=missing,protocol_verified=True,training_logs_reused=False))
    print('REUSE',len(reused),'RD points; missing',len(missing),flush=True)
if __name__=='__main__':main()
