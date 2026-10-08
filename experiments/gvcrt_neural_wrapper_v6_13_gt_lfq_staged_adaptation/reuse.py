"""Reuse only hash-verified inputs, deployment, metrics and real-stream artifacts."""
from v68_io import *
from io_utils import command
def main():
    command([PYTHON,'-B',str(Path(__file__).resolve())]);cfg=frozen()
    assert load(V612/'final_integrity.json')['status']=='PASS'
    assert sha(ROOT/'source_manifest.json')==sha(V612/'source_manifest.json')
    assert sha(ROOT/'qp_semantics_audit.json')==sha(V612/'qp_semantics_audit.json')
    for name in ('GT_feature_cache.json','fid_protocol_audit.json',*[f'fid_bootstrap_indices_{d}.json' for d in DATASETS]):
        assert sha(ROOT/'audits'/name)==sha(V612/'audits'/name)
    # Every historical engine/metric source remains byte-identical. These checks
    # include the metric backbones' runtime, preprocessing, flow, FID and codec.
    historic=load(V62B/'frozen_source_hashes.json');verified={};mismatch=[]
    for p,h in historic.items():
        if Path(p).suffix=='.py' or Path(p).name in ('GVC-RT_I.pt','GVC-RT_P.pt'):
            if Path(p).exists() and sha(p)==h:verified[p]=h
            else:mismatch.append(p)
    dump(ROOT/'audits/historical_metric_source_check.json',dict(status='PASS' if not mismatch else 'RECOMPUTE',verified=verified,mismatch=mismatch))
    reused=[];missing=[]
    for method,folder in [('original',V612),('v62',V62B)]:
        old_sources=load(folder/'source_manifest.json')['videos']
        for d in DATASETS:
            for v in sources(d):
                old=next(x for x in old_sources if x['dataset']==d and x['video_index']==v['video_index'])
                for q in range(10):
                    target=point(d,method,v['video_index'],q);src=folder/'parts'/d/method/target.name
                    try:
                        assert not mismatch,('historical implementation mismatch',mismatch)
                        for key in ('source_sha256','rgb_sha256','source_frame_indices','frames','width','height','rate_accounting_fps'):assert old[key]==v[key],key
                        r=load(src);assert r['evaluator_sha256']==sha(V62B/'evaluate.py')
                        assert r['source_frame_indices']==v['source_frame_indices']
                        r.update(reused=True,reused_from=str(src),reused_point_sha256=sha(src),historical_deployment_unchanged=True)
                        validate_point(r,v,q,method);dump(target,r)
                        reused.append(dict(path=str(target),source=str(src),sha256=sha(src)))
                    except (AssertionError,KeyError,FileNotFoundError) as e:
                        missing.append(dict(path=str(target),source=str(src),reason=repr(e)))
            if method!='original':continue
            for q in range(10):
                src=folder/'parts/fid'/d/method/f'qp{q}.json';dst=ROOT/'parts/fid'/d/method/src.name
                try:
                    assert not mismatch
                    r=load(src);assert r['status']=='PASS' and r['method']==method
                    assert r['GT_cache_sha256']==load(ROOT/'audits/GT_feature_cache.json')['datasets'][d]['sha256']
                    assert sha(r['bootstrap_path'])==r['bootstrap_sha256']
                    points=[load(point(d,method,v['video_index'],q)) for v in sources(d)]
                    assert len(points)==len(r['source_points'])
                    for new,pt in zip(points,r['source_points']):
                        assert sha(pt['path'])==pt['sha256'];old=load(pt['path'])
                        assert all(new[k]==old[k] for k in ('feature_sha256','bitstream_sha256','real_bytes','kbps','source_rgb_sha256','checkpoint_sha256'))
                    assert sum(x['real_bytes'] for x in points)==r['real_bytes']
                    draws=read(r['bootstrap_path']);assert len(draws)==20
                    bp=dst.with_suffix('.bootstrap.csv');write(bp,draws)
                    r.update(reused=True,reused_from=str(src),reused_point_sha256=sha(src),bootstrap_reused_from=r['bootstrap_path'],bootstrap_path=str(bp),bootstrap_sha256=sha(bp))
                    dump(dst,r)
                except (AssertionError,KeyError,FileNotFoundError) as e:missing.append(dict(path=str(dst),source=str(src),reason=repr(e)))
            if all((ROOT/'parts/fid'/d/method/f'qp{q}.json').exists() for q in range(10)):
                dump(ROOT/'parts'/f'fid_done_{d}_{method}.json',dict(status='PASS',reused=True))
    dump(ROOT/'audits/baseline_reuse_audit.json',dict(status='PASS',verified_points=reused,missing_to_recompute=missing,training_logs_reused=False))
    print('REUSE',len(reused),'RD points; recompute',len(missing),flush=True)
if __name__=='__main__':main()
