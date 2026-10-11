"""Validate historical v6.22 points; rerun unverifiable records locally."""
from io23 import *
def main():
    import numpy as np
    from adapter import validate,cache_key,historical22_adapter,frames_for
    old,oldio=historical22_adapter();oldio.frozen();good=[];bad=[];frame_checks=[]
    cfg=evalcfg();oldcfg=oldio.evalcfg()
    for key in ('compression_hash','I_hash','original_receiver_hashes','metric_module_hashes'):
        assert cfg[key]==oldcfg[key]
    proto=load(ROOT/'protocol.json');oldproto=load(V22/'protocol.json')
    for key in ('frames','IPs','I_positions','I_input','QP_phase','reference_reset','full_bytes','rate_denominator','fps','metrics','equal_rate'):
        assert proto[key]==oldproto[key]
    gts=load(ROOT/'audits/inherited_GT_feature_pools.json')['pools']
    for d in DATASETS:
        assert sha(gts[d]['path'])==gts[d]['sha256'];gt=np.load(gts[d]['path'])
        for vi,v in enumerate(sources(d)):
            assert v==next(x for x in oldio.sources(d) if x['video_index']==v['video_index'])
            frames=frames_for(v);assert len(frames)==64;del frames
            frame_checks.append(dict(dataset=d,video=v['name'],RGB_verified=True,source_frame_indices=v['source_frame_indices']))
            for method in METHODS[:3]:
                if method!='original':assert cfg['checkpoints'][method]==oldcfg['checkpoints'][method] and cfg['deployment_receiver_hashes'][method]==oldcfg['deployment_receiver_hashes'][method]
                for q in range(10):
                    p=oldio.point(d,method,v['video_index'],q,-1)
                    try:
                        r=load(p);old.validate(r,v,q,method,-1)
                        assert r['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][:64]
                        with np.load(r['feature_path']) as f:
                            assert f['real'].shape==(64,2048) and np.allclose(f['real'],gt[vi*64:(vi+1)*64],atol=1e-6,rtol=1e-5)
                            assert f['reconstruction'].shape==(64,2048) and np.isfinite(f['reconstruction']).all()
                        r.update(method=method,IP=-1,reused=True,reused_from=str(p),reused_sha256=sha(p),reuse_verified=True,
                            cache_key=cache_key(v,q,method,-1),source_method=method,source_protocol_sha256=sha(V22/'protocol.json'),reuse_dependency_sha256=sha(ROOT/'audits/dependencies.json'))
                        validate(r,v,q,method,-1);dump(point(d,method,v['video_index'],q,-1),r)
                        good.append(dict(path=str(p),sha256=sha(p),method=method,dataset=d,sequence=v['name'],QP=q,
                            feature_sha256=sha(r['feature_path']),frame_metrics_sha256=sha(r['frame_metrics_path']),transitions_sha256=sha(r['transitions_path']),
                            GT_pool_sha256=gts[d]['sha256'],model_protocol_input_match=True))
                    except (AssertionError,KeyError,FileNotFoundError,ValueError) as e:bad.append(dict(path=str(p),reason=repr(e),method=method,dataset=d,sequence=v['name'],QP=q))
    dump(ROOT/'audits/baseline_reuse.json',dict(status='PASS',expected=480,reused=len(good),records=good,rerun=bad,frame_checks=frame_checks))
    print('REUSE',len(good),'/480',flush=True)
if __name__=='__main__':main()
