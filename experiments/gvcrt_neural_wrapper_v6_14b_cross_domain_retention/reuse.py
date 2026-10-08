from retention_io import *
def main():
    from adapter import validate
    import numpy as np
    frozen();good=[];missing=[]
    for d in DATASETS:
        for m in METHODS[:2]:
            for v in sources(d):
                for q in range(10):
                    # Re-encode the baseline smoke points, not reuse them.
                    if m=='v62_initial' and v['video_index']==0 and q in (0,9):continue
                    src=oldpoint(d,m,v['video_index'],q);out=point(d,m,v['video_index'],q)
                    try:
                        r=load(src);assert r['method']==('v62' if m=='v62_initial' else m)
                        r.update(method=m,reused=True,reused_from=str(src),reused_sha256=sha(src),original_method=r['method'])
                        validate(r,v,q,m)
                        with np.load(r['feature_path']) as f:
                            assert f['real'].shape==f['reconstruction'].shape==(64,2048)
                            assert np.isfinite(f['real']).all() and np.isfinite(f['reconstruction']).all()
                        if not out.exists():dump(out,r)
                        good.append(dict(target=str(out),source=str(src),source_sha256=sha(src),source_RGB_sha256=v['rgb_sha256'],checkpoint_sha256=r['checkpoint_sha256'],bitstream_sha256=r['bitstream_sha256'],features_sha256=r['feature_sha256'],matching_frames_models_codec_metrics=True))
                    except (AssertionError,KeyError,FileNotFoundError,ValueError) as e:missing.append(dict(source=str(src),target=str(out),reason=repr(e)))
    dump(ROOT/'audits/baseline_reuse.json',dict(reused=len(good),records=good,recompute=missing));print('BASELINE REUSE',len(good),'RECOMPUTE',len(missing),flush=True)
if __name__=='__main__':main()
