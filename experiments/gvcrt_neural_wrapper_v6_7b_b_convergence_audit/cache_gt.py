"""One SHA256-audited GT feature pool per fixed dataset."""
from v67b_io import *
def main():
    import numpy as np
    frozen();assert load(ROOT/'audits/fid_protocol_audit.json')['status']=='PASS'
    audits={}
    for d in DATASETS:
        arrays=[];rows=[]
        for v in sources(d):
            r=load(point(d,O,v['video_index'],0));validate_point(r,v,0,O)
            with np.load(r['feature_path']) as f:a=f['real'].copy()
            assert a.shape==(v['frames'],2048) and np.isfinite(a).all();arrays.append(a)
            rows.append(dict(sequence=v['name'],frames=v['frames'],source_rgb_sha256=v['rgb_sha256'],feature_path=r['feature_path'],feature_file_sha256=r['feature_sha256'],GT_array_sha256=array_hash(a)))
        pool=np.concatenate(arrays);path=ROOT/'fid_cache'/f'{d}_GT.npy'
        if path.exists():assert np.array_equal(np.load(path),pool)
        else:
            with path.with_suffix('.tmp').open('wb') as f:np.save(f,pool)
            path.with_suffix('.tmp').replace(path)
        audits[d]=dict(path=str(path),sha256=sha(path),array_sha256=array_hash(pool),frames=len(pool),sequences=rows,manifest_sha256=sha(ROOT/'source_manifest.json'))
    dump(ROOT/'audits/GT_feature_cache.json',dict(status='PASS',datasets=audits));print('GT CACHE PASS',flush=True)
if __name__=='__main__':main()
