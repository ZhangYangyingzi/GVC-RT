"""Exact existing dataset-level FID plus fixed paired sequence-stratified bootstrap."""
import argparse,math,statistics
from v67b_io import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    import numpy as np
    frozen();cfg=load(ROOT/'config.json');fid=fid_function();cache=load(ROOT/'audits/GT_feature_cache.json')['datasets'][a.dataset];assert sha(cache['path'])==cache['sha256']
    gt=np.load(cache['path']);assert array_hash(gt)==cache['array_sha256'];draws=load(ROOT/'audits'/f'fid_bootstrap_indices_{a.dataset}.json')['indices'];assert len(draws)==20
    for q in ([0] if a.smoke else range(10)):
        out=ROOT/'parts/fid'/a.dataset/a.method/f'qp{q}.json'
        if out.exists():
            r=load(out);assert r['status']=='PASS' and sha(r['bootstrap_path'])==r['bootstrap_sha256'];continue
        start=time.time();features=[];records=[];offset=0;max_diff=0.
        for v in sources(a.dataset):
            r=load(point(a.dataset,a.method,v['video_index'],q));validate_point(r,v,q,a.method);records.append(r)
            with np.load(r['feature_path']) as f:
                real=f['real'];rec=f['reconstruction'];expected=gt[offset:offset+v['frames']]
                assert real.shape==rec.shape==expected.shape==(v['frames'],2048) and np.isfinite(real).all() and np.isfinite(rec).all()
                assert np.allclose(real,expected,rtol=1e-5,atol=1e-6),'Reference feature mismatch'
                max_diff=max(max_diff,float(np.abs(real-expected).max()));features.append(rec.copy())
            offset+=v['frames']
        reconstructed=np.concatenate(features);assert len(gt)==len(reconstructed)==offset
        value=fid(gt,reconstructed);assert math.isfinite(value)
        bits=sum(r['total_bits'] for r in records);duration=sum(v['frames']/v['rate_accounting_fps'] for v in sources(a.dataset));pixels=sum(v['frames']*v['width']*v['height'] for v in sources(a.dataset))
        bootstrap=[]
        for i,indices in enumerate(draws):
            ix=np.asarray(indices,dtype=np.int64);assert len(ix)==len(gt);fv=fid(gt[ix],reconstructed[ix]);assert math.isfinite(fv)
            bootstrap.append(dict(dataset=a.dataset,method=a.method,QP=q,repeat=i,seed=load(ROOT/'audits'/f'fid_bootstrap_indices_{a.dataset}.json')['seed'],num_frames=len(ix),FID=fv,GT_cache_sha256=cache['sha256']))
        bpath=out.with_suffix('.bootstrap.csv');write(bpath,export_rows(bootstrap))
        vals=[r['FID'] for r in bootstrap];summary=dict(mean=statistics.mean(vals),std=statistics.pstdev(vals),median=statistics.median(vals),P05=float(np.quantile(vals,.05)),P95=float(np.quantile(vals,.95)),repeats=20)
        result=dict(status='PASS',dataset=a.dataset,method=a.method,QP=q,external_qp=q,total_real_bytes=bits//8,real_bytes=bits//8,total_bits=bits,total_duration_seconds=duration,dataset_kbps=bits/duration/1000,kbps=bits/duration/1000,dataset_bpp=bits/pixels,bpp=bits/pixels,macro_average_kbps=statistics.mean(r['kbps'] for r in records),num_frames=len(gt),FID=value,GT_cache_sha256=cache['sha256'],GT_array_sha256=cache['array_sha256'],reconstruction_feature_pool_sha256=array_hash(reconstructed),GT_max_abs_feature_difference=max_diff,bootstrap=summary,bootstrap_path=str(bpath),bootstrap_sha256=sha(bpath),source_points=[dict(path=str(point(a.dataset,a.method,r['video_index'],q)),sha256=sha(point(a.dataset,a.method,r['video_index'],q))) for r in records],elapsed_seconds=time.time()-start)
        dump(out,result);print('FID',a.dataset,a.method,q,'PASS seconds',round(time.time()-start,1),flush=True)
    if a.smoke:dump(ROOT/'audits/fid_smoke.json',dict(status='PASS',dataset=a.dataset,method=a.method,QP=0,bootstrap_repeats=20))
    else:dump(ROOT/'parts'/f'fid_done_{a.dataset}_{a.method}.json',dict(status='PASS',points=10,bootstrap_points=200))
if __name__=='__main__':main()
