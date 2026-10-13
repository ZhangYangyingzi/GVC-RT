"""CPU-only pooled statistics and dedicated three-QP diagnostics."""
from audit_io import *
import argparse, fcntl, math, resource
import numpy as np
from scipy.linalg import eigh
from scipy.interpolate import PchipInterpolator
from fid_statistics import stats,sqrt_psd
from worker import features_path,gt_path
COMPARISONS=(('B_covstats_1000','A_continue_1000'),('A_continue_1000','B1_initial'),('B_covstats_1000','B1_initial'),('B_covstats_1000','original'))
def fid(a,b,cache):
    assert a.shape==b.shape and len(a)>1 and np.isfinite(a).all() and np.isfinite(b).all()
    key=hashlib.sha256(a.tobytes()).hexdigest()
    resultkey=hashlib.sha256((key+hashlib.sha256(b.tobytes()).hexdigest()+sha(Path(__file__))+sha(ROOT/'fid_statistics.py')).encode()).hexdigest()
    resultpath=ROOT/'cache/FID_results'/f'{resultkey}.json';resultpath.parent.mkdir(parents=True,exist_ok=True)
    result_lock=resultpath.with_suffix('.lock').open('a');fcntl.flock(result_lock,fcntl.LOCK_EX)
    if resultpath.exists():return load(resultpath)
    p=ROOT/'cache/statistics'/f'{key}.npz'
    if key not in cache:
        if p.exists():
            with np.load(p) as z:cache[key]=(z['mean'],z['cov'],z['sqrt'])
        else:
            # Lock shared GT statistics; no duplicate eigendecomposition by workers.
            p.parent.mkdir(parents=True,exist_ok=True);stats_lock=p.with_suffix('.lock').open('a');fcntl.flock(stats_lock,fcntl.LOCK_EX)
            if p.exists():
                with np.load(p) as z:cache[key]=(z['mean'],z['cov'],z['sqrt'])
            else:
                mu,cov=stats(a);s,_=sqrt_psd(cov);save_npz(p,mean=mu,cov=cov,sqrt=s);cache[key]=(mu,cov,s)
            stats_lock.close()
    ma,ca,sa=cache[key];mb,cb=stats(b);prod=sa@cb@sa;prod=(prod+prod.T)*.5
    w=eigh(prod,eigvals_only=True,driver='evd',check_finite=True)
    tol=64*np.finfo(np.float64).eps*2048*max(1.,float(np.abs(w).max()));assert w.min()>=-tol
    mu=float(np.square(ma-mb).sum());cov=float(np.trace(ca)+np.trace(cb)-2*np.sqrt(np.maximum(w,0)).sum())
    value=mu+cov;assert math.isfinite(value)
    result=dict(FID=value,D_mu=mu,D_cov=cov,PSD_rounding_tolerance=tol,PSD_product_min=float(w.min()),samples=len(a),ddof=1,arithmetic='float64',clamped=False)
    dump(resultpath,result);return result
def grid_compare(curves, comparisons, pool, metric='FID'):
    # Requires exactly QP 0/5/9, not the ten-QP primary validator.
    valid_curves={}
    for m,rs in curves.items():
        assert sorted(r['QP'] for r in rs)==[0,5,9]
        rs=sorted(rs,key=lambda r:r['bpp']);x=np.array([r['bpp'] for r in rs]);y=np.array([r[metric] for r in rs])
        assert np.isfinite(x).all() and np.isfinite(y).all() and (x>0).all() and (np.diff(x)>0).all()
        valid_curves[m]=(x,y)
    lo=max(x[0] for x,y in valid_curves.values());hi=min(x[-1] for x,y in valid_curves.values());rows=[];grids=[]
    for m,ref in comparisons:
        base=dict(pool=pool,metric=metric,method=m,reference=ref,delta_definition='method-reference',diagnostic='3-QP stability diagnostic; not v6.29 primary ten-QP comparison',rate_axis='dataset mean actual bpp' if metric=='FID' else 'per-video actual bpp',low_bpp=float(lo),high_bpp=float(hi),grid_count=100)
        if not hi>lo:
            rows.append(dict(**base,status='unavailable',reason='no common measured bpp interval'));continue
        grid=np.linspace(np.log(lo),np.log(hi),100)
        def interp(k):
            x,y=valid_curves[k];return PchipInterpolator(np.log(x),y,extrapolate=False)(grid)
        a,b=interp(m),interp(ref);assert np.isfinite(a).all() and np.isfinite(b).all()
        am,bm=float(a.mean()),float(b.mean());delta=am-bm
        rows.append(dict(**base,status='available',method_mean=am,reference_mean=bm,delta=delta,relative_percent=100*delta/bm if bm else None,better_grid_fraction=float((a<b).mean())))
        grids.extend(dict(pool=pool,metric=metric,method=m,reference=ref,index=i,bpp=float(np.exp(z)),method_value=float(a[i]),reference_value=float(b[i]),delta=float(a[i]-b[i])) for i,z in enumerate(grid))
    return rows,grids
def pool_arrays(d,ids,m,q,kind):
    real=[];rec=[];rates=[];inputs=[]
    for i in ids:
        oldr=load(point(d,m,i,q))
        t=dict(kind='closure',dataset=d,video=i,method=m,qp=q)
        if kind=='cached':p=Path(oldr['feature_path']);r=oldr
        else:p=features_path(t);r=oldr
        with np.load(p) as z:real.append(z['real'].copy());rec.append(z['reconstruction'].copy())
        bytes_=r['real_bytes'];frames=64
        inputs.append(dict(path=str(p),sha256=sha(p)))
        if kind in ('new64','combined_two_clips'):
            nt=dict(kind='new',dataset=d,video=i,method=m,qp=q);nr=load(output(nt))['result'];p=features_path(nt)
            with np.load(p) as z:na=z['real'].copy();nb=z['reconstruction'].copy()
            if kind=='new64':real[-1]=na;rec[-1]=nb;bytes_=nr['real_bytes']
            else:real[-1]=np.concatenate((real[-1],na));rec[-1]=np.concatenate((rec[-1],nb));bytes_+=nr['real_bytes'];frames=128
            inputs.append(dict(path=str(p),sha256=sha(p)))
        rates.append(bytes_*8/(1920*1080*frames))
    return np.concatenate(real),np.concatenate(rec),float(np.mean(rates)),inputs
def main(t):
    lf=(ROOT/'tasks'/(task_id(t)+'.lock')).open('a');fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if valid(t):return
    spec=t['spec'];d=t['dataset'];ids=spec['videos'];label=spec['label'];kind=spec['kind'];cache={};pooled=[];checked=[];comparisons=[];grids=[];inputs=[]
    methods=METHODS if kind!='subset' else ('A_continue_1000','B_covstats_1000');ar_kind='old64' if kind in ('subset','closure_pool','matched_old64') else kind
    canonical=None
    for q in QPS:
        for m in methods:
            a,b,rate,refs=pool_arrays(d,ids,m,q,ar_kind);inputs.extend(refs)
            digest=hashlib.sha256(a.tobytes()).hexdigest()
            if canonical is None:canonical=digest
            assert canonical==digest, 'GT members or ordering differ across models/QP'
            f=fid(a,b,cache);row=dict(pool=label,dataset=d,method=m,QP=q,bpp=rate,sequences=len(ids),GT_features_sha256=digest,aggregation='pooled features; equal-video mean actual bpp',**f);pooled.append(row)
            if kind=='closure_pool':
                ca,cb,cr,crefs=pool_arrays(d,ids,m,q,'cached');inputs.extend(crefs);cached=fid(ca,cb,cache)['FID'];err=abs(f['FID']-cached);tol=1e-4+1e-4*max(1,abs(cached),abs(f['FID']))
                table=OLD/'results/ip_-1'/d/'fid_pooled.csv' if label!='mcl29_old64' else OLD/'results/covariance_diagnostic/mcljcv_exclude_overlap29/fid_decomposition_raw.csv'
                with table.open() as file:history=next(r for r in csv.DictReader(file) if r['method']==m and int(r['QP'])==q)
                historical=float(history['FID']);historical_error=abs(f['FID']-historical);htol=1e-4+1e-4*max(1,abs(historical),abs(f['FID']))
                checked.append(dict(dataset=d,pool=label,method=m,QP=q,fresh_FID=f['FID'],cached_FID=cached,historical_FID=historical,historical_table_path=str(table),historical_table_sha256=sha(table),error=err,tolerance=tol,historical_error=historical_error,historical_tolerance=htol,PASS=err<=tol and historical_error<=htol,samples=len(a),fresh_GT_sha256=digest,cached_GT_sha256=hashlib.sha256(ca.tobytes()).hexdigest()))
    curves={m:[r for r in pooled if r['method']==m] for m in methods};cs=(COMPARISONS[0],) if kind=='subset' else COMPARISONS
    c,g=grid_compare(curves,cs,label);comparisons.extend(c);grids.extend(g)
    for r in comparisons:r.update(dataset=d,sequences=len(ids),samples=pooled[0]['samples'],video_indices=ids,aggregation='pooled feature FID; dataset mean actual bpp')
    for r in grids:r.update(dataset=d,sequences=len(ids))
    p=ROOT/'statistics'/f'{task_id(t)}.json';dump(p,dict(pooled=pooled,closure=checked,comparisons=comparisons,grids=grids,input_features=inputs,videos=ids,pool=label,kind=kind))
    artifacts={str(p):sha(p)}
    dump(output(t),dict(task=t,signature=signature(t),execution_complete=True,status='PASS' if all(x['PASS'] for x in checked) else 'FAIL',artifacts=artifacts,statistics_path=str(p),peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    print('STATISTICS DONE',label,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);a=p.parse_args()
    main(json.loads(a.task))
