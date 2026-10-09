"""Raw outputs and four-model common-rate numerical comparisons only."""
import math,statistics,traceback
from retention_io import *
PAIRS=(('v62_initial','original'),('uvg_adapt_1000','v62_initial'),('uvg_adapt_1000','original'),('uvg_adapt_5000','v62_initial'),('uvg_adapt_5000','original'),('uvg_adapt_5000','uvg_adapt_1000'))
def comparisons(dataset,sequence,metric,rows,pooled=False):
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    common=dict(dataset=dataset,sequence=sequence,metric=metric,rate_axis='dataset mean actual bpp' if pooled else 'per-video actual bpp',interpolation='PCHIP on 100 uniform ln(bpp) points; no extrapolation',delta_definition='method-reference; negative better',aggregation='pooled dataset features' if pooled else 'per-video grid mean',expected_models=4)
    curves={};gridrows=[]
    try:
        for m in METHODS:
            rs=[r for r in rows if r['method']==m];assert len(rs)==10
            assert sorted(r['QP'] for r in rs)==list(range(10))
            points={}
            for r in rs:
                x=float(r['bpp']);y=float(r[metric]);assert x>0 and np.isfinite(x) and np.isfinite(y)
                if x in points and points[x]!=y:raise ValueError('conflicting metric values at identical bpp')
                points[x]=y
            if len(points)<2:raise ValueError('less than two distinct rates')
            x=sorted(points);curves[m]=(np.log(x),np.array([points[k] for k in x]))
        lo=max(x[0] for x,y in curves.values());hi=min(x[-1] for x,y in curves.values())
        if hi<=lo:raise ValueError('no positive-length four-model common interval')
        grid=np.linspace(lo,hi,100);vals={m:PchipInterpolator(x,y,extrapolate=False)(grid) for m,(x,y) in curves.items()}
        assert all(np.isfinite(y).all() for y in vals.values())
        for i,x in enumerate(grid):gridrows.append(dict(**common,grid_index=i,ln_bpp=float(x),bpp=float(np.exp(x)),**{m:float(vals[m][i]) for m in METHODS}))
        output=[dict(**common,method=m,reference=b,status='available',available_models=4,grid_points=100,common_min_bpp=float(np.exp(lo)),common_max_bpp=float(np.exp(hi)),method_mean=float(vals[m].mean()),reference_mean=float(vals[b].mean()),delta=float((vals[m]-vals[b]).mean()),better_grid_fraction=float((vals[m]<vals[b]).mean()),equal_grid_fraction=float((vals[m]==vals[b]).mean()),reason='') for m,b in PAIRS]
    except (ValueError,AssertionError) as e:
        output=[dict(**common,method=m,reference=b,status='unavailable',available_models=len(curves),grid_points=0,method_mean=None,reference_mean=None,delta=None,better_grid_fraction=None,reason=str(e) or 'incomplete/nonfinite measurements') for m,b in PAIRS]
    return output,gridrows
def main():
    import numpy as np,torch
    torch.set_num_threads(2);frozen();command([PYTHON,'-B','-u',str(Path(__file__).resolve())])
    from adapter import validate
    assert load(ROOT/'audits/smoke_audit.json')['status']=='PASS'
    fid=module('retention_fid',REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py').low_rank_fid
    counts={};gt_audit={};unavailable=[];reused=0;fresh=0
    for d in DATASETS:
        vs=sources(d);out=ROOT/'results'/d;raw=[];gt=[]
        for v in vs:
            for m in METHODS:
                for q in range(10):
                    p=point(d,m,v['video_index'],q);r=load(p);validate(r,v,q,m)
                    reused+=bool(r.get('reused'));fresh+=not bool(r.get('reused'))
                    raw.append(dict(r,actual_bpp=r['bpp'],rate_axis='actual bpp',frames_evaluated=v['frames'],aggregation='per-video native measured point'))
                    prefix=out/'per_frame'/m/f'video_{v["video_index"]:02d}_qp{q}.csv';write(prefix,read(r['frame_metrics_path']))
                    write(out/'transitions'/m/prefix.name,read(r['transitions_path']))
            r=next(r for r in raw if r['method']=='original' and r['video_index']==v['video_index'] and r['QP']==0)
            with np.load(r['feature_path']) as f:gt.append(f['real'].copy())
        reference=np.concatenate(gt);assert reference.shape==(64*len(vs),2048) and np.isfinite(reference).all()
        fp=hashlib.sha256(json.dumps([(v['source_sha256'],v['source_frame_indices'],v['rgb_sha256']) for v in vs],sort_keys=True).encode()).hexdigest()
        cache=ROOT/'features'/f'GT_{d}.npy';cache.parent.mkdir(exist_ok=True)
        if cache.exists():assert np.array_equal(np.load(cache),reference)
        else:
            with cache.with_suffix('.tmp').open('wb') as f:np.save(f,reference)
            cache.with_suffix('.tmp').replace(cache)
        gt_audit[d]=dict(path=str(cache),sha256=sha(cache),frame_pool_sha256=fp,frames=len(reference),sequences=len(vs),source_feature_files=[dict(path=r['feature_path'],sha256=r['feature_sha256']) for r in raw if r['method']=='original' and r['QP']==0])
        pooled=[];macro=[]
        for m in METHODS:
            for q in range(10):
                rs=[next(r for r in raw if r['method']==m and r['QP']==q and r['video_index']==v['video_index']) for v in vs];rec=[]
                for i,r in enumerate(rs):
                    with np.load(r['feature_path']) as f:
                        assert np.allclose(f['real'],gt[i],rtol=1e-5,atol=1e-6)
                        assert f['reconstruction'].shape==(64,2048) and np.isfinite(f['reconstruction']).all();rec.append(f['reconstruction'].copy())
                value=float(fid(reference,np.concatenate(rec)));assert math.isfinite(value)
                row=dict(dataset=d,method=m,QP=q,FID=value,bpp=statistics.mean(r['bpp'] for r in rs),rate_axis='dataset mean actual bpp',samples=len(reference),sequences=len(vs),aggregation='pooled all dataset frame features; not mean per-video FID',GT_pool_sha256=fp,GT_cache_sha256=sha(cache));pooled.append(row)
                macro.append(dict(dataset=d,method=m,QP=q,**{k:statistics.mean(r[k] for r in rs) for k in ('bpp','kbps','LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM')},FID=value,sequences=len(vs),samples=len(reference),rate_axis='dataset mean actual bpp',kbps_aggregation='equal video mean of native per-video kbps',aggregation='equal video mean metrics; pooled FID'))
        write(out/'per_sequence_all_qp.csv',raw);write(out/'all_qp_summary.csv',macro);write(out/'fid_pooled.csv',pooled)
        per=[];grid=[];summary=[]
        for v in vs:
            for metric in ('LPIPS','DISTS','FloLPIPS'):
                rows,g=comparisons(d,v['name'],metric,[r for r in raw if r['video_index']==v['video_index']]);per.extend(rows);grid.extend(g)
        for m,b in PAIRS:
            for metric in ('LPIPS','DISTS','FloLPIPS'):
                group=[r for r in per if r['method']==m and r['reference']==b and r['metric']==metric];ok=[r for r in group if r['status']=='available'];complete=len(ok)==len(vs)
                summary.append(dict(dataset=d,method=m,reference=b,metric=metric,status='available' if complete else 'unavailable',expected_sequences=len(vs),available_sequences=len(ok),grid_points_per_sequence=100,method_mean=statistics.mean(r['method_mean'] for r in ok) if complete else None,reference_mean=statistics.mean(r['reference_mean'] for r in ok) if complete else None,delta=statistics.mean(r['delta'] for r in ok) if complete else None,better_grid_fraction=statistics.mean(r['better_grid_fraction'] for r in ok) if complete else None,rate_axis='per-video actual bpp',shared_interval='four-model common interval for each video; see per_sequence and common_rate_grid',aggregation='equal video weights; unavailable if any video missing',delta_definition='method-reference; negative better'))
        fidrows,g=comparisons(d,'pooled_dataset','FID',pooled,True);grid.extend(g)
        write(out/'equal_rate_per_sequence.csv',per);write(out/'equal_rate_summary.csv',summary);write(out/'equal_rate_fid_pooled.csv',fidrows)
        write(out/'common_rate_grid.csv',grid if grid else [dict(status='unavailable',reason='no common intervals')])
        unavailable.extend(r for r in [*per,*fidrows] if r['status']=='unavailable')
        counts[d]=dict(expected_RD=len(vs)*40,completed_RD=len(raw),expected_pooled_FID=40,completed_pooled_FID=len(pooled),frames_per_point=64,transitions_per_point=63,per_sequence_comparisons=len(per),summary_comparisons=len(summary),pooled_FID_comparisons=len(fidrows))
        print('TABLES PASS',d,len(raw),flush=True)
    dump(ROOT/'audits/GT_feature_pools.json',gt_audit)
    argv=[PLOT_PYTHON,'-B',str(ROOT/'plot.py')];command(argv);subprocess.run(argv,cwd=REPO,check=True)
    assert inventory()==load(ROOT/'audits/old_inventory.json'),'historical file inventory changed'
    frozen();assert all(sha(ROOT/p)==h for p,h in load(ROOT/'audits/executed_source_hashes.json').items())
    outputs={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results').rglob('*') if p.is_file()}
    assert len(list((ROOT/'results').glob('*/Rate_*.png')))==8 and fresh+reused==520
    light=[*ROOT.glob('*.py'),*ROOT.glob('*.json'),*list((ROOT/'results').rglob('*.csv')),*list((ROOT/'results').rglob('*.png'))]
    check=subprocess.run(['git','check-ignore','--stdin'],cwd=REPO,input='\n'.join(str(p.relative_to(REPO)) for p in light)+'\n',text=True,capture_output=True);assert check.returncode==1 and not check.stdout,check.stdout
    dump(ROOT/'final_integrity.json',dict(status='PASS',expected_RD=520,completed_RD=520,reused_baseline_points=reused,fresh_real_RANS_points=fresh,counts=counts,expected_RD_PNG=8,completed_RD_PNG=8,missing=[],failed=[],unavailable_comparisons=unavailable,checks=dict(real_RANS=True,independent_decode=True,state_sync=True,models_fixed=True,historical_unchanged=True,no_training=True,no_checkpoint_selection=True,source_frames_verified=True,per_dataset_GT_pools=True),source_hashes=load(ROOT/'audits/dependencies.json'),output_hashes=outputs,finished_unix=time.time()))
    print('FINAL PASS 520/520',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'final_integrity.json',dict(status='FAIL',error=traceback.format_exc()));raise
