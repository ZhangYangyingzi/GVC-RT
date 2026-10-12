"""Float64 Inception FID components; deterministic diagnostic interpolation."""
from io21 import *
import math,statistics
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
HM=('original','B1_initial','v625_A_continue_1000','v625_B_capped_projected_1000','v625_C_capped_conditional_projected_1000')
CURRENT='--current' in sys.argv
if CURRENT:HM=METHODS
def main():
 import numpy as np
 from report import comparisons,summarize
 fid=module('v626_historical_fid',REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py').low_rank_fid
 records={};out=ROOT/('results/fid_decomposition' if CURRENT else 'historical');out.mkdir(exist_ok=True)
 if (out/'fid_decomposition_raw.json').exists():records={f"{r['dataset']}/{r['method']}/{r['QP']}":r for r in load(out/'fid_decomposition_raw.json')}
 while True:
  for d in DATASETS:
   vs=sources(d);gtpath=V20/'features'/f'GT_{d}.npy';gt=np.asarray(np.load(gtpath),np.float64);poolhash=hashlib.sha256(json.dumps([(v['source_sha256'],v['source_frame_indices'],v['rgb_sha256']) for v in vs],sort_keys=True).encode()).hexdigest()
   for m in HM:
    for q in range(10):
     key=f'{d}/{m}/{q}'
     if key in records:continue
     ps=[point(d,m,v['video_index'],q) for v in vs]
     if not all(p.exists() for p in ps):continue
     from adapter import validate
     rs=[load(p) for p in ps]
     for r,v in zip(rs,vs):validate(r,v,q,m)
     feats=[];src=[]
     for r in rs:
      f=Path(r['feature_path']);assert sha(f)==r['feature_sha256']
      with np.load(f) as z:feats.append(z['reconstruction'].astype(np.float64))
      src.append(dict(path=str(f),sha256=sha(f)))
     rec=np.concatenate(feats);assert rec.shape==gt.shape and np.isfinite(rec).all();a,b=gt-gt.mean(0),rec-rec.mean(0)
     mu=float(np.square(gt.mean(0)-rec.mean(0)).sum());cov=float(np.square(a).sum()/(len(a)-1)+np.square(b).sum()/(len(b)-1)-2*np.linalg.svd((a@b.T)/math.sqrt((len(a)-1)*(len(b)-1)),compute_uv=False).sum());total=mu+cov;ref=float(fid(gt,rec));err=abs(total-ref);assert err<1e-7+1e-8*abs(ref)
     oldm=m.removeprefix('v625_');table=read((ROOT if CURRENT else V25)/'results/ip_-1'/d/'fid_pooled.csv');hist=float(next(r['FID'] for r in table if r['method']==oldm and int(r['QP'])==q));assert abs(ref-hist)<1e-7+1e-8*abs(hist)
     r=dict(dataset=d,method=m,QP=q,bpp=statistics.mean(r['bpp'] for r in rs),D_mu=mu,D_cov=cov,FID=total,reproduced_FID=ref,historical_FID=hist,closure_error=err,tolerance='1e-7+1e-8*abs(FID)',samples=len(gt),videos=len(vs),checkpoint_sha256=rs[0]['checkpoint_sha256'],GT_features=str(gtpath),GT_features_sha256=sha(gtpath),GT_frame_pool_sha256=poolhash,feature_sources=src,**{k:statistics.mean(r[k] for r in rs) for k in ('LPIPS','DISTS','FloLPIPS')})
     records[key]=r;dump(out/'fid_decomposition_raw.json',list(records.values()));write(out/'fid_decomposition_raw.csv',list(records.values()));dump(out/'status.json',dict(status='RUNNING',pooled_completed=len(records),pooled_expected=200));print('FID_COMPONENT',key,flush=True)
  if len(records)==200:break
  time.sleep(30)
 equal=[];bands=[];main=[];pairs=tuple((m,ref) for m in HM[2:] for ref in ('original','B1_initial'))
 for d in DATASETS:
  rows=[r for r in records.values() if r['dataset']==d];lo=max(min(r['bpp'] for r in rows if r['method']==m) for m in HM);hi=min(max(r['bpp'] for r in rows if r['method']==m) for m in HM)
  assert lo<hi;grid=np.linspace(np.log(lo),np.log(hi),100);values={}
  for m in HM:
   rr=sorted([r for r in rows if r['method']==m],key=lambda r:r['bpp']);xs=np.log([r['bpp'] for r in rr]);assert np.all(np.diff(xs)>0)
   values[m]={k:np.interp(grid,xs,[r[k] for r in rr]) for k in ('D_mu','D_cov','FID')}
   assert np.allclose(values[m]['D_mu']+values[m]['D_cov'],values[m]['FID'],atol=1e-7)
  bounds=np.linspace(grid[0],grid[-1],4)
  for m,ref in pairs:
   for i,z in enumerate(grid):equal.append(dict(dataset=d,method=m,reference=ref,grid_index=i,bpp=float(np.exp(z)),interpolation='diagnostic: shared piecewise linear ln(bpp) weights',common_low_bpp=lo,common_high_bpp=hi,**{'delta_'+k:float(values[m][k][i]-values[ref][k][i]) for k in values[m]}))
   for j,label in enumerate(('low','middle','high')):
    idx=(grid>=bounds[j]) & ((grid<bounds[j+1]) if j<2 else (grid<=bounds[j+1]));bands.append(dict(dataset=d,method=m,reference=ref,band=label,low_ln_bpp=float(bounds[j]),high_ln_bpp=float(bounds[j+1]),count=int(idx.sum()),**{'delta_'+k:float((values[m][k]-values[ref][k])[idx].mean()) for k in values[m]}))
  for metric in ('FID','LPIPS','DISTS','FloLPIPS'):
   if metric=='FID':rs,_=comparisons(d,'pooled_dataset',metric,rows,True,methods=HM,pairs=pairs);main.extend(rs)
   else:
    per=[]
    for v in sources(d):
     rr=[load(point(d,m,v['video_index'],q)) for m in HM for q in range(10)];rs,_=comparisons(d,v['name'],metric,rr,methods=HM,pairs=pairs);per.extend(rs)
    main.extend([r for r in summarize(d,per,len(sources(d)),pairs) if r['metric']==metric])
 write(out/'fid_decomposition_equal_rate.csv',equal);write(out/'fid_component_rate_bands.csv',bands);write(out/'checkpoint_progression_equal_rate.csv',main)
 subprocess.run([PLOT_PYTHON,'-B',str(ROOT/'historical_plot.py'),*(['--current'] if CURRENT else [])],check=True)
 dump(out/'final_integrity.json',dict(status='PASS',pooled_rows=200,Inception_dimensions=2048,GT_fixed=True,sample_covariance_ddof=1,float64=True,FID_closure_verified=True,shared_linear_component_interpolation=True,main_FID_PCHIP=True,no_extrapolation=True,fixed_band_boundaries=True))
 print('HISTORICAL PASS',flush=True)
def passed_visual():return (ROOT/'historical/visual_integrity.json').exists() and load(ROOT/'historical/visual_integrity.json')['status']=='PASS'
if __name__=='__main__':main()
