"""Diagnostic interpolation of mean vectors and covariance matrices, separately from main PCHIP."""
from io21 import *
from fid_statistics import stats,sqrt_psd
import argparse,numpy as np,statistics
from scipy.linalg import eigvalsh
from comparison import PAIRS

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=(*DATASETS,'mcljcv_exclude_overlap29'),required=True);a=ap.parse_args();d=a.dataset;source='mcljcv' if d.endswith('29') else d
 vs=[v for v in sources(source) if not(d.endswith('29') and v['name']=='videoSRC17_1920x1080_24')]
 out=ROOT/'results/covariance_diagnostic'/d;raw=[];arrays={};files=[]
 real=[]
 for v in vs:
  r=load(point(source,'original',v['video_index'],0))
  with np.load(r['feature_path']) as f:real.append(f['real'])
 real=np.concatenate(real).astype(np.float64);mr,cr=stats(real);sr,audit=sqrt_psd(cr)
 for m in METHODS:
  arrays[m]=[]
  for q in range(10):
   rs=[load(point(source,m,v['video_index'],q)) for v in vs];rec=[]
   for r in rs:
    assert sha(r['feature_path'])==r['feature_sha256']
    with np.load(r['feature_path']) as f:rec.append(f['reconstruction'])
   rec=np.concatenate(rec).astype(np.float64);mu,cov=stats(rec);bpp=statistics.mean(r['bpp'] for r in rs);dm=float(np.square(mu-mr).sum());z0=real-mr;z1=rec-mu
   singular=np.linalg.svd(z0@z1.T/(len(real)-1),compute_uv=False);dc=float(np.trace(cr)+np.trace(cov)-2*singular.sum());fid=dm+dc
   raw.append(dict(dataset=d,method=m,QP=q,bpp=bpp,samples=len(real),videos=len(vs),D_mu=dm,D_cov=dc,FID=fid,closure_error=abs(fid-dm-dc),feature_sources=[dict(path=r['feature_path'],sha256=r['feature_sha256']) for r in rs],ddof=1))
   arrays[m].append((bpp,mu,cov,q))
  arrays[m].sort(key=lambda x:x[0])
 write(out/'fid_decomposition_raw.csv',raw)
 lo=max(xs[0][0] for xs in arrays.values());hi=min(xs[-1][0] for xs in arrays.values());assert lo<hi
 grid=np.linspace(np.log(lo),np.log(hi),100);rows=[]
 for m,xs in arrays.items():
  xx=np.log([x[0] for x in xs]);assert np.all(np.diff(xx)>0)
  for i,x in enumerate(grid):
   j=max(0,min(len(xs)-2,int(np.searchsorted(xx,x,side='right')-1)));w=float((x-xx[j])/(xx[j+1]-xx[j]));assert -1e-12<=w<=1+1e-12
   mu=(1-w)*xs[j][1]+w*xs[j+1][1];cov=(1-w)*xs[j][2]+w*xs[j+1][2]
   prod=sr@cov@sr;ev=eigvalsh((prod+prod.T)*.5,check_finite=False,driver='evd');tol=64*np.finfo(np.float64).eps*len(ev)*max(1.,float(np.abs(ev).max()));assert ev.min()>=-tol
   dm=float(np.square(mu-mr).sum());dc=float(np.trace(cr)+np.trace(cov)-2*np.sqrt(np.maximum(ev,0.)).sum());fid=dm+dc
   rows.append(dict(dataset=d,method=m,grid_index=i,ln_bpp=float(x),bpp=float(np.exp(x)),D_mu=dm,D_cov=dc,FID=fid,closure_error=abs(fid-dm-dc),left_QP=xs[j][3],right_QP=xs[j+1][3],linear_weight=w,PSD_product_min=float(ev.min()),PSD_roundoff_tolerance=tol,interpolation='diagnostic: same linear ln(bpp) weights on mean VECTOR and covariance MATRIX; NOT main PCHIP'))
   if i%20==0:print('COV DIAGNOSTIC',d,m,i,flush=True)
 write(out/'fid_decomposition_equal_rate.csv',rows)
 diffs=[]
 for m,ref in PAIRS:
  for i in range(100):
   mm=next(r for r in rows if r['method']==m and r['grid_index']==i);rr=next(r for r in rows if r['method']==ref and r['grid_index']==i)
   diffs.append(dict(dataset=d,method=m,reference=ref,grid_index=i,bpp=mm['bpp'],delta_D_mu=mm['D_mu']-rr['D_mu'],delta_D_cov=mm['D_cov']-rr['D_cov'],delta_FID=mm['FID']-rr['FID'],closure_error=abs((mm['D_mu']-rr['D_mu'])+(mm['D_cov']-rr['D_cov'])-(mm['FID']-rr['FID'])),interpolation='diagnostic mean-vector and covariance-matrix linear interpolation'))
 write(out/'component_deltas.csv',diffs)
 bands=[]
 for m,ref in PAIRS:
  for j,label in enumerate(('low','mid','high')):
   lower=float(grid[0]+(grid[-1]-grid[0])*j/3);upper=float(grid[0]+(grid[-1]-grid[0])*(j+1)/3)
   indices=[i for i,x in enumerate(grid) if x>=lower and (x<upper if j<2 else x<=upper)]
   rr=[x for x in diffs if x['method']==m and x['reference']==ref and x['grid_index'] in indices]
   bands.append(dict(dataset=d,method=m,reference=ref,band=label,lower_ln_bpp=lower,upper_ln_bpp=upper,grid_points=len(rr),**{k:statistics.mean(x[k] for x in rr) for k in ('delta_D_mu','delta_D_cov','delta_FID')},interpolation='diagnostic linear mean/covariance matrices, not main PCHIP'))
 write(out/'fid_component_rate_bands.csv',bands)
 dump(out/'integrity.json',dict(status='PASS',samples=len(real),videos=len(vs),raw_points=40,diagnostic_grid_points=400,lower_bpp=lo,upper_bpp=hi,no_extrapolation=True,mean_and_covariance_same_linear_weights=True,not_main_PCHIP=True,GT_covariance_PSD=audit))
if __name__=='__main__':main()
