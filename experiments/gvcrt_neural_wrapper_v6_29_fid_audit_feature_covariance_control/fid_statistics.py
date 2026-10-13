"""Independent explicit-covariance PSD eigensolver; unclamped FID."""
import numpy as np
from scipy.linalg import eigh

def sqrt_psd(c):
 c=(c+c.T)*.5;w,v=eigh(c,check_finite=True,driver='evd');tol=64*np.finfo(np.float64).eps*len(c)*max(1.,float(np.abs(w).max()))
 if w.min() < -tol:raise ArithmeticError(f'PSD violation {w.min()} < {-tol}')
 positive=np.maximum(w,0.)
 return (v*np.sqrt(positive))@v.T,dict(min_eigenvalue=float(w.min()),rounding_tolerance=tol,numerical_rank=int((w>tol).sum()),truncated_count=int((w<0).sum()),truncated_mass=float(-w[w<0].sum()))
def stats(x):
 x=np.asarray(x,np.float64);m=x.mean(0);z=x-m;return m,z.T@z/(len(x)-1)
def explicit_fid(a,b):
 ma,ca=stats(a);mb,cb=stats(b);sa,ra=sqrt_psd(ca);_,rb=sqrt_psd(cb)
 # Eigenvalues of the symmetric product define the Bures covariance term.
 prod=sa@cb@sa;sp,rp=sqrt_psd(prod)
 dm=float(np.square(ma-mb).sum());dc=float(np.trace(ca)+np.trace(cb)-2*np.trace(sp));fid=dm+dc
 return dict(FID=fid,D_mu=dm,D_cov=dc,trace_real=float(np.trace(ca)),trace_rec=float(np.trace(cb)),PSD_real=ra,PSD_rec=rb,PSD_product=rp)
def lowrank_unclamped(a,b):
 a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);za=a-a.mean(0);zb=b-b.mean(0)
 return float(np.square(a.mean(0)-b.mean(0)).sum()+np.square(za).sum()/(len(a)-1)+np.square(zb).sum()/(len(b)-1)-2*np.linalg.svd(za@zb.T/np.sqrt((len(a)-1)*(len(b)-1)),compute_uv=False).sum())
