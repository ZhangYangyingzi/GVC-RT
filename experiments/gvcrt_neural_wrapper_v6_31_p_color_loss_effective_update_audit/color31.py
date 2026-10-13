"""FloatRGB diagnostics and fixed fit/eval oracle transformation."""
import torch
import torch.nn.functional as F
def color(x,gt):
 assert torch.isfinite(x).all() and torch.isfinite(gt).all()
 a=x.double();b=gt.double();diff=a-b;rows={}
 for j,n in enumerate(('R','G','B')):
  rows.update({n+'_signed_mean_error':float(diff[:,j].mean()),n+'_MAE':float(diff[:,j].abs().mean()),n+'_std':float(a[:,j].std(unbiased=False)),n+'_error_std':float(diff[:,j].std(unbiased=False)),n+'_mean':float(a[:,j].mean())})
 def ycc(z):
  y=.2126*z[:,0]+.7152*z[:,1]+.0722*z[:,2]
  return y,(z[:,2]-y)/(2*(1-.0722)),(z[:,0]-y)/(2*(1-.2126))
 for n,z,t in zip(('Y','Cb','Cr'),ycc(a),ycc(b)):
  rows.update({n+'_signed_mean_error':float((z-t).mean()),n+'_std':float(z.std(unbiased=False)),n+'_mean':float(z.mean())})
 def sat(z):
  hi=z.max(1).values;lo=z.min(1).values;return torch.where(hi>0,(hi-lo)/hi.clamp_min(1e-30),torch.zeros_like(hi))
 rows['saturation_mean_difference']=float((sat(a)-sat(b)).mean())
 rows['lowfrequency_large_scale_L1']=float((F.interpolate(x,size=(36,64),mode='area')-F.interpolate(gt,size=(36,64),mode='area')).abs().mean())
 return rows
def residual(wrapper,x,center=False):
 r=wrapper.output(wrapper.blocks(wrapper.input(x)));u=x+r
 p=u.clamp(0,1)
 if not center:assert torch.equal(p,wrapper(x))
 else:p=(u-r.mean((2,3),keepdim=True)).clamp(0,1)
 stats={}
 for j,n in enumerate(('R','G','B')):
  stats.update({n+'_residual_mean':float(r[:,j].mean()),n+'_preclamp_below0':float((u[:,j]<0).float().mean()),n+'_preclamp_above1':float((u[:,j]>1).float().mean()),n+'_clamped0':float((p[:,j]==0).float().mean()),n+'_clamped1':float((p[:,j]==1).float().mean()),n+'_actual_output_mean_difference':float((p[:,j]-x[:,j]).mean())})
 return p,stats
def fit_stats(x,gt):
 a=F.interpolate(x,size=(36,64),mode='area').double().flatten(2)
 b=F.interpolate(gt,size=(36,64),mode='area').double().flatten(2)
 return torch.stack((a.sum((0,2)),b.sum((0,2)),(a*a).sum((0,2)),(a*b).sum((0,2)))),float(a.shape[0]*a.shape[2])
def fit_finish(s,n,bias):
 mx,my,xx,xy=s/n;var=xx-mx*mx;cov=xy-mx*my;a_raw=cov/var.clamp_min(1e-8);a=a_raw.clamp(.5,2);b_raw=my-a*mx;b=b_raw.clamp(-.25,.25)
 return dict(BIAS=dict(a=[1.,1.,1.],b=bias.tolist()),AFFINE=dict(a=a.tolist(),b=b.tolist(),raw_a=a_raw.tolist(),raw_b=b_raw.tolist(),variance=var.tolist(),covariance=cov.tolist(),a_boundary=(a!=a_raw).tolist(),b_boundary=(b!=b_raw).tolist()))
def transform(x,typ,params):
 if typ=='RAW':return x
 p=params[typ];a=torch.tensor(p['a'],device=x.device,dtype=x.dtype).view(1,3,1,1);b=torch.tensor(p['b'],device=x.device,dtype=x.dtype).view(1,3,1,1)
 return (x*a+b).clamp(0,1)
