"""Matched frame-pool statistics, no linear attribution of covariance FID."""
from io21 import *
import numpy as np
from scipy.linalg import eigvalsh
from fid_statistics import stats,lowrank_unclamped,explicit_fid
from audit_statistics import historical_point

def components(a,b):
 mu=float(np.square(a.mean(0)-b.mean(0)).sum());fid=lowrank_unclamped(a,b);return dict(FID=fid,D_mu=mu,D_cov=fid-mu,closure_error=0.)
def main():
 toy=np.array([[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]],dtype=np.float64);scaled=toy*2+np.array([1.,-2.]);t=explicit_fid(toy,scaled);expected=5.+4./3;assert abs(t['FID']-expected)<1e-10
 selfr=explicit_fid(toy,toy);sw=explicit_fid(scaled,toy);assert abs(t['FID']-sw['FID'])<1e-10 and abs(selfr['FID'])<1e-10
 dump(ROOT/'audits/FID_analytic_tests.json',dict(status='PASS',toy_expected=expected,toy_computed=t['FID'],GT_self_unclamped=selfr['FID'],input_swap_error=abs(t['FID']-sw['FID'])))
 per=[];group=[];lovo=[];decomp=[];spectra=[];closure=[]
 for d in ('ulong','hevc_b','mcljcv'):
  for q in (0,5,9):
   arrays={};gt=[]
   for m in ('original','B1_initial'):
    arrays[m]=[]
    for v in sources(d):
     r=load(historical_point(d,m,v['video_index'],q))
     with np.load(r['feature_path']) as f:
      a=f['real'].astype(np.float64);b=f['reconstruction'].astype(np.float64)
     arrays[m].append(b)
     if m=='original':gt.append(a)
     else:assert np.array_equal(a,gt[len(arrays[m])-1])
     ma,ca=stats(a);mb,cb=stats(b);err=cb-ca;diagonal=np.diag(err);off=err-np.diag(diagonal)
     per.append(dict(dataset=d,QP=q,sequence=v['name'],method=m,samples=len(a),**components(a,b),mean_shift_L2=float(np.linalg.norm(ma-mb)),covariance_trace_GT=float(np.trace(ca)),covariance_trace_rec=float(np.trace(cb)),diagonal_squared_error=float(np.square(diagonal).sum()),off_diagonal_squared_error=float(np.square(off).sum())))
   real=np.concatenate(gt)
   for m in arrays:
    pooled=np.concatenate(arrays[m]);ma,ca=stats(real);mb,cb=stats(pooled);N=len(pooled)
    within=np.zeros_like(cb);between=np.zeros_like(cb)
    for b in arrays[m]:
     mv,cv=stats(b);within+=(len(b)-1)*cv;dv=mv-mb;between+=len(b)*np.outer(dv,dv)
    e=float(np.abs((within+between)/(N-1)-cb).max());assert e<1e-10
    closure.append(dict(dataset=d,QP=q,method=m,matrix_max_abs_error=e,total_trace=float(np.trace(cb)),within_trace=float(np.trace(within)/(N-1)),between_trace=float(np.trace(between)/(N-1)),status='PASS',warning='Covariance additivity does not imply additive covariance FID distances'))
    eig=eigvalsh(cb,check_finite=False);spectra.extend(dict(dataset=d,QP=q,method=m,index=i,eigenvalue=float(x)) for i,x in enumerate(eig))
    dd=cb-ca;diag=np.diag(dd)
    decomp.append(dict(dataset=d,QP=q,method=m,samples=N,videos=len(gt),**components(real,pooled),covariance_trace=float(np.trace(cb)),GT_covariance_trace=float(np.trace(ca)),diagonal_squared_error=float(np.square(diag).sum()),off_diagonal_squared_error=float(np.square(dd).sum()-np.square(diag).sum())))
    for label,lo,hi in (('I0',0,1),('P1_16',1,17),('P17_32',17,33),('P33_48',33,49),('P49_63',49,64)):
     aa=np.concatenate([a[lo:hi] for a in gt]);bb=np.concatenate([b[lo:hi] for b in arrays[m]])
     group.append(dict(dataset=d,QP=q,method=m,group=label,samples=len(aa),**components(aa,bb),comparison='same-QP within identical group only; not equal-rate temporal drift'))
    for j,v in enumerate(sources(d)):
     aa=np.concatenate([a for i,a in enumerate(gt) if i!=j]);bb=np.concatenate([a for i,a in enumerate(arrays[m]) if i!=j]);lovo.append(dict(dataset=d,QP=q,method=m,excluded_sequence=v['name'],samples=len(aa),**components(aa,bb),comparison='same external QP diagnostic; not equal-rate'))
   # Native I is identical under fixed input/QP across the two models.
   for original,b1 in zip(arrays['original'],arrays['B1_initial']):assert np.array_equal(original[0],b1[0])
   for name,rows in [('fid_decomposition_raw',decomp),('per_video_statistics',per),('frame_group_statistics',group),('leave_one_video_out',lovo),('covariance_spectrum',spectra),('within_between_closure',closure)]:write(ROOT/'results/FID_audit'/f'{name}.csv',rows)
   print('GAP AUDIT',d,q,flush=True)
 dump(ROOT/'audits/FID_gap_integrity.json',dict(status='PASS',pooled_points=len(decomp),same_GT_UID=True,I0_features_identical=True,within_between_matrix_closure=True,frame_groups_fixed=True,leave_one_video_out=True,per_video_FID_not_averaged=True,first_appearance_stage='尚未定位首次出现阶段',reason='Only protocol-matched original and B1 feature pools audited; intermediate compatible evidence not established.'))
if __name__=='__main__':main()
