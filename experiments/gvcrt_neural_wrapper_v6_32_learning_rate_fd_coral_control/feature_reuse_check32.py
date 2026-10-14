"""Independent CPU-only official feature equality check; no new encoding or gradients."""
from io32 import *
def main():
 import numpy as np
 from evaluate32 import receipt,oldpoint,valid
 t=dict(stage='baseline',dataset='ulong',video=0,method='original',qp=0);assert valid(t)
 old=load(oldpoint(t));now=load(receipt(t))['result'];rows=[]
 with np.load(old['feature_path']) as a,np.load(now['feature_path']) as b:
  for key in ('real','reconstruction'):
   x=b[key].astype(np.float64);y=a[key].astype(np.float64);assert x.shape==y.shape==(64,2048)
   delta=x-y;rel=np.linalg.norm(delta,axis=1)/np.maximum(np.linalg.norm(y,axis=1),1e-12);ma=np.abs(delta).max(1);tol=1e-4+1e-4*np.abs(y).max(1)
   for i in range(64):rows.append(dict(kind=key,frame=i,relative_L2=float(rel[i]),max_abs=float(ma[i]),max_abs_tolerance=float(tol[i]),PASS=bool(rel[i]<=1e-4 and ma[i]<=tol[i])))
 assert all(r['PASS'] for r in rows)
 write(ROOT/'audits/official_feature_forward_equivalence.csv',rows)
 dump(ROOT/'audits/official_feature_forward_equivalence.json',dict(status='PASS',task=t,reference='historical floating RGB official features',previous_path=old['feature_path'],previous_sha256=sha(old['feature_path']),fresh_path=now['feature_path'],fresh_sha256=sha(now['feature_path']),V630_tolerances=dict(relative_L2=1e-4,max_abs='1e-4+1e-4*maxabs(reference)'),max_relative_L2=max(r['relative_L2'] for r in rows),max_abs=max(r['max_abs'] for r in rows),new_encodings=0,optimizer_updates=0))
if __name__=='__main__':main()
