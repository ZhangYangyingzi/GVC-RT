from io21 import *
import numpy as np
from fid_statistics import explicit_fid
from audit_statistics import historical_point

def main():
 rows=[]
 for d in ('ulong','hevc_b','mcljcv'):
  gt=[];rec=[]
  for v in sources(d):
   r=load(historical_point(d,'B1_initial',v['video_index'],5))
   with np.load(r['feature_path']) as f:gt.append(f['real']);rec.append(f['reconstruction'])
  gt=np.concatenate(gt);rec=np.concatenate(rec);selfr=explicit_fid(gt,gt);normal=explicit_fid(gt,rec);reverse=explicit_fid(rec,gt);selftol=1e-4+1e-6*(selfr['trace_real']+selfr['trace_rec']);swaptol=1e-4+1e-4*max(1.,abs(normal['FID']),abs(reverse['FID']))
  row=dict(dataset=d,samples=len(gt),GT_self_unclamped=selfr['FID'],GT_self_tolerance=selftol,swap_error=abs(normal['FID']-reverse['FID']),swap_tolerance=swaptol,status='PASS' if abs(selfr['FID'])<=selftol and abs(normal['FID']-reverse['FID'])<=swaptol else 'FAIL',GT_self_PSD=selfr)
  rows.append(row);dump(ROOT/'audits/FID_invariances_points.json',rows);print('FID INVARIANCES',d,row['status'],flush=True)
 dump(ROOT/'audits/FID_invariances_integrity.json',dict(status='PASS' if all(x['status']=='PASS' for x in rows) else 'FAIL',points=rows,pre_registered_tolerances=True,no_clamp=True,no_jitter=True))
if __name__=='__main__':main()
