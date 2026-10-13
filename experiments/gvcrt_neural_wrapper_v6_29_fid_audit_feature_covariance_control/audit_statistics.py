from io21 import *
import numpy as np
from fid_statistics import explicit_fid,lowrank_unclamped,stats

def historical_point(d,m,i,q):
 root=V27 if d=='mcljcv' else V26
 return root/'parts/ip_-1'/d/m/f'video_{i:02d}_qp{q}.json'
def main():
 rows=[];out=ROOT/'audits/FID_statistics_points.json'
 if out.exists():rows=load(out)
 for d in ('ulong','hevc_b','mcljcv'):
  for m in ('original','B1_initial'):
   for q in (0,5,9):
    if any(x['dataset']==d and x['method']==m and x['QP']==q for x in rows):continue
    real=[];rec=[];refs=[]
    for v in sources(d):
     p=historical_point(d,m,v['video_index'],q);r=load(p);f=Path(r.get('features_path',r.get('feature_path','')))
     if not f.is_file():
      candidates=[x for x in (V27 if d=='mcljcv' else V26).glob(f'features/**/{v["name"]}/qp{q}.npz') if m in x.parts]
      assert len(candidates)==1,(d,m,v['name'],candidates);f=candidates[0]
     with np.load(f) as z:real.append(z['real']);rec.append(z['reconstruction'])
     refs.append(dict(sequence=v['name'],point_path=str(p),point_sha256=sha(p),features_path=str(f),features_sha256=sha(f),RGB_hashes=v['rgb_sha256'],indices=v['source_frame_indices']))
    a,b=np.concatenate(real),np.concatenate(rec);e=explicit_fid(a,b);old=lowrank_unclamped(a,b);tol=1e-4+1e-4*max(1.,abs(old),abs(e['FID']))
    row=dict(dataset=d,method=m,QP=q,samples=len(a),sources=refs,**e,lowrank_FID_unclamped=old,error=abs(old-e['FID']),tolerance=tol,status='PASS' if abs(old-e['FID'])<=tol else 'FAIL')
    rows.append(row);dump(out,rows);print('FID EXPLICIT',d,m,q,row['status'],row['error'],flush=True)
 dump(ROOT/'audits/FID_statistics_integrity.json',dict(status='PASS' if len(rows)==18 and all(x['status']=='PASS' for x in rows) else 'FAIL',expected=18,completed=len(rows),points=rows,independent_symmetric_eigendecomposition=True,no_jitter=True,unclamped=True))
if __name__=='__main__':main()
