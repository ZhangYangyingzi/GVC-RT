from io21 import *
from fid_statistics import lowrank_unclamped
from comparison import comparisons,PAIRS
import argparse,numpy as np,statistics

def diagnostic_comparisons(dataset,excluded,curves):
 """Three-QP sensitivity only; do not weaken the main ten-QP validator."""
 from scipy.interpolate import PchipInterpolator
 common=dict(dataset=dataset,sequence=excluded,metric='FID',rate_axis='remaining-dataset mean actual bpp',
             grid_points=100,expected_models=len(METHODS),measured_QPs=[0,5,9],
             interpolation='three-QP diagnostic PCHIP on100 uniform ln(bpp) points; no extrapolation',
             delta_definition='method-reference; negative better',aggregation='remaining pooled feature set')
 try:
  measured={}
  for m in METHODS:
   rs=[r for r in curves if r['method']==m];assert sorted(r['QP'] for r in rs)==[0,5,9]
   points={}
   for r in rs:
    x,y=float(r['bpp']),float(r['FID']);assert x>0 and np.isfinite(x) and np.isfinite(y)
    if x in points and points[x]!=y:raise ValueError('conflicting values at identical bpp')
    points[x]=y
   assert len(points)>=2
   xs=sorted(points);measured[m]=(np.log(xs),[points[x] for x in xs])
  lo=max(x[0] for x,y in measured.values());hi=min(x[-1] for x,y in measured.values())
  if hi<=lo:raise ValueError('no common measured interval')
  grid=np.linspace(lo,hi,100);values={m:PchipInterpolator(x,y,extrapolate=False)(grid) for m,(x,y) in measured.items()}
  assert all(np.isfinite(y).all() for y in values.values())
  return [dict(common,method=m,reference=ref,status='available',common_min_bpp=float(np.exp(lo)),common_max_bpp=float(np.exp(hi)),
               method_mean=float(values[m].mean()),reference_mean=float(values[ref].mean()),delta=float((values[m]-values[ref]).mean()),
               relative_change_percent=100*float((values[m]-values[ref]).mean())/float(values[ref].mean()) if values[ref].mean()!=0 else None,
               better_grid_fraction=float((values[m]<values[ref]).mean()),reason='') for m,ref in PAIRS]
 except (ValueError,AssertionError) as e:
  return [dict(common,method=m,reference=ref,status='unavailable',reason=str(e) or 'invalid three-QP measurements') for m,ref in PAIRS]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=(*DATASETS,'mcljcv_exclude_overlap29'),required=True);a=ap.parse_args();d=a.dataset;source='mcljcv' if d.endswith('29') else d;vs=[v for v in sources(source) if not(d.endswith('29') and v['name']=='videoSRC17_1920x1080_24')];out=ROOT/'results/new_leave_one_video_out'/d
 if len(vs)<2:
  dump(out/'integrity.json',dict(status='PASS',available=False,reason='Removing the only sequence leaves an empty pool'));return
 features={};rates={};gt=[]
 for m in METHODS:
  for q in (0,5,9):
   fs=[];rs=[]
   for v in vs:
    r=load(point(source,m,v['video_index'],q));rs.append(r['bpp'])
    with np.load(r['feature_path']) as f:
     fs.append(f['reconstruction'].astype(np.float64))
     if m=='original' and q==0:gt.append(f['real'].astype(np.float64))
   features[(m,q)]=fs;rates[(m,q)]=rs
 rows=[];compare=[]
 for excluded in range(len(vs)):
  real=np.concatenate([f for i,f in enumerate(gt) if i!=excluded]);curves=[]
  for m in METHODS:
   for q in (0,5,9):
    rec=np.concatenate([f for i,f in enumerate(features[(m,q)]) if i!=excluded]);fid=lowrank_unclamped(real,rec);bpp=statistics.mean(f for i,f in enumerate(rates[(m,q)]) if i!=excluded)
    row=dict(dataset=d,excluded_sequence=vs[excluded]['name'],method=m,QP=q,bpp=bpp,FID=fid,samples=len(real),diagnostic='three-QP sensitivity at0/5/9, not the main ten-QP benchmark');rows.append(row);curves.append(row)
  # Only diagnostic grid over these three measured QPs; never substitute for main10-QP results.
  rs=diagnostic_comparisons(d,vs[excluded]['name'],curves);compare.extend(dict(r,excluded_sequence=vs[excluded]['name'],diagnostic='three-QP leave-one-out PCHIP; not primary ten-QP curve') for r in rs)
  write(out/'raw.csv',rows);write(out/'equal_rate_diagnostic.csv',compare);print('NEW LOVO',d,excluded+1,'/',len(vs),flush=True)
 dump(out/'integrity.json',dict(status='PASS',available=True,excluded_sequences=len(vs),raw_points=len(rows),three_QP_only=[0,5,9],pooled_FID=True,not_main_curve=True))
if __name__=='__main__':main()
