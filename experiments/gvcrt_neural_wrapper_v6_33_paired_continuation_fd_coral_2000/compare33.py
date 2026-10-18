"""Fixed eight-method main grid and separately labelled three-QP trajectory."""
from io32 import *
import collections,math
DATASETS=('ulong','uvg_holdout','uvg_validation','hevc_b','mcljcv','mcljcv29')
MAIN=['original','B1_initial']+[f'{b}_{s}' for s in (1000,2000) for b in BRANCHES]
PAIRS=[(f'{b}_2000',f'{b}_1000') for b in BRANCHES]+[(f'{BRANCHES[1]}_2000',f'{BRANCHES[0]}_2000'),(f'{BRANCHES[2]}_2000',f'{BRANCHES[0]}_2000'),(f'{BRANCHES[1]}_2000',f'{BRANCHES[2]}_2000')]+[(f'{b}_2000',r) for b in BRANCHES for r in ('original','B1_initial')]+[(f'{b}_1000',f'{BRANCHES[0]}_1000') for b in BRANCHES[1:]]+[(f'{b}_1000',r) for b in BRANCHES for r in ('original','B1_initial')]
def run(records,pools):
 import numpy as np
 from scipy.interpolate import PchipInterpolator
 grids=[];rows=[];fallback=[];intervals=[]
 def curve(curves,d,v,metric,methods,pairs,phase,expected):
  xs={m:np.array([p['bpp'] for p in curves[m]],float) for m in methods};ys={m:np.array([p[metric] for p in curves[m]],float) for m in methods}
  complete=all(len(xs[m])==expected and np.isfinite(xs[m]).all() and np.isfinite(ys[m]).all() and np.all(xs[m]>0) and len(set(xs[m]))==expected for m in methods)
  lo=max(x.min() for x in xs.values()) if complete else None;hi=min(x.max() for x in xs.values()) if complete else None;available=complete and hi>lo
  interpolation='diagnostic linear ln(bpp); common weights for D_mu/D_cov' if metric in ('D_mu','D_cov') else 'PCHIP ln(bpp)'
  scope='eight-model common interval' if phase=='main_2000' else 'all-stage common interval; only QP0/5/9 for all methods'
  intervals.append(dict(phase=phase,dataset=d,video=v,metric=metric,status='available' if available else 'unavailable',bpp_min=lo,bpp_max=hi,methods=methods,expected_QPs=expected,interpolation=interpolation,extrapolation=False))
  def interp(m,grid):
   x=xs[m];y=ys[m];o=np.argsort(x)
   return np.interp(grid,np.log(x[o]),y[o]) if metric in ('D_mu','D_cov') else PchipInterpolator(np.log(x[o]),y[o],extrapolate=False)(grid)
  if available:
   loggrid=np.linspace(np.log(lo),np.log(hi),100);vals={m:interp(m,loggrid) for m in methods};assert all(np.isfinite(x).all() for x in vals.values())
   for i,x in enumerate(loggrid):grids.append(dict(phase=phase,dataset=d,video=v,metric=metric,index=i,rate_band=('low','mid','high')[min(i*3//99,2)],bpp=float(np.exp(x)),interpolation=interpolation,**{m:float(vals[m][i]) for m in methods}))
  for m,r in pairs:
   for band in ('all','low','mid','high'):
    base=dict(phase=phase,dataset=d,video=v,metric=metric,method=m,reference=r,rate_band=band,interval_scope=scope,bpp_min=lo,bpp_max=hi,interpolation=interpolation,relative_definition='100*(method_mean-reference_mean)/reference_mean; not BD-rate')
    if not available:rows.append(dict(base,status='unavailable',reason='no complete positive common measured interval'));continue
    mask=np.ones(100,dtype=bool) if band=='all' else np.array([('low','mid','high')[min(i*3//99,2)]==band for i in range(100)])
    a=float(vals[m][mask].mean());b=float(vals[r][mask].mean());rows.append(dict(base,status='available',method_mean=a,reference_mean=b,delta=a-b,relative_change_percent=100*(a-b)/b if b else None,better_grid_fraction=float((vals[m][mask]<vals[r][mask]).mean()),band_ln_bpp_min=float(loggrid[mask].min()),band_ln_bpp_max=float(loggrid[mask].max())))
  if phase=='main_2000':
   for b in BRANCHES[1:]:
    b2=f'{b}_2000';b1=f'{b}_1000';ref2=f'{BRANCHES[0]}_2000';ref1=f'{BRANCHES[0]}_1000'
    for band in ('all','low','mid','high'):
     mask=np.ones(100,dtype=bool) if band=='all' else np.array([('low','mid','high')[min(i*3//99,2)]==band for i in range(100)])
     if not available:
      rows.append(dict(phase=phase,dataset=d,video=v,metric=metric,method=b2,reference=ref2,rate_band=band,contrast='difference_of_differences',status='unavailable',reason='eight-model common interval unavailable',interval_scope=scope,bpp_min=lo,bpp_max=hi));continue
     oldgap=vals[b1]-vals[ref1];newgap=vals[b2]-vals[ref2]
     rows.append(dict(phase=phase,dataset=d,video=v,metric=metric,method=b2,reference=ref2,rate_band=band,contrast='difference_of_differences',formula='(branch2000-B2000)-(branch1000-B1000)',status='available',old_gap_mean=float(oldgap[mask].mean()),new_gap_mean=float(newgap[mask].mean()),delta=float((newgap[mask]-oldgap[mask]).mean()),interval_scope=scope,bpp_min=lo,bpp_max=hi,interpolation=interpolation,relative_change_percent=None,relative_definition='raw metric contrast; no percentage denominator or BD-rate'))
  if phase=='main_2000' and not available:
   for m,r in pairs:
    if len(xs[m])!=10 or len(xs[r])!=10:continue
    a=max(xs[m].min(),xs[r].min());b=min(xs[m].max(),xs[r].max());row=dict(dataset=d,video=v,metric=metric,method=m,reference=r,interval_scope='PAIRWISE ONLY; not eight-model common mean',bpp_min=float(a),bpp_max=float(b))
    if b<=a:row.update(status='unavailable',reason='no pairwise common interval')
    else:
     g=np.linspace(np.log(a),np.log(b),100);va=interp(m,g);vb=interp(r,g);ma=float(va.mean());mb=float(vb.mean());row.update(status='available',method_mean=ma,reference_mean=mb,delta=ma-mb,relative_change_percent=100*(ma-mb)/mb if mb else None)
    fallback.append(row)
 def points(d):
  vs=load(ROOT/'eval_manifest.json')['mcljcv' if d=='mcljcv29' else d]
  return [v for v in vs if not(d=='mcljcv29' and 'videoSRC17' in v['name'])]
 for phase in ('main_2000','trajectory_3QP'):
  methods=MAIN if phase=='main_2000' else MAIN+[f'{b}_1500' for b in BRANCHES];pairs=PAIRS if phase=='main_2000' else [(f'{b}_{u}',f'{b}_{v}') for b in BRANCHES for u,v in ((1500,1000),(2000,1500),(2000,1000))]+[(f'{b}_1500',f'{BRANCHES[0]}_1500') for b in BRANCHES[1:]];expected=10 if phase=='main_2000' else 3
  for d in DATASETS:
   base='mcljcv' if d=='mcljcv29' else d
   for v in points(d):
    cs={m:[r for r in records if r['dataset']==base and r['video']==v['video_index'] and r['method']==m and (phase=='main_2000' or r['QP'] in (0,5,9))] for m in methods}
    for metric in ('LPIPS','DISTS','FloLPIPS'):curve(cs,d,v['name'],metric,methods,pairs,phase,expected)
   cs={m:[r for r in pools if r['dataset']==d and r['method']==m and r['status']=='available' and (phase=='main_2000' or r['QP'] in (0,5,9))] for m in methods}
   for metric in ('FID','D_mu','D_cov'):curve(cs,d,'pooled',metric,methods,pairs,phase,expected)
 groups=collections.defaultdict(list)
 for r in rows:groups[(r['phase'],r['dataset'],r['metric'],r['method'],r['reference'],r['rate_band'],r.get('contrast','paired_difference'))].append(r)
 summary=[]
 for (phase,d,k,m,ref,band,contrast),rs in groups.items():
  row=dict(phase=phase,dataset=d,metric=k,method=m,reference=ref,rate_band=band,contrast=contrast,expected_videos=len(rs),completed_videos=sum(r['status']=='available' for r in rs),aggregation='fixed dataset pooled FID features' if rs[0]['video']=='pooled' else 'equal video weight',interval_scope=rs[0]['interval_scope'])
  if row['completed_videos']!=len(rs):row.update(status='unavailable',reason='one or more required videos lack common measured interval')
  elif contrast=='difference_of_differences':row.update(status='available',delta=sum(r['delta'] for r in rs)/len(rs),old_gap_mean=sum(r['old_gap_mean'] for r in rs)/len(rs),new_gap_mean=sum(r['new_gap_mean'] for r in rs)/len(rs),relative_change_percent=None)
  else:
   a=sum(r['method_mean'] for r in rs)/len(rs);b=sum(r['reference_mean'] for r in rs)/len(rs);row.update(status='available',method_mean=a,reference_mean=b,delta=a-b,relative_change_percent=100*(a-b)/b if b else None,better_grid_fraction=sum(r['better_grid_fraction'] for r in rs)/len(rs),relative_definition='100*(dataset_method_mean-dataset_reference_mean)/dataset_reference_mean; NOT mean of video percentages or BD-rate')
  summary.append(row)
 signrows=[];gg=collections.defaultdict(list)
 for g in grids:
  if g['phase']=='main_2000':gg[(g['dataset'],g['video'],g['metric'])].append(g)
 for (d,v,metric),gs in gg.items():
  gs.sort(key=lambda x:x['index'])
  for m,ref in PAIRS:
   start=0;signs=[-1 if g[m]<g[ref] else 1 if g[m]>g[ref] else 0 for g in gs]
   for j in range(1,len(gs)+1):
    if j<len(gs) and signs[j]==signs[start]:continue
    signrows.append(dict(dataset=d,video=v,metric=metric,method=m,reference=ref,sign=signs[start],meaning={-1:'method lower',0:'equal',1:'method higher'}[signs[start]],bpp_first=gs[start]['bpp'],bpp_last=gs[j-1]['bpp'],grid_first=start,grid_last=j-1,scope='sampled interpolated grid sign run, not independent measured encodes'));start=j
 write(ROOT/'rate_sign_ranges.csv',signrows or [dict(status='unavailable')])
 write(ROOT/'equal_rate_per_sequence.csv',rows);write(ROOT/'equal_rate_summary.csv',summary);write(ROOT/'common_rate_grid.csv',grids);write(ROOT/'common_intervals.csv',intervals);write(ROOT/'pairwise_fallback.csv',fallback or [dict(status='not_needed')]);write(ROOT/'difference_of_differences.csv',[r for r in summary if r['contrast']=='difference_of_differences']);write(ROOT/'trajectory_3QP.csv',[r for r in summary if r['phase']=='trajectory_3QP'])
 return summary,intervals
