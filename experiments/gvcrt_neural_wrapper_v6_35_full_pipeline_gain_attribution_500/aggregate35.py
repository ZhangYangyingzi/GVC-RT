import csv,hashlib,json,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
METHODS=['original','N_oursLoss_noP_500','P_oursLoss_withP_500']; DATASETS=['ulong','uvg_holdout','hevc_b']; QPS=[0,5,9]; METRICS=['LPIPS','DISTS','FloLPIPS']
def readcsv(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
# Start with the 75 previously verified normalized records, then add the 42 new task receipts.
points=readcsv(R/'results/all_rd_points.csv')
for p in points:
 vmeta=next(v for v in json.load(open(R/'eval_manifest.json'))[p['dataset']] if v['video_index']==int(p['video']))
 p['source_sha256']=p['source_sha256'] or vmeta['source_sha256']
 p['source_fps_metadata']=float(p['source_fps_metadata'] or vmeta.get('source_fps_metadata',vmeta['rate_accounting_fps']))
 if str(p.get('reuse','')).startswith('v6.34b') or str(p.get('reuse','')).startswith('v6.24'):
  rp=Path(p['reuse_source_receipt'])
  if rp.suffix=='.json':
   src=json.load(open(rp))['result']
  else:
   rows0=readcsv(rp);src=next(z for z in rows0 if z['dataset']==p['dataset'] and z['method']=='original' and int(z['video_index'])==int(p['video']) and int(z['QP'])==int(p['QP']))
  for key in ('frame_metrics_path','frame_metrics_sha256','transitions_path','transitions_sha256','I_bits_including_associated_headers','P_bits_including_headers','header_and_SPS_bits'):
   if src.get(key) not in (None,''):p[key]=src[key]
 for k in ('video','QP','adaptation_step','frames','num_transitions','real_bytes'):p[k]=int(p[k])
 for k in ('actual_bpp','LPIPS','DISTS','FloLPIPS'):p[k]=float(p[k])
 p['real_RANS']=str(p['real_RANS']).lower()=='true';p['independent_decode_pass']=str(p['independent_decode_pass']).lower()=='true';p['state_sync_pass']=str(p['state_sync_pass']).lower()=='true'
for rp in sorted((R/'tasks').glob('adapt_*.json')):
 d=json.load(open(rp))
 if d.get('status')!='PASS':continue
 t=d['task'];r=d['result'];v=next(x for x in json.load(open(R/'eval_manifest.json'))[t['dataset']] if x['video_index']==t['video'])
 row=dict(dataset=t['dataset'],video=t['video'],sequence=v['name'],source_fps_metadata=v.get('source_fps_metadata',v['rate_accounting_fps']),rate_accounting_fps=v['rate_accounting_fps'],QP=t['qp'],method=t['method'],adaptation_step=500,
 actual_bpp=float(r['bpp']),real_bytes=int(r['real_bytes']),total_bits=int(r['real_bytes'])*8,bits_per_frame=float(r['bits_per_frame']),kbps_20fps=float(r['kbps_20fps']),frames=int(r['frames']),num_transitions=int(r['num_transitions']),real_RANS=bool(r['real_RANS']),independent_decode_pass=bool(r['independent_decode_pass']),state_sync_pass=bool(r['state_sync_pass']),
 bitstream_path=r['bitstream_path'],bitstream_sha256=r['bitstream_sha256'],feature_path=r['feature_path'],feature_sha256=r['feature_sha256'],frame_metrics_path=r['frame_metrics_path'],frame_metrics_sha256=r['frame_metrics_sha256'],transitions_path=r['transitions_path'],transitions_sha256=r['transitions_sha256'],actual_qps=json.dumps(r['actual_qps']),reconstruction_sha256=r['reconstruction_sha256'],source_sha256=v['source_sha256'],source_rgb_sha256=r['source_rgb_sha256'],source_frame_indices=json.dumps(r['source_frame_indices']),
 LPIPS=float(r['LPIPS']),DISTS=float(r['DISTS']),FloLPIPS=float(r['FloLPIPS']),PSNR=float(r.get('PSNR',float('nan'))),SSIM=float(r.get('SSIM',float('nan'))),MS_SSIM=float(r.get('MS_SSIM',float('nan'))),reuse='new v6.35 real encode',reuse_source_receipt=str(rp),reuse_source_receipt_sha256=sha_file(rp),physical_gpu=d.get('physical_GPU'),peak_memory_MiB=d.get('peak_memory_MiB'))
 assert row['frames']==64 and row['num_transitions']==63 and row['real_RANS'] and row['independent_decode_pass'] and row['state_sync_pass']
 assert Path(row['bitstream_path']).stat().st_size==row['real_bytes'] and sha_file(row['bitstream_path'])==row['bitstream_sha256'] and sha_file(row['feature_path'])==row['feature_sha256']
 assert abs(row['actual_bpp']-row['real_bytes']*8/(64*v['width']*v['height']))<1e-15
 points.append(row)
# unique key and expected coverage
keys=[(p['dataset'],p['video'],p['method'],p['QP']) for p in points]
assert len(keys)==len(set(keys)), 'duplicate point key'
assert len(points)==117,('point count',len(points))
for ds in DATASETS:
 vs=json.load(open(R/'eval_manifest.json'))[ds]
 for v in vs:
  for m in METHODS:
   for q in QPS:assert sum((p['dataset'],p['video'],p['method'],p['QP'])==(ds,v['video_index'],m,q) for p in points)==1
assert all(p['real_RANS'] and p['independent_decode_pass'] and p['state_sync_pass'] for p in points)
writecsv(R/'results/all_rd_points.csv',points)
# Pooled FID: inherited low-rank cross-SVD formula, pooled over all frames of the dataset.
pooled=[]; feature_cache={}
for ds in DATASETS:
 vs=sorted(json.load(open(R/'eval_manifest.json'))[ds],key=lambda x:x['video_index'])
 for m in METHODS:
  for q in QPS:
   real=[];rec=[];rates=[];feat_hashes=[]
   for v in vs:
    p=next(x for x in points if (x['dataset'],x['video'],x['method'],x['QP'])==(ds,v['video_index'],m,q))
    z=np.load(p['feature_path']);a=z['real'].astype(np.float64,copy=True);b=z['reconstruction'].astype(np.float64,copy=True)
    assert a.shape==b.shape==(64,2048) and np.isfinite(a).all() and np.isfinite(b).all()
    real.append(a);rec.append(b);rates.append(float(p['actual_bpp']));feat_hashes.append(sha_file(p['feature_path']))
   x=np.concatenate(real);y=np.concatenate(rec);assert x.shape==y.shape==(64*len(vs),2048)
   # Confirm every method/QP sees the same fixed GT feature pool.
   gt_hash=hashlib.sha256(b''.join(a.astype('<f4').tobytes(order='C') for a in real)).hexdigest()
   means_x=x.mean(0);means_y=y.mean(0);xc=x-means_x;yc=y-means_y
   cross=xc@yc.T/math.sqrt((len(x)-1)*(len(y)-1))
   covtrace=float((xc*xc).sum()/(len(x)-1)+(yc*yc).sum()/(len(y)-1))
   dmu=float(np.square(means_x-means_y).sum());fid=float(max(dmu+covtrace-2*float(np.linalg.svd(cross,compute_uv=False).sum()),0.0));dcov=fid-dmu
   pooled.append(dict(dataset=ds,method=m,QP=q,FID=fid,D_mu=dmu,D_cov=dcov,samples=len(x),videos=len(vs),mean_actual_bpp=float(np.mean(rates)),GT_pool_sha256=gt_hash,feature_files_sha256=json.dumps(feat_hashes),aggregation='pooled dataset Inception2048; float64; sample covariance ddof=1; inherited low-rank cross-SVD FID'))
   feature_cache[(ds,m,q)]=(x,y)
assert len(pooled)==27
for ds in DATASETS:
 vals={p['GT_pool_sha256'] for p in pooled if p['dataset']==ds};assert len(vals)==1,(ds,vals)
writecsv(R/'results/pooled_fid.csv',pooled)
# Common actual-bpp intersection across all listed videos and all three methods.
grid_rows=[];per_rows=[];summ_rows=[];bounds=[];band_rows=[]
COMPS=[('P_oursLoss_withP_500','original'),('N_oursLoss_noP_500','original'),('P_oursLoss_withP_500','N_oursLoss_noP_500')]
for ds in DATASETS:
 vs=sorted(json.load(open(R/'eval_manifest.json'))[ds],key=lambda x:x['video_index'])
 curves_by_video={};los=[];his=[];unavailable=False
 for v in vs:
  sub=[p for p in points if p['dataset']==ds and p['video']==v['video_index']]
  ranges={m:(min(p['actual_bpp'] for p in sub if p['method']==m),max(p['actual_bpp'] for p in sub if p['method']==m)) for m in METHODS}
  lo=max(z[0] for z in ranges.values());hi=min(z[1] for z in ranges.values());los.append(lo);his.append(hi)
  if not hi>lo:unavailable=True
  curves_by_video[v['video_index']]=(sub,ranges)
 glo=max(los);ghi=min(his);available=(not unavailable and ghi>glo)
 bounds.append(dict(dataset=ds,status='available' if available else 'unavailable',videos=len(vs),methods='|'.join(METHODS),qps='0,5,9',bpp_min=glo,bpp_max=ghi,log_bpp_min=float(np.log(glo)) if glo>0 else None,log_bpp_max=float(np.log(ghi)) if ghi>0 else None,grid_points=20,interpolation='piecewise linear in ln(actual_bpp)',extrapolation=False))
 if not available:continue
 gx=np.linspace(np.log(glo),np.log(ghi),20);per_method={m:{met:[] for met in METRICS} for m in METHODS}
 for v in vs:
  sub,_=curves_by_video[v['video_index']]
  for m in METHODS:
   rr=sorted([p for p in sub if p['method']==m],key=lambda p:p['actual_bpp'])
   # Collapse duplicate measured rates using the arithmetic mean at exactly the same bpp.
   rate_groups={}
   for p in rr:rate_groups.setdefault(float(p['actual_bpp']),[]).append(p)
   xr=np.log(np.array(sorted(rate_groups)))
   for met in METRICS:
    yr=np.array([np.mean([float(p[met]) for p in rate_groups[r]]) for r in sorted(rate_groups)])
    vals=np.interp(gx,xr,yr)
    per_method[m][met].append(vals)
    for i,(lx,val) in enumerate(zip(gx,vals)):
     grid_rows.append(dict(dataset=ds,video=v['video_index'],sequence=v['name'],method=m,metric=met,grid_index=i,actual_bpp=float(np.exp(lx)),ln_actual_bpp=float(lx),value=float(val),interpolation='piecewise linear; duplicate rates averaged',dataset_common_bpp_min=glo,dataset_common_bpp_max=ghi,extrapolation=False))
    per_rows.append(dict(dataset=ds,video=v['video_index'],sequence=v['name'],method=m,metric=met,mean_over_20_points=float(vals.mean()),dataset_common_bpp_min=glo,dataset_common_bpp_max=ghi))
 for m in METHODS:
  for met in METRICS:
   arr=np.stack(per_method[m][met]);summ_rows.append(dict(dataset=ds,method=m,metric=met,method_mean=float(arr.mean()),video_count=len(vs),aggregation='equal weight per video; mean of fixed 20-point ln-bpp grid',common_actual_bpp_min=glo,common_actual_bpp_max=ghi))
 for m,ref in COMPS:
  for met in METRICS:
   a=np.stack(per_method[m][met]).mean(0);b=np.stack(per_method[ref][met]).mean(0);delta=a-b
   summ_rows.append(dict(dataset=ds,method=m,reference=ref,metric=met,method_mean=float(a.mean()),reference_mean=float(b.mean()),delta=float(delta.mean()),relative_change_percent=float(100*delta.mean()/b.mean()) if b.mean() else None,better_grid_fraction=float((delta<0).mean()),video_count=len(vs),aggregation='equal video weight; 20-point ln-bpp grid',common_actual_bpp_min=glo,common_actual_bpp_max=ghi))
   cuts=np.linspace(gx[0],gx[-1],4)
   for band,(aa,bb) in zip(('low','mid','high'),zip(cuts[:-1],cuts[1:])):
    mask=(gx>=aa)&((gx<=bb) if band=='high' else (gx<bb));z=delta[mask]
    band_rows.append(dict(dataset=ds,method=m,reference=ref,metric=met,band=band,grid_points=int(mask.sum()),method_mean=float(a[mask].mean()),reference_mean=float(b[mask].mean()),delta=float(z.mean()),relative_change_percent=float(100*z.mean()/b[mask].mean()) if b[mask].mean() else None,ln_bpp_min=float(aa),ln_bpp_max=float(bb),common_actual_bpp_min=glo,common_actual_bpp_max=ghi))
writecsv(R/'results/common_rate_bounds.csv',bounds);writecsv(R/'results/common_rate_grid.csv',grid_rows);writecsv(R/'results/equal_rate_per_video.csv',per_rows);writecsv(R/'results/equal_rate_summary.csv',summ_rows);writecsv(R/'results/equal_rate_bands.csv',band_rows)
video_deltas=[]
for ds in DATASETS:
 for v in sorted(json.load(open(R/'eval_manifest.json'))[ds],key=lambda x:x['video_index']):
  for met in METRICS:
   lookup={(x['method'],x['metric']):x for x in per_rows if x['dataset']==ds and x['video']==v['video_index']}
   for m,ref in COMPS:
    if (m,met) not in lookup or (ref,met) not in lookup:continue
    a=float(lookup[(m,met)]['mean_over_20_points']);b=float(lookup[(ref,met)]['mean_over_20_points']);delta=a-b
    video_deltas.append(dict(dataset=ds,video=v['video_index'],sequence=v['name'],method=m,reference=ref,metric=met,method_mean=a,reference_mean=b,delta=delta,relative_change_percent=100*delta/b if b else None,better=delta<0,common_bpp_min=lookup[(m,met)]['dataset_common_bpp_min'],common_bpp_max=lookup[(m,met)]['dataset_common_bpp_max']))
writecsv(R/'results/equal_rate_video_comparisons.csv',video_deltas)
# Pooled-FID trend: dataset-level FID against mean actual bpp; no per-video FID averaging.
fid_grid=[];fid_summary=[];fid_bounds=[]
for ds in DATASETS:
 pds=[p for p in pooled if p['dataset']==ds];ranges={m:(min(float(p['mean_actual_bpp']) for p in pds if p['method']==m),max(float(p['mean_actual_bpp']) for p in pds if p['method']==m)) for m in METHODS}
 lo=max(a for a,b in ranges.values());hi=min(b for a,b in ranges.values());ok=hi>lo
 fid_bounds.append(dict(dataset=ds,status='available' if ok else 'unavailable',mean_bpp_min=lo,mean_bpp_max=hi,method_count=3,anchors_per_method=3,interpolation='linear in ln(mean actual bpp)',extrapolation=False))
 if not ok:continue
 gx=np.linspace(np.log(lo),np.log(hi),20);c={}
 for m in METHODS:
  rr=sorted([p for p in pds if p['method']==m],key=lambda p:float(p['mean_actual_bpp']));xr=np.log([float(p['mean_actual_bpp']) for p in rr]);yr=np.array([float(p['FID']) for p in rr]);c[m]=np.interp(gx,xr,yr)
  for i,x in enumerate(gx):fid_grid.append(dict(dataset=ds,method=m,grid_index=i,mean_actual_bpp=float(np.exp(x)),ln_mean_actual_bpp=float(x),FID=float(c[m][i]),common_mean_bpp_min=lo,common_mean_bpp_max=hi,interpolation='linear pooled-FID diagnostic'))
 for m,ref in COMPS:
  delta=c[m]-c[ref];fid_summary.append(dict(dataset=ds,method=m,reference=ref,metric='FID',method_mean=float(c[m].mean()),reference_mean=float(c[ref].mean()),delta=float(delta.mean()),relative_change_percent=float(100*delta.mean()/c[ref].mean()) if c[ref].mean() else None,better_grid_fraction=float((delta<0).mean()),samples=len(json.load(open(R/'eval_manifest.json'))[ds])*64,common_mean_bpp_min=lo,common_mean_bpp_max=hi,aggregation='pooled dataset FID; linear diagnostic interpolation'))
  cuts=np.linspace(gx[0],gx[-1],4)
  for band,(aa,bb) in zip(('low','mid','high'),zip(cuts[:-1],cuts[1:])):
   mask=(gx>=aa)&((gx<=bb) if band=='high' else (gx<bb));fid_summary.append(dict(dataset=ds,method=m,reference=ref,metric='FID',band=band,method_mean=float(c[m][mask].mean()),reference_mean=float(c[ref][mask].mean()),delta=float(delta[mask].mean()),relative_change_percent=float(100*delta[mask].mean()/c[ref][mask].mean()) if c[ref][mask].mean() else None,common_mean_bpp_min=lo,common_mean_bpp_max=hi,aggregation='pooled dataset FID; linear diagnostic interpolation'))
writecsv(R/'results/pooled_fid_common_grid.csv',fid_grid);writecsv(R/'results/pooled_fid_equal_rate_summary.csv',fid_summary);writecsv(R/'results/pooled_fid_common_bounds.csv',fid_bounds)
# Figures: one per dataset and perceptual metric; lines are per-video, never cross-sequence joins.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for ds in DATASETS:
 for met in METRICS:
  fig,ax=plt.subplots(figsize=(8,5))
  for v in sorted(json.load(open(R/'eval_manifest.json'))[ds],key=lambda x:x['video_index']):
   for m in METHODS:
    rr=sorted([p for p in points if p['dataset']==ds and p['video']==v['video_index'] and p['method']==m],key=lambda p:p['actual_bpp'])
    ax.plot([p['actual_bpp'] for p in rr],[p[met] for p in rr],marker='o',alpha=.34,label=m if v['video_index']==min(x['video_index'] for x in json.load(open(R/'eval_manifest.json'))[ds]) else None)
  ax.set_xscale('log');ax.set_xlabel('Actual RANS bpp');ax.set_ylabel(met+' (lower is better)');ax.set_title(f'{ds}: per-video measured RD points (no cross-video joins)');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(R/f'plots/{ds}_{met}_rd.png',dpi=150);plt.close(fig)
# common-grid equal-video curves (rate coordinate is normalized only if dataset intervals differ; keep ln bpp shown here).
for ds in DATASETS:
 b=next(x for x in bounds if x['dataset']==ds)
 if b['status']!='available':continue
 for met in METRICS:
  fig,ax=plt.subplots(figsize=(8,5))
  for m in METHODS:
   rr=[x for x in grid_rows if x['dataset']==ds and x['method']==m and x['metric']==met]
   x=sorted(set(float(y['ln_actual_bpp']) for y in rr));vals=[]
   for lx in x:vals.append(np.mean([float(y['value']) for y in rr if float(y['ln_actual_bpp'])==lx]))
   ax.plot(np.exp(x),vals,marker='o',label=m)
  ax.set_xscale('log');ax.set_xlabel('Actual RANS bpp (dataset-wide common interval)');ax.set_ylabel(met+' (lower is better)');ax.set_title(f'{ds}: equal-video mean, piecewise-linear ln-bpp grid');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(R/f'plots/{ds}_{met}_common_rate.png',dpi=150);plt.close(fig)
for ds in DATASETS:
 fig,ax=plt.subplots(figsize=(8,5))
 for m in METHODS:
  rr=sorted([x for x in pooled if x['dataset']==ds and x['method']==m],key=lambda x:float(x['mean_actual_bpp']))
  ax.plot([float(x['mean_actual_bpp']) for x in rr],[float(x['FID']) for x in rr],marker='o',label=m)
 ax.set_xscale('log');ax.set_xlabel('Dataset mean actual bpp');ax.set_ylabel('Pooled FID (lower is better)');ax.set_title(f'{ds}: pooled FID anchors');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(R/f'plots/{ds}_pooled_fid.png',dpi=150);plt.close(fig)
print('AGGREGATED',len(points),'RD',len(pooled),'pooled FID',len(grid_rows),'grid rows')
