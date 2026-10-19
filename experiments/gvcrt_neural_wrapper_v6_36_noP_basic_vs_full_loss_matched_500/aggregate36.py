import csv,json,hashlib,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
METHODS=['O_original','B_basicLoss_noP_500','F_fullLoss_noP_500'];DATASETS=['ulong','uvg_holdout','hevc_b'];QPS=[0,5,9];METRICS=['LPIPS','DISTS','FloLPIPS']
def readcsv(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def interp(rows,metric,g):
 rr=sorted(rows,key=lambda z:float(z['actual_bpp']));grp={}
 for x in rr:grp.setdefault(float(x['actual_bpp']),[]).append(x)
 rates=sorted(grp);return np.interp(g,np.log(rates),[np.mean([float(x[metric]) for x in grp[r]]) for r in rates])
def main():
 manifest=json.load(open(R/'eval_manifest.json'));points=[]
 for r in readcsv(R/'results/reused_O_F_points.csv'):
  for k in ('video','QP','adaptation_step','frames','num_transitions','real_bytes','total_bits'):r[k]=int(r[k])
  for k in ('actual_bpp','LPIPS','DISTS','FloLPIPS'):r[k]=float(r[k])
  points.append(r)
 for p in sorted((R/'tasks').glob('adapt_*.json')):
  if p.name.endswith('.attempts.json'):continue
  d=json.load(open(p));assert d['status']=='PASS',p;t=d['task'];r=d['result'];v=next(v for v in manifest[t['dataset']] if v['video_index']==int(t['video']))
  points.append(dict(dataset=t['dataset'],video=int(t['video']),sequence=v['name'],source_fps_metadata=v.get('source_fps_metadata',v['rate_accounting_fps']),rate_accounting_fps=v['rate_accounting_fps'],QP=int(t['qp']),method='B_basicLoss_noP_500',adaptation_step=500,actual_bpp=float(r['bpp']),real_bytes=int(r['real_bytes']),total_bits=int(r['real_bytes'])*8,bits_per_frame=float(r['bits_per_frame']),kbps_20fps=float(r['kbps_20fps']),frames=int(r['frames']),num_transitions=int(r['num_transitions']),real_RANS=bool(r['real_RANS']),independent_decode_pass=bool(r['independent_decode_pass']),state_sync_pass=bool(r['state_sync_pass']),bitstream_path=r['bitstream_path'],bitstream_sha256=r['bitstream_sha256'],feature_path=r['feature_path'],feature_sha256=r['feature_sha256'],frame_metrics_path=r['frame_metrics_path'],frame_metrics_sha256=r['frame_metrics_sha256'],transitions_path=r['transitions_path'],transitions_sha256=r['transitions_sha256'],reconstruction_sha256=r['reconstruction_sha256'],source_sha256=v['source_sha256'],source_rgb_sha256=r['source_rgb_sha256'],source_frame_indices=json.dumps(r['source_frame_indices']),actual_qps=json.dumps(r['actual_qps']),LPIPS=float(r['LPIPS']),DISTS=float(r['DISTS']),FloLPIPS=float(r['FloLPIPS']),PSNR=float(r.get('PSNR',float('nan'))),SSIM=float(r.get('SSIM',float('nan'))),MS_SSIM=float(r.get('MS_SSIM',float('nan'))),reuse='new v6.36 B real encode',reuse_source_receipt=str(p),reuse_source_receipt_sha256=sha(p),physical_gpu=d.get('physical_GPU'),peak_memory_MiB=d.get('peak_memory_MiB')))
 assert len(points)==117,len(points)
 keys=[(x['dataset'],int(x['video']),x['method'],int(x['QP'])) for x in points];assert len(keys)==len(set(keys))
 for ds in DATASETS:
  for v in manifest[ds]:
   for m in METHODS:
    for q in QPS:assert sum(1 for x in points if x['dataset']==ds and int(x['video'])==v['video_index'] and x['method']==m and int(x['QP'])==q)==1
 for x in points:
  assert x['frames']==64 and x['num_transitions']==63 and x['real_RANS'] and x['independent_decode_pass'] and x['state_sync_pass']
  assert abs(x['actual_bpp']-x['real_bytes']*8/(64*1920*1080))<1e-15
  for k in ('bitstream','feature','frame_metrics','transitions'):assert sha(x[k+'_path'])==x[k+'_sha256'],(k,x[k+'_path'])
  assert Path(x['bitstream_path']).stat().st_size==x['real_bytes']
 writecsv(R/'results/all_rd_points.csv',points)
 # Same-QP diagnostics and explicit bitstream identity.
 pairs=[]
 for ds in DATASETS:
  for v in manifest[ds]:
   for q in QPS:
    trio={m:next(x for x in points if x['dataset']==ds and int(x['video'])==v['video_index'] and x['method']==m and int(x['QP'])==q) for m in METHODS}
    for m,ref in [('B_basicLoss_noP_500','O_original'),('F_fullLoss_noP_500','O_original'),('F_fullLoss_noP_500','B_basicLoss_noP_500')]:
     a,b=trio[m],trio[ref];pairs.append(dict(dataset=ds,video=v['video_index'],sequence=v['name'],QP=q,method=m,reference=ref,method_actual_bpp=a['actual_bpp'],reference_actual_bpp=b['actual_bpp'],rate_equal=abs(a['actual_bpp']-b['actual_bpp'])<=1e-15,bitstream_sha_equal=a['bitstream_sha256']==b['bitstream_sha256'],method_bytes=a['real_bytes'],reference_bytes=b['real_bytes'],delta_LPIPS=a['LPIPS']-b['LPIPS'],delta_DISTS=a['DISTS']-b['DISTS'],delta_FloLPIPS=a['FloLPIPS']-b['FloLPIPS']))
 writecsv(R/'results/same_qp_pairs.csv',pairs)
 # Pooled FID, fixed full dataset features.
 pooled=[]
 for ds in DATASETS:
  vs=sorted(manifest[ds],key=lambda v:v['video_index'])
  for m in METHODS:
   for q in QPS:
    real=[];rec=[];rates=[];fh=[]
    for v in vs:
     p=next(x for x in points if x['dataset']==ds and int(x['video'])==v['video_index'] and x['method']==m and int(x['QP'])==q)
     z=np.load(p['feature_path']);a=z['real'].astype(np.float64,copy=True);b=z['reconstruction'].astype(np.float64,copy=True)
     assert a.shape==b.shape==(64,2048) and np.isfinite(a).all() and np.isfinite(b).all();real.append(a);rec.append(b);rates.append(float(p['actual_bpp']));fh.append(p['feature_sha256'])
    x=np.concatenate(real);y=np.concatenate(rec);assert x.shape==y.shape==(64*len(vs),2048)
    mx=x.mean(0);my=y.mean(0);xc=x-mx;yc=y-my;cross=xc@yc.T/math.sqrt((len(x)-1)*(len(y)-1));sv=np.linalg.svd(cross,compute_uv=False)
    dmu=float(np.square(mx-my).sum());dcov=float((xc*xc).sum()/(len(x)-1)+(yc*yc).sum()/(len(y)-1)-2*sv.sum());fid=dmu+dcov
    gh=hashlib.sha256(b''.join(a.astype('<f4').tobytes(order='C') for a in real)).hexdigest()
    pooled.append(dict(dataset=ds,method=m,QP=q,FID=fid,D_mu=dmu,D_cov=dcov,closure_error=fid-dmu-dcov,samples=len(x),videos=len(vs),mean_actual_bpp=float(np.mean(rates)),GT_pool_sha256=gh,feature_sha256s=json.dumps(fh),aggregation='pooled Inception2048; float64; sample covariance ddof=1; low-rank cross-SVD'))
 assert len(pooled)==27
 for ds in DATASETS:assert len({r['GT_pool_sha256'] for r in pooled if r['dataset']==ds})==1
 writecsv(R/'results/pooled_fid.csv',pooled)
 fid_grid=[];fid_bounds=[];fid_summary=[]
 for ds in DATASETS:
  sub=[x for x in pooled if x['dataset']==ds]
  ranges={m:(min(float(x['mean_actual_bpp']) for x in sub if x['method']==m),max(float(x['mean_actual_bpp']) for x in sub if x['method']==m)) for m in METHODS}
  lo=max(v[0] for v in ranges.values());hi=min(v[1] for v in ranges.values());ok=hi>lo
  fid_bounds.append(dict(dataset=ds,status='available' if ok else 'unavailable',mean_bpp_min=lo,mean_bpp_max=hi,grid_points=20,interpolation='linear in ln(dataset mean actual bpp)',extrapolation=False))
  if not ok:continue
  gx=np.linspace(math.log(lo),math.log(hi),20);curves={}
  for m in METHODS:
   rr=sorted([x for x in sub if x['method']==m],key=lambda x:float(x['mean_actual_bpp']))
   xp=np.log([float(x['mean_actual_bpp']) for x in rr]);curves[m]={}
   for metric in ('FID','D_mu','D_cov'):
    yp=np.array([float(x[metric]) for x in rr]);curves[m][metric]=np.interp(gx,xp,yp)
   for i,lx in enumerate(gx):fid_grid.append(dict(dataset=ds,method=m,grid_index=i,mean_actual_bpp=float(np.exp(lx)),ln_mean_actual_bpp=float(lx),FID=float(curves[m]['FID'][i]),D_mu=float(curves[m]['D_mu'][i]),D_cov=float(curves[m]['D_cov'][i]),component_interpolation='same linear weights in ln(mean bpp); diagnostic',FID_component_closure=float(curves[m]['FID'][i]-curves[m]['D_mu'][i]-curves[m]['D_cov'][i]),common_mean_bpp_min=lo,common_mean_bpp_max=hi))
  for m,ref in [('F_fullLoss_noP_500','B_basicLoss_noP_500'),('B_basicLoss_noP_500','O_original'),('F_fullLoss_noP_500','O_original')]:
   for metric in ('FID','D_mu','D_cov'):
    a=curves[m][metric];b=curves[ref][metric];d=a-b
    fid_summary.append(dict(dataset=ds,method=m,reference=ref,metric=metric,method_mean=float(a.mean()),reference_mean=float(b.mean()),delta=float(d.mean()),relative_change_percent=float(100*d.mean()/b.mean()) if b.mean() else None,better_grid_fraction=float((d<0).mean()),common_mean_bpp_min=lo,common_mean_bpp_max=hi,interpolation='linear diagnostic; not independent samples'))
 writecsv(R/'results/pooled_fid_common_bounds.csv',fid_bounds);writecsv(R/'results/pooled_fid_common_grid.csv',fid_grid);writecsv(R/'results/pooled_fid_equal_rate_summary.csv',fid_summary)
 # Per-video shared ranges and equal-video interpolation.
 comps=[('F_fullLoss_noP_500','B_basicLoss_noP_500'),('B_basicLoss_noP_500','O_original'),('F_fullLoss_noP_500','O_original')]
 per=[];gridrows=[];bounds=[];summary=[];bands=[]
 for ds in DATASETS:
  vs=sorted(manifest[ds],key=lambda v:v['video_index']);vdata={};allbounds=[]
  for v in vs:
   sub=[x for x in points if x['dataset']==ds and int(x['video'])==v['video_index']]
   ranges={m:(min(x['actual_bpp'] for x in sub if x['method']==m),max(x['actual_bpp'] for x in sub if x['method']==m)) for m in METHODS};lo=max(z[0] for z in ranges.values());hi=min(z[1] for z in ranges.values())
   vdata[v['video_index']]=(sub,lo,hi);allbounds.append((lo,hi));bounds.append(dict(dataset=ds,video=v['video_index'],sequence=v['name'],status='available' if hi>lo else 'unavailable',bpp_min=lo,bpp_max=hi,grid_points=20,interpolation='piecewise linear in ln(actual_bpp)',extrapolation=False))
   if hi<=lo:continue
   gx=np.linspace(math.log(lo),math.log(hi),20)
   for m in METHODS:
    rr=[x for x in sub if x['method']==m]
    for met in METRICS:
     yy=interp(rr,met,gx)
     for i,(lx,val) in enumerate(zip(gx,yy)):gridrows.append(dict(dataset=ds,video=v['video_index'],sequence=v['name'],method=m,metric=met,grid_index=i,actual_bpp=float(np.exp(lx)),ln_actual_bpp=float(lx),value=float(val),video_common_bpp_min=lo,video_common_bpp_max=hi,interpolation='piecewise linear',extrapolation=False))
     per.append(dict(dataset=ds,video=v['video_index'],sequence=v['name'],method=m,metric=met,mean_over_20_points=float(yy.mean()),common_bpp_min=lo,common_bpp_max=hi))
  glo=max(z[0] for z in allbounds);ghi=min(z[1] for z in allbounds);bounds.append(dict(dataset=ds,video='ALL_VIDEOS_GLOBAL',sequence='dataset-wide shared range',status='available' if ghi>glo else 'unavailable',bpp_min=glo,bpp_max=ghi,grid_points=20,interpolation='piecewise linear in ln(actual_bpp)',extrapolation=False))
  valid=[v for v in vs if vdata[v['video_index']][2]>vdata[v['video_index']][1]]
  for m,ref in comps:
   for met in METRICS:
    am=[];bm=[]
    for v in valid:
     am.append(next(z['mean_over_20_points'] for z in per if z['dataset']==ds and z['video']==v['video_index'] and z['method']==m and z['metric']==met));bm.append(next(z['mean_over_20_points'] for z in per if z['dataset']==ds and z['video']==v['video_index'] and z['method']==ref and z['metric']==met))
    if not valid:continue
    a=float(np.mean(am));b=float(np.mean(bm));summary.append(dict(dataset=ds,method=m,reference=ref,metric=met,method_mean=a,reference_mean=b,delta=a-b,relative_change_percent=100*(a-b)/b if b else None,video_count=len(valid),aggregation='per-video shared actual-bpp ranges; 20-point linear ln-bpp interpolation; equal video weighting'))
    for bi,band in enumerate(('low','mid','high')):
     lo_i=bi*20//3;hi_i=(bi+1)*20//3 if bi<2 else 20;av=[];bv=[]
     for v in valid:
      sub=[z for z in gridrows if z['dataset']==ds and z['video']==v['video_index'] and z['metric']==met]
      av.append(float(np.mean([z['value'] for z in sub if z['method']==m][lo_i:hi_i])));bv.append(float(np.mean([z['value'] for z in sub if z['method']==ref][lo_i:hi_i])))
     x=float(np.mean(av));y=float(np.mean(bv));bands.append(dict(dataset=ds,method=m,reference=ref,metric=met,band=band,method_mean=x,reference_mean=y,delta=x-y,relative_change_percent=100*(x-y)/y if y else None,video_count=len(valid),rule='equal video weight; thirds of each video-specific common ln-bpp range'))
 writecsv(R/'results/common_rate_bounds.csv',bounds);writecsv(R/'results/common_rate_grid.csv',gridrows);writecsv(R/'results/equal_rate_per_video.csv',per);writecsv(R/'results/equal_rate_summary.csv',summary);writecsv(R/'results/equal_rate_bands.csv',bands)
 # Secondary dataset-wide interval, used only when all videos share one interval.
 global_summary=[]
 for ds in DATASETS:
  vs=sorted(manifest[ds],key=lambda v:v['video_index']);v_ranges=[]
  for v in vs:
   sub=[x for x in points if x['dataset']==ds and int(x['video'])==v['video_index']]
   rr={m:(min(x['actual_bpp'] for x in sub if x['method']==m),max(x['actual_bpp'] for x in sub if x['method']==m)) for m in METHODS};v_ranges.append((max(x[0] for x in rr.values()),min(x[1] for x in rr.values())))
  lo=max(x[0] for x in v_ranges);hi=min(x[1] for x in v_ranges)
  if hi<=lo:global_summary.append(dict(dataset=ds,status='unavailable',reason='no dataset-wide shared interval across every video and method'));continue
  gx=np.linspace(math.log(lo),math.log(hi),20)
  vals={m:{met:[] for met in METRICS} for m in METHODS}
  for v in vs:
   sub=[x for x in points if x['dataset']==ds and int(x['video'])==v['video_index']]
   for m in METHODS:
    rr=[x for x in sub if x['method']==m]
    for met in METRICS:vals[m][met].append(interp(rr,met,gx))
  for m,ref in comps:
   for met in METRICS:
    a=np.stack(vals[m][met]).mean(0);b=np.stack(vals[ref][met]).mean(0);de=a-b
    global_summary.append(dict(dataset=ds,status='available',method=m,reference=ref,metric=met,method_mean=float(a.mean()),reference_mean=float(b.mean()),delta=float(de.mean()),relative_change_percent=100*float(de.mean())/float(b.mean()) if b.mean() else None,common_bpp_min=lo,common_bpp_max=hi,grid_points=20,aggregation='dataset-wide common interval; equal video weighting'))
 writecsv(R/'results/global_common_rate_summary.csv',global_summary)
 json.dump({'status':'PASS','logical_points':117,'verified_reuse_O_F':78,'new_B_points':39,'pooled_fid_points':len(pooled),'per_video_common_rate_rows':len(per),'bitstream_hash_equal_fraction':sum(x['bitstream_sha_equal'] for x in pairs)/len(pairs),'same_qp_pair_rows':len(pairs),'FID_closure_max_abs':max(abs(x['closure_error']) for x in pooled),'all_source_artifacts_hash_verified':True},open(R/'audits/aggregation_integrity.json','w'),indent=2)
 print('PASS: RD=117 FID=27; comparisons',len(summary),'per-video summaries')
if __name__=='__main__':main()
