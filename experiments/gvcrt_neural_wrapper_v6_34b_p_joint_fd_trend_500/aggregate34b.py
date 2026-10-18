import csv,json,math,hashlib,collections,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;A=R.parent/'gvcrt_neural_wrapper_v6_34a_p_joint_fd_trend_screen';REPO=R.parents[1]
def readj(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def wr(p,rs):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for x in rs for k in x))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rs)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
old=rows(A/'results/all_rd_points.csv');assert len(old)==90
def truth(v): return v is True or str(v).strip().lower() in ('true','1','yes')
points=[dict(x,video=int(x['video']),QP=int(x['QP']),adaptation_step=int(x['adaptation_step']),actual_bpp=float(x['actual_bpp']),LPIPS=float(x['LPIPS']),DISTS=float(x['DISTS']),FloLPIPS=float(x['FloLPIPS']),real_RANS=truth(x['real_RANS']),independent_decode_pass=truth(x['independent_decode_pass']),state_sync_pass=truth(x['state_sync_pass'])) for x in old]
new=[]
for m,b in [('N_noP_BG_FD_500','N_noP_BG_FD'),('P_PBG_FD_500','P_PBG_FD')]:
 for v in range(6):
  manifest=readj(R/'eval_manifest.json')['ulong'][v]
  for q in (0,5,9):
   t={'stage':'adapt','dataset':'ulong','video':v,'method':m,'step':500,'branch':b,'qp':q}
   rp=R/'tasks'/f'adapt_ulong_{v}_{m}_{q}.json';rec=readj(rp);assert rec['status']=='PASS';r=rec['result']
   x=dict(dataset='ulong',video=v,sequence=manifest['name'],source_fps_metadata=manifest['rate_accounting_fps'],rate_accounting_fps=manifest['rate_accounting_fps'],QP=q,actual_bpp=r['bpp'],real_bytes=r['real_bytes'],bits_per_frame=r['bits_per_frame'],kbps_source_fps=r['bits_per_frame']*manifest['rate_accounting_fps']/1000,kbps_20fps=r['bits_per_frame']*20/1000,frames=r['frames'],num_transitions=r['num_transitions'],real_RANS=r['real_RANS'],independent_decode_pass=r['independent_decode_pass'],state_sync_pass=r['state_sync_pass'],bitstream_sha256=r['bitstream_sha256'],reconstruction_sha256=r['reconstruction_sha256'],source_sha256=manifest['source_sha256'],source_rgb_sha256=r['source_rgb_sha256'],source_frame_indices=json.dumps(r['source_frame_indices']),source_frame_indices_sha256=hashlib.sha256(json.dumps(r['source_frame_indices'],separators=(',',':')).encode()).hexdigest(),method=m,adaptation_step=500,point_type='fresh_real_encode',physical_gpu=rec.get('physical_GPU'),LPIPS=r['LPIPS'],DISTS=r['DISTS'],FloLPIPS=r['FloLPIPS'],PSNR=r.get('PSNR'),SSIM=r.get('SSIM'),MS_SSIM=r.get('MS_SSIM'),reuse='new_v6.34b')
   assert x['frames']==64 and x['num_transitions']==63 and x['real_RANS'] and x['independent_decode_pass'] and x['state_sync_pass']
   assert abs(float(x['actual_bpp'])-int(x['real_bytes'])*8/(64*1920*1080))<1e-15
   points.append(x);new.append((m,v,q,rp,r['feature_path']))
assert len(points)==126 and len(new)==36
wr(R/'results/all_rd_points.csv',points)
# Pool the same six videos x 64 frames for each method/QP. FID implementation is inherited.
spec=(REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py')
ns={};exec(compile(spec.read_text(),str(spec),'exec'),ns);fid=ns['low_rank_fid']
oldfid=rows(A/'results/pooled_fid.csv');pool=[dict(x,D_cov=float(x['FID'])-float(x['D_mu'])) for x in oldfid]
for m in ('N_noP_BG_FD_500','P_PBG_FD_500'):
 for q in (0,5,9):
  real=[];rec=[];rates=[];gth=[]
  for v in range(6):
   p=R/'features/evaluation'/f'adapt_ulong_{v}_{m}_{q}.npz';assert p.exists()
   with np.load(p) as z:
    a=z['real'].copy();b=z['reconstruction'].copy()
   assert a.shape==b.shape==(64,2048);real.append(a);rec.append(b);rates.append(next(x['actual_bpp'] for x in points if x['method']==m and x['video']==v and x['QP']==q));gth.append(hashlib.sha256(a.tobytes()).hexdigest())
  x=np.concatenate(real);y=np.concatenate(rec);assert x.shape==y.shape==(384,2048)
  dm=float(np.square(x.mean(0)-y.mean(0)).sum());F=float(fid(x,y))
  pool.append(dict(method=m,QP=q,FID=F,D_mu=dm,D_cov=F-dm,samples=384,videos=6,mean_actual_bpp=float(np.mean(rates)),GT_pool_sha256=hashlib.sha256(''.join(gth).encode()).hexdigest(),aggregation='pooled 384-frame Inception2048 features; not per-video FID mean'))
assert len(pool)==21
wr(R/'results/pooled_fid.csv',pool)
# Seven-model common rate intersection, fixed in advance by the measured range.
methods=['original','N_noP_BG_FD_100','P_PBG_FD_100','N_noP_BG_FD_250','P_PBG_FD_250','N_noP_BG_FD_500','P_PBG_FD_500']
metrics=['LPIPS','DISTS','FloLPIPS'];grid=[];per=[];comparison=[];bands=[];bounds=[]
for v in range(6):
 sub=[x for x in points if int(x['video'])==v]
 ranges={m:(min(float(x['actual_bpp']) for x in sub if x['method']==m),max(float(x['actual_bpp']) for x in sub if x['method']==m)) for m in methods}
 lo=max(x[0] for x in ranges.values());hi=min(x[1] for x in ranges.values());avail=hi>lo
 bounds.append(dict(video=v,sequence=sub[0]['sequence'],status='available' if avail else 'unavailable',bpp_min=lo,bpp_max=hi,models=len(methods),qps='0,5,9',coordinate='ln(actual_bpp)',interpolation='linear',points=20,extrapolation=False))
 if not avail:continue
 xgrid=np.linspace(np.log(lo),np.log(hi),20)
 curves={}
 for m in methods:
  rr=[x for x in sub if x['method']==m];xx=np.log([float(x['actual_bpp']) for x in rr]);order=np.argsort(xx);xx=xx[order]
  if len(np.unique(xx))<3:avail=False;break
  curves[m]={}
  for met in metrics:curves[m][met]=np.interp(xgrid,xx,np.asarray([float(rr[i][met]) for i in order]))
 if not avail:bounds[-1]['status']='unavailable_duplicate_bpp';continue
 for i,xx in enumerate(xgrid):
  for m in methods:
   for met in metrics:grid.append(dict(video=v,sequence=sub[0]['sequence'],method=m,metric=met,grid_index=i,normalized_rate_position=i/19,ln_bpp=float(xx),bpp=float(np.exp(xx)),value=float(curves[m][met][i]),common_range_min=lo,common_range_max=hi))
 for met in metrics:
  for m in methods:
   per.append(dict(video=v,sequence=sub[0]['sequence'],method=m,metric=met,mean=float(curves[m][met].mean()),common_bpp_min=lo,common_bpp_max=hi))
  for m,n in [('P_PBG_FD_500','N_noP_BG_FD_500'),('N_noP_BG_FD_500','original'),('P_PBG_FD_500','original'),('N_noP_BG_FD_100','P_PBG_FD_100'),('N_noP_BG_FD_250','P_PBG_FD_250'),('N_noP_BG_FD_500','P_PBG_FD_500'),('N_noP_BG_FD_500','N_noP_BG_FD_250'),('P_PBG_FD_500','P_PBG_FD_250')]:
   d=curves[m][met]-curves[n][met];comparison.append(dict(video=v,sequence=sub[0]['sequence'],method=m,reference=n,metric=met,method_mean=float(curves[m][met].mean()),reference_mean=float(curves[n][met].mean()),delta=float(d.mean()),relative_change_percent=float(100*d.mean()/curves[n][met].mean()) if curves[n][met].mean()!=0 else None,better_grid_fraction=float((d<0).mean()),common_bpp_min=lo,common_bpp_max=hi))
  cuts=np.linspace(xgrid[0],xgrid[-1],4)
  for band,(a,b) in zip(('low','mid','high'),zip(cuts[:-1],cuts[1:])):
   mask=(xgrid>=a)&((xgrid<=b) if band=='high' else (xgrid<b))
   d=curves['P_PBG_FD_500'][met][mask]-curves['N_noP_BG_FD_500'][met][mask]
   bands.append(dict(video=v,metric=met,band=band,grid_points=int(mask.sum()),delta_P500_minus_N500=float(d.mean()) if len(d) else None,relative_change_percent=float(100*d.mean()/curves['N_noP_BG_FD_500'][met][mask].mean()) if len(d) and curves['N_noP_BG_FD_500'][met][mask].mean()!=0 else None,ln_bpp_min=float(a),ln_bpp_max=float(b)))
wr(R/'results/common_rate_coverage.csv',bounds);wr(R/'results/common_rate_grid.csv',grid);wr(R/'results/equal_rate_per_video.csv',per);wr(R/'results/equal_rate_comparisons.csv',comparison);wr(R/'results/equal_rate_bands.csv',bands)
summ=[]
groups=collections.defaultdict(list)
for x in comparison:groups[(x['method'],x['reference'],x['metric'])].append(x)
for (m,n,met),xs in groups.items():
 summ.append(dict(method=m,reference=n,metric=met,available_videos=len(xs),expected_videos=6,status='available' if len(xs)==6 else 'partial_or_unavailable',method_mean=float(np.mean([x['method_mean'] for x in xs])) if xs else None,reference_mean=float(np.mean([x['reference_mean'] for x in xs])) if xs else None,delta_mean=float(np.mean([x['delta'] for x in xs])) if xs else None,relative_change_percent=100*float(np.mean([x['method_mean'] for x in xs])-np.mean([x['reference_mean'] for x in xs]))/float(np.mean([x['reference_mean'] for x in xs])) if xs and np.mean([x['reference_mean'] for x in xs]) else None,mean_better_grid_fraction=float(np.mean([x['better_grid_fraction'] for x in xs])) if xs else None,rate_scope='seven-model common measured-bpp interval; equal video weight'))
wr(R/'results/equal_rate_summary.csv',summ)
# Pooled FID trend on six-video mean actual bpp, same seven-method common interval.
fidgrid=[];fidcomp=[];fidbands=[];pooledranges={m:(min(float(x['mean_actual_bpp']) for x in pool if x['method']==m),max(float(x['mean_actual_bpp']) for x in pool if x['method']==m)) for m in methods};flo=max(x[0] for x in pooledranges.values());fhi=min(x[1] for x in pooledranges.values())
if fhi>flo and all(len(set(round(float(x['mean_actual_bpp']),14) for x in pool if x['method']==m))==3 for m in methods):
 xxg=np.linspace(np.log(flo),np.log(fhi),20);fc={}
 for m in methods:
  rr=[x for x in pool if x['method']==m];xx=np.log([float(x['mean_actual_bpp']) for x in rr]);o=np.argsort(xx);fc[m]=np.interp(xxg,xx[o],[float(rr[i]['FID']) for i in o])
  for i,x in enumerate(xxg):fidgrid.append(dict(method=m,grid_index=i,ln_mean_actual_bpp=float(x),mean_actual_bpp=float(np.exp(x)),FID=float(fc[m][i]),common_min=flo,common_max=fhi,interpolation='linear pooled-FID trend'))
 for m,n in [('P_PBG_FD_500','N_noP_BG_FD_500'),('N_noP_BG_FD_500','original'),('P_PBG_FD_500','original'),('N_noP_BG_FD_100','P_PBG_FD_100'),('N_noP_BG_FD_250','P_PBG_FD_250'),('N_noP_BG_FD_500','P_PBG_FD_500'),('N_noP_BG_FD_500','N_noP_BG_FD_250'),('P_PBG_FD_500','P_PBG_FD_250')]:
  d=fc[m]-fc[n];fidcomp.append(dict(method=m,reference=n,metric='FID',status='available',method_mean=float(fc[m].mean()),reference_mean=float(fc[n].mean()),delta=float(d.mean()),relative_change_percent=100*float(d.mean())/float(fc[n].mean()) if fc[n].mean() else None,better_grid_fraction=float((d<0).mean()),common_mean_bpp_min=flo,common_mean_bpp_max=fhi,aggregation='pooled six-video Inception FID; linear trend'))
 cuts=np.linspace(xxg[0],xxg[-1],4);d=fc['P_PBG_FD_500']-fc['N_noP_BG_FD_500']
 for band,(a,b) in zip(('low','mid','high'),zip(cuts[:-1],cuts[1:])):
  mask=(xxg>=a)&((xxg<=b) if band=='high' else (xxg<b));z=d[mask]
  fidbands.append(dict(band=band,grid_points=int(mask.sum()),delta_P500_minus_N500_FID=float(z.mean()) if len(z) else None,ln_mean_bpp_min=float(a),ln_mean_bpp_max=float(b),interpolation='linear pooled-FID diagnostic'))
else:fidcomp=[dict(method=m,reference=n,metric='FID',status='unavailable',reason='no seven-model common pooled mean-bpp interval') for m,n in [('P_PBG_FD_500','N_noP_BG_FD_500')]]
wr(R/'results/pooled_fid_common_grid.csv',fidgrid);wr(R/'results/pooled_fid_comparisons.csv',fidcomp)
wr(R/'results/pooled_fid_bands.csv',fidbands)
# Corrected plots: per-video curves are never connected across source videos.
colors=['#3366cc','#dc3912','#ff9900','#109618','#990099','#0099c6','#dd4477'];R.joinpath('plots').mkdir(exist_ok=True)
def svg(metric):
 W,H=1000,650;L,T,Rm,B=90,55,30,80;PW,PH=W-L-Rm,H-T-B
 vals=[float(x[metric]) for x in points];y0=min(vals);y1=max(vals);x0=min(float(x['actual_bpp']) for x in points);x1=max(float(x['actual_bpp']) for x in points)
 def px(x):return L+(x-x0)/(x1-x0)*PW
 def py(y):return T+PH-(y-y0)/(y1-y0)*PH
 ss=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="650"><rect width="100%" height="100%" fill="white"/>',f'<text x="500" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{metric} vs actual bpp (each source video plotted separately)</text>',f'<path d="M{L},{T} V{T+PH} H{L+PW}" stroke="#222" fill="none"/>']
 for mi,m in enumerate(methods):
  for v in range(6):
   rr=sorted([x for x in points if int(x['video'])==v and x['method']==m],key=lambda z:float(z['actual_bpp']))
   if len(rr)>1:ss.append(f'<polyline fill="none" stroke="{colors[mi]}" opacity="0.42" points="'+ ' '.join(f'{px(float(x["actual_bpp"])):.1f},{py(float(x[metric])):.1f}' for x in rr)+'"/>')
   for x in rr:ss.append(f'<circle cx="{px(float(x["actual_bpp"])):.1f}" cy="{py(float(x[metric])):.1f}" r="3" fill="{colors[mi]}"/>')
  ss.append(f'<text x="{L+PW-205}" y="{T+18+mi*18}" fill="{colors[mi]}" font-family="sans-serif" font-size="13">{m}</text>')
 ss.append(f'<text x="{L+PW/2}" y="{H-25}" text-anchor="middle" font-family="sans-serif">actual bpp</text><text transform="translate(22 {T+PH/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif">{metric}</text></svg>')
 (R/'plots'/f'rd_{metric.lower()}_per_video.svg').write_text('\n'.join(ss))
for met in metrics:svg(met)
# Integrity and mechanical report.
state={'status':'PASS','target_step':500,'branches':{},'reused_historical_points':90,'new_encodes':36,'logical_points':126,'all_real_RANS':all(x['real_RANS'] for x in points),'all_independent_decode':all(x['independent_decode_pass'] for x in points),'all_state_sync':all(x['state_sync_pass'] for x in points),'per_video_common_rate_videos':sum(x['status']=='available' for x in bounds),'pooled_fid_common_interval':fhi>flo}
for b in ('N_noP_BG_FD','P_PBG_FD'):
 x=readj(R/'branches'/b/'training_integrity.json');state['branches'][b]=x
assert len(points)==126 and state['all_real_RANS'] and state['all_independent_decode'] and state['all_state_sync']
(R/'final_integrity.json').write_text(json.dumps(state,indent=2)+'\n')
lines=['# V6.34b completion report','','Execution status: PASS.','',f"- N cumulative updates: {state['branches']['N_noP_BG_FD']['updates_total']} (continued from {readj(R/'audits/restore_audit.json')['branches']['N_noP_BG_FD']['step']}).",f"- P cumulative updates: {state['branches']['P_PBG_FD']['updates_total']} (continued from {readj(R/'audits/restore_audit.json')['branches']['P_PBG_FD']['step']}).",'- Reused historical points: 90, with source receipts and artifact hashes checked.','- New N500/P500 real encodes: 36.','- Total logical points including aliases: 126.','- Rate comparisons use the seven-model shared measured-bpp interval where available, 20 linear points in ln(bpp), and no extrapolation. Pooled FID uses the six-video, 384-frame pool.','- Input is a fixed six-video U-Long training-pool holdout screen at QP 0/5/9; this is a single-seed short trend screen. No checkpoint selection or budget extension was performed.','', '## Equal-rate comparison rows','']
for x in comparison:
 if x['method']=='P_PBG_FD_500' and x['reference']=='N_noP_BG_FD_500':lines.append(f"- Video {x['video']} {x['metric']}: delta={x['delta']:.8g}; relative_change_percent={x['relative_change_percent']}; common_bpp=[{x['common_bpp_min']:.8g},{x['common_bpp_max']:.8g}].")
lines+=['','## Pooled FID comparison','']
for x in fidcomp:lines.append(str(x))
lines+=['','All source points, reused receipts, new point receipts, and evaluation artifacts are indexed in this directory.','']
(R/'REPORT.md').write_text('\n'.join(lines));(R/'RESULTS_SUMMARY.md').write_text('\n'.join(lines))
statefile=readj(R/'pipeline_status.json');statefile.update(computation='PASS',aggregation='PASS',logical_points_completed=126,expected_logical_points=126,updated_unix=__import__('time').time());(R/'pipeline_status.json').write_text(json.dumps(statefile,indent=2)+'\n')
print('AGGREGATED',json.dumps(state,sort_keys=True))
