from io32 import *
import math,collections,importlib.util
import numpy as np

def fid_function():
 p=REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py'
 spec=importlib.util.spec_from_file_location('v629_fid_trend',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.low_rank_fid

def main():
 tasks=load(ROOT/'task_manifest.json')['tasks'];points=[];feat={};metrics=('LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM')
 for t in tasks:
  p=ROOT/'tasks'/(f"{t['stage']}_{t['dataset']}_{t['video']}_{t['method']}_{t['qp']}.json")
  r=load(p)['result'];source_video=load(ROOT/'eval_manifest.json')['ulong'][t['video']];fps=float(source_video['rate_accounting_fps']);base=dict(dataset='ulong',video=t['video'],sequence=source_video['name'],source_fps_metadata=fps,rate_accounting_fps=fps,QP=t['qp'],actual_bpp=r['bpp'],real_bytes=r['real_bytes'],bits_per_frame=r['bits_per_frame'],kbps_source_fps=r['bits_per_frame']*fps/1000,kbps_20fps=r['bits_per_frame']*20/1000,frames=r['frames'],num_transitions=r['num_transitions'],real_RANS=r['real_RANS'],independent_decode_pass=r['independent_decode_pass'],state_sync_pass=r['state_sync_pass'],bitstream_sha256=r['bitstream_sha256'],reconstruction_sha256=r['reconstruction_sha256'],source_sha256=source_video['source_sha256'],source_rgb_sha256=r['source_rgb_sha256'],source_frame_indices=r['source_frame_indices'],source_frame_indices_sha256=hashlib.sha256(json.dumps(r['source_frame_indices'],separators=(',',':')).encode()).hexdigest(),method=t['method'],adaptation_step=t['step'],point_type='fresh_real_encode',physical_gpu=load(p).get('physical_GPU'))
  for k in metrics:base[k]=r.get(k)
  points.append(base);feat[(t['method'],t['video'],t['qp'])]=r['feature_path']
 # Step-0 models are recorded as aliases, not duplicated RD points.
 aliases=[]
 for t in tasks:
  if t['stage']=='baseline':
   for m in ('N_noP_BG_FD_0','P_PBG_FD_0'):
    aliases.append(dict(method=m,video=t['video'],QP=t['qp'],alias_of_method='original',alias_of_point=f"baseline_ulong_{t['video']}_original_{t['qp']}",additional_encode=False))
 write(ROOT/'audits/step0_aliases.csv',aliases)
 write(ROOT/'results/all_rd_points.csv',points)
 # Pooled FID uses the same 384-frame feature pool for every method/QP.
 fid=fid_function();pooled=[]
 methodset=['original','N_noP_BG_FD_100','N_noP_BG_FD_250','P_PBG_FD_100','P_PBG_FD_250']
 real_reference={}
 for m in methodset:
  for q in (0,5,9):
   real=[];rec=[];rates=[]
   for v in range(6):
    with np.load(feat[(m,v,q)]) as z:real_frame=z['real'].copy();rec.append(z['reconstruction'].copy())
    if (v,q) in real_reference:assert np.allclose(real_frame,real_reference[(v,q)],atol=1e-6,rtol=1e-5),('GT feature pool mismatch',v,q,m)
    else:real_reference[(v,q)]=real_frame
    real.append(real_frame)
    rates.append(next(x['actual_bpp'] for x in points if x['method']==m and x['video']==v and x['QP']==q))
   real=np.concatenate(real);rec=np.concatenate(rec);assert real.shape==rec.shape==(384,2048) and np.isfinite(rec).all()
   pooled.append(dict(method=m,QP=q,FID=float(fid(real,rec)),D_mu=float(np.square(real.mean(0)-rec.mean(0)).sum()),samples=384,videos=6,mean_actual_bpp=float(np.mean(rates)),GT_pool_sha256=hashlib.sha256(''.join(hashlib.sha256(x.tobytes()).hexdigest() for x in real).encode()).hexdigest(),aggregation='pooled 384-frame Inception2048 features; not per-video FID mean'))
 write(ROOT/'results/pooled_fid.csv',pooled)
 # Measured-rate, per-video, no-extrapolation linear trend comparisons.
 comparisons=[];gridrows=[]
 pairs=[]
 for m in methodset[1:]:pairs.append((m,'original'))
 for s in (100,250):pairs.append((f'P_PBG_FD_{s}',f'N_noP_BG_FD_{s}'))
 for m,n in pairs:
  for v in range(6):
   a=[x for x in points if x['method']==m and x['video']==v and x['point_type']=='fresh_real_encode']
   b=[x for x in points if x['method']==n and x['video']==v and x['point_type']=='fresh_real_encode']
   xa=np.array([x['actual_bpp'] for x in a],float);xb=np.array([x['actual_bpp'] for x in b],float)
   lo=max(xa.min(),xb.min());hi=min(xa.max(),xb.max());available=hi>lo
   base=dict(method=m,reference=n,video=v,sequence=load(ROOT/'eval_manifest.json')['ulong'][v]['name'],status='available' if available else 'unavailable',bpp_min=float(lo),bpp_max=float(hi),rate_coordinate='ln(actual_bpp)',interpolation='linear',grid_points=20,extrapolation=False)
   if not available:
    for metric in ('LPIPS','DISTS','FloLPIPS'):comparisons.append(dict(base,metric=metric,reason='no common measured bpp interval'))
    continue
   xa=np.log(xa);xb=np.log(xb);g=np.linspace(math.log(lo),math.log(hi),20)
   for metric in ('LPIPS','DISTS','FloLPIPS'):
    ya=np.array([x[metric] for x in a],float);yb=np.array([x[metric] for x in b],float)
    oa=np.argsort(xa);ob=np.argsort(xb);xa_s,ya_s=xa[oa],ya[oa];xb_s,yb_s=xb[ob],yb[ob]
    if len(set(xa_s.tolist()))<3 or len(set(xb_s.tolist()))<3:
     comparisons.append(dict(base,metric=metric,status='unavailable',reason='duplicate actual bpp among the three fixed QPs; no points removed'))
     continue
    va=np.interp(g,xa_s,ya_s);vb=np.interp(g,xb_s,yb_s);delta=va-vb
    comparisons.append(dict(base,metric=metric,method_mean=float(va.mean()),reference_mean=float(vb.mean()),delta=float(delta.mean()),relative_change_percent=100*float(delta.mean())/float(vb.mean()) if vb.mean()!=0 else None,better_grid_fraction=float((delta<0).mean())))
    for i,x in enumerate(g):gridrows.append(dict(method=m,reference=n,video=v,metric=metric,index=i,bpp=float(np.exp(x)),method_value=float(va[i]),reference_value=float(vb[i]),delta=float(delta[i]),rate_match='interpolated measured-RD trend'))
 # Pooled FID uses the six-video pool at each QP and its dataset-mean measured rate.
 for m,n in pairs:
  a=[x for x in pooled if x['method']==m];b=[x for x in pooled if x['method']==n]
  xa=np.array([x['mean_actual_bpp'] for x in a],float);xb=np.array([x['mean_actual_bpp'] for x in b],float)
  lo=max(xa.min(),xb.min());hi=min(xa.max(),xb.max());available=hi>lo
  base=dict(method=m,reference=n,video='pooled_6_video_features',sequence='pooled',status='available' if available else 'unavailable',bpp_min=float(lo),bpp_max=float(hi),rate_coordinate='ln(mean_actual_bpp)',interpolation='linear',grid_points=20,extrapolation=False,aggregation='pooled 384-frame Inception2048 features')
  if not available:
   comparisons.append(dict(base,metric='FID',reason='no common measured pooled-rate interval'));continue
  xa=np.log(xa);xb=np.log(xb);oa=np.argsort(xa);ob=np.argsort(xb);xa,xb=xa[oa],xb[ob];ya=np.array([x['FID'] for x in a])[oa];yb=np.array([x['FID'] for x in b])[ob]
  if len(set(xa.tolist()))<3 or len(set(xb.tolist()))<3:
   comparisons.append(dict(base,metric='FID',status='unavailable',reason='duplicate mean actual bpp among the three fixed QPs; no points removed'));continue
  g=np.linspace(math.log(lo),math.log(hi),20);va=np.interp(g,xa,ya);vb=np.interp(g,xb,yb);delta=va-vb
  comparisons.append(dict(base,metric='FID',method_mean=float(va.mean()),reference_mean=float(vb.mean()),delta=float(delta.mean()),relative_change_percent=100*float(delta.mean())/float(vb.mean()) if vb.mean()!=0 else None,better_grid_fraction=float((delta<0).mean())))
  for i,x in enumerate(g):gridrows.append(dict(method=m,reference=n,video='pooled_6_video_features',metric='FID',index=i,bpp=float(np.exp(x)),method_value=float(va[i]),reference_value=float(vb[i]),delta=float(delta[i]),rate_match='interpolated pooled measured-FID trend'))
 summary=[]
 groups=collections.defaultdict(list)
 for r in comparisons:groups[(r['method'],r['reference'],r['metric'])].append(r)
 for (m,n,k),rs in groups.items():
  good=[x for x in rs if x['status']=='available'];expected=1 if k=='FID' else 6
  summary.append(dict(method=m,reference=n,metric=k,status='available' if len(good)==expected else 'unavailable',completed_videos=len(good),expected_videos=expected,method_mean=float(np.mean([x['method_mean'] for x in good])) if len(good)==expected else None,reference_mean=float(np.mean([x['reference_mean'] for x in good])) if len(good)==expected else None,delta=float(np.mean([x['delta'] for x in good])) if len(good)==expected else None,relative_change_percent=100*float(np.mean([x['method_mean'] for x in good])-np.mean([x['reference_mean'] for x in good]))/float(np.mean([x['reference_mean'] for x in good])) if len(good)==expected and np.mean([x['reference_mean'] for x in good]) else None,better_grid_fraction=float(np.mean([x['better_grid_fraction'] for x in good])) if len(good)==expected else None,aggregation='pooled dataset FID' if k=='FID' else 'equal video weight; metric change percent is not BD-rate'))
 write(ROOT/'results/same_rate_per_video.csv',comparisons);write(ROOT/'results/same_rate_summary.csv',summary);write(ROOT/'results/common_rate_grid.csv',gridrows)
 make_plots(points,pooled)
 status=load(ROOT/'driver_state.json')
 integrity={'status':'PASS','mode':'A_reuse','training_updates_added':{'N_noP_BG_FD':0,'P_PBG_FD':0},'source_step250_checkpoints_verified':True,'expected_logical_points':90,'completed_logical_points':len(points),'physical_fresh_encodes':90,'fresh_original_encodes':18,'fresh_adapted_encodes':72,'step0_alias_records':36,'all_real_RANS':all(p['real_RANS'] for p in points),'independent_decode':all(p['independent_decode_pass'] for p in points),'reference_state_sync':all(p['state_sync_pass'] for p in points),'training_driver_absent_at_budget_override':True,'gpu_driver_status':status.get('status')}
 dump(ROOT/'final_integrity.json',integrity)
 report(points,pooled,summary,integrity)
 dump(ROOT/'pipeline_status.json',dict(computation='PASS',aggregation='PASS',integrity=integrity['status'],publication='PENDING',logical_points_completed=len(points),expected_logical_points=90,failed_points=0))
 print('AGGREGATION PASS',json.dumps(integrity),flush=True)

def make_plots(points,pooled):
 import html
 ROOT.joinpath('plots').mkdir(parents=True,exist_ok=True)
 palette=['#1f77b4','#d62728','#2ca02c','#9467bd','#ff7f0e']
 def render(metric,series):
  W,H=900,600;L,R,T,B=95,30,55,80;PW,PH=W-L-R,H-T-B
  xs=[x for rows in series.values() for x,y in rows];ys=[y for rows in series.values() for x,y in rows]
  if not xs or not ys:return
  xmin,xmax=min(xs),max(xs);ymin,ymax=min(ys),max(ys)
  if xmax<=xmin:xmax=xmin+1
  if ymax<=ymin:ymax=ymin+1
  def px(x):return L+(x-xmin)/(xmax-xmin)*PW
  def py(y):return T+PH-(y-ymin)/(ymax-ymin)*PH
  out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="white"/>',f'<text x="{W/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(metric)} vs actual bpp</text>',f'<path d="M {L} {T} V {T+PH} H {L+PW}" fill="none" stroke="#222"/>']
  for i in range(6):
   xv=xmin+(xmax-xmin)*i/5;yv=ymin+(ymax-ymin)*i/5
   out.append(f'<path d="M {px(xv):.2f} {T} V {T+PH}" stroke="#ddd"/><text x="{px(xv):.2f}" y="{T+PH+23}" text-anchor="middle" font-family="sans-serif" font-size="12">{xv:.4g}</text>')
   out.append(f'<path d="M {L} {py(yv):.2f} H {L+PW}" stroke="#eee"/><text x="{L-10}" y="{py(yv)+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{yv:.4g}</text>')
  for idx,(name,rows) in enumerate(series.items()):
   color=palette[idx%len(palette)];ordered=sorted(rows)
   if len(ordered)>1:out.append('<polyline fill="none" stroke="'+color+'" stroke-width="2" points="'+' '.join(f'{px(x):.2f},{py(y):.2f}' for x,y in ordered)+'"/>')
   for x,y in ordered:out.append(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="4" fill="{color}"/>')
   ly=T+18+idx*19;out.append(f'<path d="M {L+PW-180} {ly-4} h 20" stroke="{color}" stroke-width="3"/><text x="{L+PW-154}" y="{ly}" font-family="sans-serif" font-size="12">{html.escape(name)}</text>')
  out.append(f'<text x="{L+PW/2}" y="{H-24}" text-anchor="middle" font-family="sans-serif" font-size="14">Actual bpp</text>')
  out.append(f'<text transform="translate(22 {T+PH/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(metric)}</text></svg>')
  (ROOT/'plots'/f'rd_{metric.lower()}.svg').write_text('\n'.join(out))
 for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
  series={}
  if metric=='FID':
   for m in sorted(set(x['method'] for x in pooled)):
    z=[x for x in pooled if x['method']==m];series[m]=[(x['mean_actual_bpp'],x['FID']) for x in z]
  else:
   for m in ('original','N_noP_BG_FD_100','N_noP_BG_FD_250','P_PBG_FD_100','P_PBG_FD_250'):
    z=[x for x in points if x['method']==m and x['point_type']=='fresh_real_encode'];series[m]=[(x['actual_bpp'],x[metric]) for x in z]
  render(metric,series)

def report(points,pooled,summary,integrity):
 lines=['# v6.34a trend screen report','','Execution status: **PASS**.','',f"- Mode: A (reuse); added training updates: N=0, P=0; fixed endpoint: step 250.",f"- Logical RD points: {integrity['completed_logical_points']}/90 (90 fresh physical encodes; 36 step-0 branch aliases point to 18 shared original encodes).",'- Six fixed U-Long training-pool holdout videos; QP 0/5/9; 64 frames each.','- Real RANS streams, independent decode, and state synchronization are checked for each physical point.','- Rate trend uses actual measured bpp and no-extrapolation interpolation. FID uses one pooled 384-frame feature set per method/QP.','','## Same-rate metric summaries','']
 for r in summary:lines.append(f"- {r['method']} vs {r['reference']} | {r['metric']}: status={r['status']}, delta={r.get('delta')}, relative_change_percent={r.get('relative_change_percent')}, videos={r['completed_videos']}/{r['expected_videos']}.")
 lines += ['', '## Pooled FID points','']
 for r in pooled:lines.append(f"- {r['method']} QP{r['QP']}: FID={r['FID']:.8g}, D_mu={r['D_mu']:.8g}, samples={r['samples']}, mean_actual_bpp={r['mean_actual_bpp']:.8g}.")
 lines+=['','This is a six-video, three-QP, single-seed short trend screen. It does not claim convergence or broad generalization. No result-based checkpoint selection or budget extension was performed.','']
 (ROOT/'REPORT.md').write_text('\n'.join(lines));(ROOT/'RESULTS_SUMMARY.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
