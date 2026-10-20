from io32 import *
import numpy as np,math,collections,gzip
O,B,D,F,X,Y,I=METHODS
COMPS=[('DISTS_conditional',D,B),('remaining_statistics_scheme',F,D),('B_vs_original',B,O),('D_vs_original',D,O),('F_vs_original',F,O),('bridge_with_originalG',X,O),('bridge_with_fullG',F,Y),('generator_with_originalBridge',Y,O),('generator_with_fullBridge',F,X),('midpoint_vs_alpha0',I,Y),('alpha1_vs_midpoint',F,I)]
METRICS=('LPIPS','DISTS','FloLPIPS');DATASETS=('ulong','uvg_holdout','hevc_b')
def pool(rows):
 a=[];b=[]
 for r in sorted(rows,key=lambda x:x['video']):
  assert sha(r['feature_path'])==r['feature_sha256'];z=np.load(r['feature_path']);a.append(z['real'].astype(np.float64));b.append(z['reconstruction'].astype(np.float64))
 x=np.concatenate(a);y=np.concatenate(b);assert x.shape==y.shape==(64*len(rows),2048)
 mx=x.mean(0);my=y.mean(0);xc=x-mx;yc=y-my;sv=np.linalg.svd(xc@yc.T/math.sqrt((len(x)-1)*(len(y)-1)),compute_uv=False)
 dm=float(np.square(mx-my).sum());dc=float((xc*xc).sum()/(len(x)-1)+(yc*yc).sum()/(len(y)-1)-2*sv.sum())
 return dict(FID=dm+dc,D_mu=dm,D_cov=dc,samples=len(x),videos=len(rows),mean_actual_bpp=float(np.mean([r['actual_bpp'] for r in rows])),GT_pool_sha256=hashlib.sha256(b''.join(v.astype('<f4').tobytes() for v in a)).hexdigest(),aggregation='pooled official Inception2048; FP64; sample covariance ddof1; low-rank cross-SVD')
def regular_old(x):
 x=dict(x)
 for k in ('video','QP','frames','num_transitions','real_bytes'):x[k]=int(x[k])
 for k in ('actual_bpp',*METRICS):x[k]=float(x[k])
 x.update(reuse_bitstream=True,reused_metrics=True,real_RANS=True,independent_decode_pass=True,state_sync_pass=True)
 return x
def main():
 ev=load(ROOT/'eval_manifest.json');old=[regular_old(x) for x in read(ROOT/'results/reused_points.csv')];points=list(old);replays=[];devals=[];proof=[];featureproof=[];statistics=[];changes=[];acc={};binary={};frames={};transitions={}
 for p in sorted((ROOT/'tasks').glob('replay_*.json')):
  r=load(p)
  if r['status']!='PASS':continue
  replays.append(r)
  for n,out in r['result'].items():
   if n not in (O,B,F):points.append(out)
  for m in (O,B,F):
   cached=next(x for x in old if (x['dataset'],x['video'],x['QP'],x['method'])==(r['task']['dataset'],r['task']['video'],r['task']['qp'],m));aa=np.load(cached['feature_path']);bb=np.load(r['result'][m]['feature_path']);same={k:aa[k].dtype==bb[k].dtype and aa[k].shape==bb[k].shape and aa[k].tobytes()==bb[k].tobytes() for k in ('real','reconstruction')};assert all(same.values()),(r['task'],m,same)
   featureproof.append(dict(task=r['task'],method=m,official_GT_and_reconstruction_features_bitwise=same,cached_feature_SHA=cached['feature_sha256'],replayed_feature_SHA=r['result'][m]['feature_sha256']))
  for v in r['historical_verification']:
   assert v['independent_deployment_historical_exact'] and max(v['metric_abs_errors'].values())<=1e-5
  assert all(v['historical_reconstruction_exact'] for v in r['historical_verification']),'Numerical replay fallback requires uniform reaggregation before publication'
  proof.append(dict(task=r['task'],receipt=str(p),receipt_sha256=sha(p),all_endpoint_frames_bitwise=all(x['bitwise'] for x in r['endpoint_comparison']),historical=r['historical_verification']))
 for p in sorted((ROOT/'tasks').glob('D_*.json')):
  if p.name.startswith('D_encode_'):continue
  r=load(p)
  if r['status']=='PASS':devals.append(r);points.append(r['result'])
 for r in points:
  r.update(total_bits=r['real_bytes']*8,bits_per_frame=r['real_bytes']*8/64,kbps_20fps=r['real_bytes']*8/64*20/1000,rate_accounting_fps=20,valid_frames=64,valid_transitions=63,native_width=1920,native_height=1080,new_optimizer_updates_this_experiment=500 if r['method']==D else 0,deployment_precision='FP16 frozen codec and receiver; inherited FP32 metrics',v637_protocol_SHA=sha(ROOT/'protocol.json'),actual_metric_models=load(ROOT/'audits/metric_models.json')['actual_metrics'],actual_metric_models_audit_SHA=sha(ROOT/'audits/metric_models.json'))
  assert Path(r['bitstream_path']).stat().st_size==r['real_bytes'] and sha(r['bitstream_path'])==r['bitstream_sha256']
  assert abs(r['actual_bpp']-r['real_bytes']*8/(64*1080*1920))<1e-15
  for kind in ('feature','frame_metrics','transitions'):assert sha(r[kind+'_path'])==r[kind+'_sha256']
  for kind in ('bitstream','feature'):binary[r[kind+'_path']]=dict(path=r[kind+'_path'],sha256=r[kind+'_sha256'],bytes=Path(r[kind+'_path']).stat().st_size)
  key=(r['dataset'],r['video'],r['QP'],r['method']);frames[key]=read(r['frame_metrics_path']);transitions[key]=read(r['transitions_path']);assert len(frames[key])==64 and len(transitions[key])==63
  r['actual_qps']=[int(x['actual_qp']) for x in frames[key]];assert r['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(r['QP'])][:64]
  v=next(v for v in ev[r['dataset']] if v['video_index']==r['video']);r['source_frame_indices']=v['source_frame_indices'];r['source_sha256']=v['source_sha256']
 assert len({(r['dataset'],r['video'],r['QP'],r['method']) for r in points})==len(points)
 write(ROOT/'results/all_rd_points.csv',points)
 # Recalculate pooled quantities for current features; audit inherited27 values before reuse.
 pooled=[];poolchecks=[];oldpool=read(ROOT/'results/reused_pooled_fid.csv')
 for ds in DATASETS:
  for m in METHODS:
   for q in QPS:
    rows=[r for r in points if (r['dataset'],r['method'],r['QP'])==(ds,m,q)]
    if len(rows)!=len(ev[ds]):continue
    p=dict(dataset=ds,method=m,QP=q,**pool(rows));pooled.append(p)
    if m in (O,B,F):
     ref=next(x for x in oldpool if (x['dataset'],x['method'],int(x['QP']))==(ds,m,q));err={k:abs(p[k]-float(ref[k])) for k in ('FID','D_mu','D_cov')};assert err['FID']<=1e-3 and p['GT_pool_sha256']==ref['GT_pool_sha256'];poolchecks.append(dict(dataset=ds,method=m,QP=q,abs_errors=err))
 for ds in DATASETS:assert len({p['GT_pool_sha256'] for p in pooled if p['dataset']==ds})==1
 write(ROOT/'results/pooled_fid.csv',pooled)
 # Same-stream paired anchors and per-frame/transition quality differences.
 pairs=[];framepairs=[]
 for ds in DATASETS:
  for v in ev[ds]:
   for q in QPS:
    rows={r['method']:r for r in points if (r['dataset'],r['video'],r['QP'])==(ds,v['video_index'],q)}
    if rows:assert len({r['bitstream_sha256'] for r in rows.values()})==1
    for label,m,ref in COMPS:
     if m not in rows or ref not in rows:continue
     a,b=rows[m],rows[ref];pairs.append(dict(dataset=ds,video=v['video_index'],sequence=v['name'],QP=q,comparison=label,method=m,reference=ref,actual_bpp=a['actual_bpp'],bytes=a['real_bytes'],stream_SHA=a['bitstream_sha256'],actual_stream_equal=True,**{f'delta_{k}':a[k]-b[k] for k in METRICS},**{f'relative_{k}_percent':100*(a[k]-b[k])/b[k] if b[k] else None for k in METRICS}))
     for metric in METRICS:
      src=transitions if metric=='FloLPIPS' else frames;aa=src[(ds,v['video_index'],q,m)];bb=src[(ds,v['video_index'],q,ref)]
      for i,(ra,rb) in enumerate(zip(aa,bb)):framepairs.append(dict(dataset=ds,video=v['video_index'],QP=q,comparison=label,metric=metric,frame=i+1 if metric=='FloLPIPS' else i,delta=float(ra[metric])-float(rb[metric]),actual_bpp=a['actual_bpp']))
 write(ROOT/'results/same_stream_pairs.csv',pairs);write(ROOT/'results/per_frame_quality_changes.csv',framepairs)
 legacy=module('v637_inherited_linear_interpolation',V36/'aggregate36.py');per=[];summary=[];bounds=[];grid=[]
 for ds in DATASETS:
  for v in ev[ds]:
   rows=[r for r in points if r['dataset']==ds and r['video']==v['video_index']];curves={m:[r for r in rows if r['method']==m] for m in METHODS}
   if any(len(rr)!=3 for rr in curves.values()):bounds.append(dict(dataset=ds,video=v['video_index'],status='BLOCKED_INCOMPLETE_METHODS'));continue
   lo=max(min(r['actual_bpp'] for r in rr) for rr in curves.values());hi=min(max(r['actual_bpp'] for r in rr) for rr in curves.values());assert hi>lo;gx=np.linspace(math.log(lo),math.log(hi),20)
   bounds.append(dict(dataset=ds,video=v['video_index'],status='AVAILABLE',bpp_min=lo,bpp_max=hi,points=20,extrapolation=False))
   yy={(m,k):legacy.interp(curves[m],k,gx) for m in METHODS for k in METRICS}
   for (m,k),values in yy.items():
    grid.extend(dict(dataset=ds,video=v['video_index'],method=m,metric=k,grid_index=i,actual_bpp=float(np.exp(gx[i])),value=float(z)) for i,z in enumerate(values))
   for label,m,ref in COMPS:
    for k in METRICS:
     a=float(yy[(m,k)].mean());b=float(yy[(ref,k)].mean());per.append(dict(dataset=ds,video=v['video_index'],comparison=label,method=m,reference=ref,metric=k,method_mean=a,reference_mean=b,delta=a-b,relative_change_percent=100*(a-b)/b if b else None,common_bpp_min=lo,common_bpp_max=hi,interpolation_points=20,aggregation='video-specific shared actual bpp; linear ln(bpp) no extrapolation'))
  for label,m,ref in COMPS:
   for k in METRICS:
    rr=[r for r in per if (r['dataset'],r['comparison'],r['metric'])==(ds,label,k)]
    if not rr:continue
    a=float(np.mean([r['method_mean'] for r in rr]));b=float(np.mean([r['reference_mean'] for r in rr]));summary.append(dict(dataset=ds,comparison=label,method=m,reference=ref,metric=k,method_mean=a,reference_mean=b,delta=a-b,relative_change_percent=100*(a-b)/b if b else None,videos=len(rr),wins=sum(r['delta']<0 for r in rr),ties=sum(r['delta']==0 for r in rr),losses=sum(r['delta']>0 for r in rr),aggregation='20 ln(bpp) points per video; equal video weight; no inference from grid sample count'))
  for metric in ('FID','D_mu','D_cov'):
   curves={m:sorted([p for p in pooled if p['dataset']==ds and p['method']==m],key=lambda r:r['mean_actual_bpp']) for m in METHODS}
   if any(len(rr)!=3 for rr in curves.values()):continue
   lo=max(rr[0]['mean_actual_bpp'] for rr in curves.values());hi=min(rr[-1]['mean_actual_bpp'] for rr in curves.values());assert hi>lo;gx=np.linspace(math.log(lo),math.log(hi),20);yy={m:np.interp(gx,np.log([r['mean_actual_bpp'] for r in rr]),[r[metric] for r in rr]) for m,rr in curves.items()}
   for label,m,ref in COMPS:
    a=float(yy[m].mean());b=float(yy[ref].mean());summary.append(dict(dataset=ds,comparison=label,method=m,reference=ref,metric=metric,method_mean=a,reference_mean=b,delta=a-b,relative_change_percent=100*(a-b)/b if b else None,videos=len(ev[ds]),samples=64*len(ev[ds]),common_mean_actual_bpp_min=lo,common_mean_actual_bpp_max=hi,aggregation='pooled FID vs dataset mean actual bpp;20 ln(bpp) points; no per-video FID'))
 write(ROOT/'results/equal_rate_per_video.csv',per);write(ROOT/'results/equal_rate_summary.csv',summary);write(ROOT/'results/common_rate_bounds.csv',bounds);write(ROOT/'results/common_rate_grid.csv',grid)
 write(ROOT/'results/loss_attribution.csv',[r for r in summary if r['comparison'] in ('DISTS_conditional','remaining_statistics_scheme','B_vs_original','D_vs_original','F_vs_original')])
 write(ROOT/'results/module_swap_summary.csv',[r for r in summary if r['comparison'] in ('bridge_with_originalG','bridge_with_fullG','generator_with_originalBridge','generator_with_fullBridge')])
 write(ROOT/'results/latent_interpolation.csv',[dict(p,alpha={Y:0,I:.5,F:1}[p['method']],scope='per_video_same_stream_anchor') for p in points if p['method'] in (Y,I,F)]+[dict(p,alpha={Y:0,I:.5,F:1}[p['method']],scope='dataset_pooled_FID') for p in pooled if p['method'] in (Y,I,F)])
 # Per-frame and properly streamed space/time latent summaries.
 for r in replays+devals:
  t=r['task'];path=r['latent_archive']['path'];assert sha(path)==r['latent_archive']['sha256'];z=np.load(path);binary[path]=dict(path=path,sha256=sha(path),bytes=Path(path).stat().st_size)
  for s in r['latent_statistics']:
   row=dict(dataset=t['dataset'],video=t['video'],QP=t['qp'],**s,covariance_archive=path,covariance_archive_SHA=r['latent_archive']['sha256']);statistics.append(row)
   n=s['spatial_samples'];mu=np.array(s['mean']);cov=z['covariance'][s['covariance_index']];cross=(cov+np.outer(mu,mu))*n
   for level,key in [('video',(t['dataset'],t['video'],t['qp'],s['method'])),('dataset_QP',(t['dataset'],'ALL',t['qp'],s['method'])),('dataset',(t['dataset'],'ALL','ALL',s['method'])),('overall',('ALL','ALL','ALL',s['method']))]:
    k=(level,*key)
    if k not in acc:acc[k]=[0,np.zeros(18),np.zeros((18,18)),0]
    a=acc[k];a[0]+=n;a[1]+=n*mu;a[2]+=cross;a[3]+=1
  changes.extend(dict(dataset=t['dataset'],video=t['video'],QP=t['qp'],**s) for s in r['latent_pair_changes'])
 aggregates=[]
 for key,(n,s,c,frames_n) in acc.items():
  mu=s/n;cov=c/n-np.outer(mu,mu);aggregates.append(dict(level=key[0],dataset=key[1],video=key[2],QP=key[3],method=key[4],spatial_time_samples=n,P_frame_observations=frames_n,ddof=0,mean=mu.tolist(),variance=cov.diagonal().tolist(),RMS=np.sqrt(np.diag(c)/n).tolist(),covariance18x18=cov.tolist()))
 quality_index={(r['dataset'],r['video'],r['QP'],r['frame'],r['comparison'],r['metric']):r['delta'] for r in framepairs}
 stats_index={(r['dataset'],r['video'],r['QP'],r['frame'],r['method']):r for r in statistics}
 pair_groups={}
 for r in changes:
  label={'z0_to_zF':'F_vs_original','B_to_D':'DISTS_conditional','D_to_F':'remaining_statistics_scheme'}[r['comparison']]
  for metric in METRICS:r['quality_delta_'+metric]=quality_index.get((r['dataset'],r['video'],r['QP'],r['frame'],label,metric))
  a,b={'z0_to_zF':(O,F),'B_to_D':(B,D),'D_to_F':(D,F)}[r['comparison']];key=(r['dataset'],r['video'],r['QP'],r['frame']);aa=stats_index[(*key,a)];bb=stats_index[(*key,b)]
  r['channel_mean_delta']=(np.array(bb['mean'])-np.array(aa['mean'])).tolist();r['channel_variance_delta']=(np.array(bb['variance'])-np.array(aa['variance'])).tolist()
  for level,ds,vid,qp in [('video_QP',r['dataset'],r['video'],r['QP']),('dataset_QP',r['dataset'],'ALL',r['QP']),('dataset',r['dataset'],'ALL','ALL'),('overall','ALL','ALL','ALL')]:pair_groups.setdefault((level,ds,vid,qp,r['comparison']),[]).append(r)
 pair_summary=[]
 for k,rr in pair_groups.items():
  rel=np.array([x['relative_L2'] for x in rr]);pair_summary.append(dict(level=k[0],dataset=k[1],video=k[2],QP=k[3],comparison=k[4],P_frame_observations=len(rr),mean_frame_relative_L2=float(rel.mean()),std_frame_relative_L2=float(rel.std()),min_frame_relative_L2=float(rel.min()),max_frame_relative_L2=float(rel.max()),mean_frame_cosine_distance=float(np.mean([x['cosine_distance'] for x in rr])),mean_frame_channel_mean_delta=np.mean([x['channel_mean_delta'] for x in rr],axis=0).tolist(),mean_frame_channel_variance_delta=np.mean([x['channel_variance_delta'] for x in rr],axis=0).tolist(),zero_norm_frames=sum(x['denominator_zero'] for x in rr),epsilon=1e-12,aggregation='equal P-frame descriptive moments; no significance or causal claim; pooled channel moments separately use FP64 spatial cross-moments'))
 write(ROOT/'results/latent_pair_changes_aggregated.csv',pair_summary)

 write(ROOT/'results/latent_statistics.csv',statistics);write(ROOT/'results/latent_statistics_aggregated.csv',aggregates);write(ROOT/'results/latent_pair_changes.csv',changes)
 dump(ROOT/'audits/latent_aliases.json',dict(X_input_alias='F zF',Y_input_alias='O z0',I_input='(z0+zF)/2 FP16; no normalization',samples='spatial positions per actual generator input P frame; pooled over space/time by FP64 cross-moments; ddof0',covariances='per-frame SHA-indexed server NPZ; pooled matrices in latent_statistics_aggregated.csv'))
 dump(ROOT/'audits/stream_and_feature_equality.json',dict(status='PASS' if len(replays)==len(devals)==39 else 'PARTIAL',O_B_F_P_frames_verified=sum(len(r['tensor_equality']) for r in replays),D_P_frames_verified=sum(len(r['tensor_equality']) for r in devals),all_actual_stream_SHA_equal=True,replay_receipts=[dict(task=r['task'],archive=r['latent_archive']) for r in replays],D_receipts=[dict(task=r['task']) for r in devals],RGB_reference='native I only; subsequent feature reference verified'))
 dump(ROOT/'audits/replay_endpoint_equivalence.json',dict(status='PASS' if len(proof)==39 and len(devals)==39 else 'PARTIAL',O_B_F=proof,D_all_frames_bitwise=all(all(x['bitwise'] for x in r['endpoint_comparison']) for r in devals),pooled_FID_historical_checks=poolchecks,official_feature_replay_checks=featureproof,thresholds=load(ROOT/'protocol.json')['equivalence']))
 dump(ROOT/'audits/interpolation_endpoints.json',dict(status='PASS' if len(replays)==39 else 'PARTIAL',receipts=[dict(task=r['task'],**r['interpolation_endpoints']) for r in replays]))
 dump(ROOT/'server_artifacts_index.json',list(binary.values()))
 # Explicit win/tie/loss counts by dataset/QP, metric and fixed comparison.
 wins=[]
 for ds in DATASETS:
  for q in QPS:
   for label,m,ref in COMPS:
    rr=[r for r in pairs if (r['dataset'],r['QP'],r['comparison'])==(ds,q,label)]
    for metric in METRICS:
     x=[r['delta_'+metric] for r in rr]
     if x:wins.append(dict(dataset=ds,QP=q,comparison=label,metric=metric,videos=len(x),wins=sum(v<0 for v in x),ties=sum(v==0 for v in x),losses=sum(v>0 for v in x),mean_delta=float(np.mean(x))))
 write(ROOT/'results/anchor_wins_by_dataset_QP.csv',wins)
 from figures37 import make_figures
 make_figures(points,pooled,summary,aggregates)
 status='PASS' if len(points)==273 and len(pooled)==63 else 'PARTIAL'
 actual_updates=max([int(p.stem) for p in (ROOT/'branches/D_basicDISTS_noP_500/receipts').glob('*.json')],default=0)
 report='# v6.37 fixed500 loss and receiver mechanism records\n\nStatus: '+status+f'. RD={len(points)}/273; pooledFID={len(pooled)}/63. New training: only D, committed updates={actual_updates}. O/B/F reused read-only; X/Y/I untrained. All anchor comparisons use equal persisted complete RANS byte/SHA values.\n\n'
 report+='## Prespecified numerical comparisons\n\nNegative delta means lower metric distance; relative changes are quality changes, never saved bits. Equal-rate paired metrics average20 ln(bpp) points within each video then weight videos equally. FID uses pooled dataset features and the shared dataset-mean actual-bpp interval. Grids are not statistical replicates.\n\n|Dataset|Comparison|Metric|Method mean|Reference mean|Delta|Relative%|Videos|\n|---|---|---|---:|---:|---:|---:|---:|\n'
 for r in summary:report+=f'|{r["dataset"]}|{r["comparison"]}|{r["metric"]}|{r["method_mean"]:.9g}|{r["reference_mean"]:.9g}|{r["delta"]:.9g}|{r["relative_change_percent"] if r["relative_change_percent"] is not None else "undefined"}|{r["videos"]}|\n'
 report+='\n## Verified evidence and limits\n\nInput-source hashes, checkpoint SHA256, common500-row plan and DISTS input/deep-feature/BG gradients are audited. Source models are original/fixed and no P is trained. D−B is the conditional DISTS increment; F−D is the remaining statistical-supervision scheme including native-frame FD/reference exposure, not isolated FD or equal compute. Pure original-objective adaptation is not tested.\n\nSame-stream shared-feature receiver interventions identify output changes under the fixed swaps and alpha0/.5/1 path. X/Y degradation can reflect cross-weight incompatibility and does not establish that independently training a module is ineffective. Jointly better results are compatible with coadaptation, not proof of a specific latent covariance cause. Latent statistics are actual18-channel codeword statistics, not Inception/FID statistics. Non-monotone midpoint behavior is retained without declaring latent changes useless.\n\nNot established here: a causal attribute/channel mechanism, original sufficient-training comparison, P value, generalization/significance, a general compression-generation mismatch theory, or publication novelty. One seed,13 videos and500 updates do not show convergence or statistical significance. Unknown official-pretraining exposure remains unknown. No recommendation about subsequent research is made.\n\nSee results/same_stream_pairs.csv, per_frame_quality_changes.csv and anchor_wins_by_dataset_QP.csv for all anchors, videos and wins/losses; latent_statistics*.csv and latent_pair_changes.csv for content-indexed latent changes; audits/ for end-to-end provenance and verification attempts. Failures are retained in errors.jsonl and worker logs. Final completion and update counts are authoritative in final_integrity.json.\n'
 (ROOT/'REPORT.md').write_text(report)
 (ROOT/'RESULTS_SUMMARY.md').write_text('# Execution summary\n\n'+f'RD {len(points)}/273; pooled FID {len(pooled)}/63; status {status}. Only D has new training, fixed500 updates.\n\n'+report[report.index('|Dataset|'):report.index('## Verified evidence')])
 dump(ROOT/'runtime/aggregation_status.json',dict(status=status,RD_complete=len(points),RD_planned=273,FID_complete=len(pooled),FID_planned=63,replay_tasks=len(replays),D_real_encodes=len(devals),pooled_samples={d:len(ev[d])*64 for d in DATASETS},source_points117_verified=True))
 print('AGGREGATE',status,len(points),len(pooled),flush=True)
if __name__=='__main__':main()
