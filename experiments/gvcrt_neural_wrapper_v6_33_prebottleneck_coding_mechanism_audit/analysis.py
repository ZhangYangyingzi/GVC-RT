"""Mechanical preregistered mechanism analysis; video-cluster CIs, no model selection."""
import gzip,math,collections
from audit_io import *
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from scipy.interpolate import PchipInterpolator
METRICS=['cosine','LPIPS','DISTS','FloLPIPS']
def save(name,df):
 p=ROOT/name;tmp=p.with_name(p.name+'.tmp');df.to_csv(tmp,index=False,compression='gzip' if name.endswith('.gz') else None);tmp.replace(p)
def boot(df,column):
 clusters=[g[column].dropna().to_numpy(float) for _,g in df.groupby(['dataset','video'])];clusters=[c for c in clusters if len(c)]
 if not clusters:return dict(n_videos=0,n_pairs=0,mean=None,median=None,mean_ci_low=None,mean_ci_high=None,median_ci_low=None,median_ci_high=None)
 a=np.full((len(clusters),max(map(len,clusters))),np.nan)
 for i,c in enumerate(clusters):a[i,:len(c)]=c
 rng=np.random.default_rng(20261015);ix=rng.integers(len(a),size=(2000,len(a)));samples=a[ix];means=np.nanmean(a,axis=1)[ix].mean(1);med=np.nanmedian(samples.reshape(2000,-1),axis=1)
 return dict(n_videos=len(a),n_pairs=sum(map(len,clusters)),mean=float(np.nanmean(a,axis=1).mean()),median=float(np.nanmedian(a)),mean_ci_low=float(np.quantile(means,.025)),mean_ci_high=float(np.quantile(means,.975)),median_ci_low=float(np.quantile(med,.025)),median_ci_high=float(np.quantile(med,.975)),bootstrap='2000 video-cluster resamples; mean equal video weights; median pooled pairs',single_video_CI_degenerate=len(a)==1)
def main():
 from worker import valid,key
 protocol=load(ROOT/'protocol.json');datasets=load(ROOT/'dataset_manifest.json')['datasets'];points=[];frames=[];quant=[];entropy=[];receipt_index=[];missing=[]
 for d,vs in datasets.items():
  for v in vs:
   for q in range(10):
    for m in protocol['methods']:
     t=dict(dataset=d,video=v['video_index'],qp=q,method=m)
     if not valid(t,deep=True):missing.append(t);continue
     p=ROOT/'parts'/key(t);r=load(p/'point.json');meta=dict(dataset=d,video=v['name'],video_index=t['video'],QP=q,method=m,alpha=r['alpha'])
     points.append(dict(**meta,actual_bpp=r['bpp'],P_frame_bpp=r['P_frame_bpp'],payload_bpp=r['payload_bpp'],P_payload_bpp=r['P_payload_bpp'],payload_bits_per_frame=r['payload_bits_per_frame'],actual_total_stream_bytes=r['real_bytes'],kbps_20fps=r['bpp']*1920*1080*20/1000,bitstream_sha256=r['bitstream_sha256'],independent_decode_PASS=r['independent_decode_pass'],model_unchanged_PASS=r['model_hashes_unchanged'],**r['metrics_P'],**r['quantization_P'],cdf_entry_bits_per_P_frame=r['cdf_entry_bits_per_P_frame']))
     for name,dest in [('frames.csv',frames),('quantization.csv',quant),('entropy.csv',entropy)]:
      for row in csv.DictReader((p/name).open()):dest.append(dict(row,**meta))
     receipt_index.append(r)
 assert not missing,missing[:20]
 df=pd.DataFrame(points);save('all_points.csv.gz',df);save('per_frame.csv.gz',pd.DataFrame(frames));save('quantization_diagnostics.csv',pd.DataFrame(quant));save('entropy_diagnostics.csv',pd.DataFrame(entropy))
 with gzip.open(ROOT/'point_receipts.jsonl.gz','wt') as out:
  for r in receipt_index:out.write(json.dumps(r,allow_nan=False)+'\n')
 controls=df[df.method=='B1_alpha0'];one=df[df.method=='B1_alpha100'];original=df[df.method=='Original'];join=['dataset','video','QP']
 delta=df.merge(controls[join+['actual_bpp','cosine','LPIPS','DISTS','FloLPIPS']],on=join,suffixes=('','_alpha0')).merge(one[join+['actual_bpp','cosine']],on=join,suffixes=('','_alpha1')).merge(original[join+['cosine']],on=join,suffixes=('','_Original'))
 for name,ref in [('alpha0','actual_bpp_alpha0'),('alpha1','actual_bpp_alpha1')]:delta['delta_bpp_vs_'+name]=delta.actual_bpp-delta[ref]
 delta['relative_delta_bpp_vs_alpha0']=delta.delta_bpp_vs_alpha0/delta.actual_bpp_alpha0
 for name in ('alpha0','alpha1','Original'):delta['delta_cos_vs_'+name]=delta.cosine-delta['cosine_'+name]
 for metric in ('LPIPS','DISTS','FloLPIPS'):delta['delta_'+metric+'_vs_alpha0']=delta[metric]-delta[metric+'_alpha0']
 bands=controls[join+['actual_bpp']].copy();bands['rate_band']=bands.groupby('dataset')['actual_bpp'].transform(lambda x:pd.qcut(x.rank(method='first'),3,labels=['low','mid','high']).astype(str));delta=delta.merge(bands[join+['rate_band']],on=join)
 save('same_qp_rate_deltas.csv',delta);save('source_change_vs_rate.csv',delta[[*join,'method','alpha','rate_band','RGB_residual_L1','RGB_residual_RMSE','RGB_residual_max_abs','actual_bpp','delta_bpp_vs_alpha0','cosine','delta_cos_vs_alpha0','LPIPS','DISTS','FloLPIPS','frequency_low_energy_delta','frequency_mid_energy_delta','frequency_high_energy_delta']])
 numeric=['actual_bpp','P_frame_bpp','payload_bpp','actual_total_stream_bytes','cosine','LPIPS','DISTS','FloLPIPS','RGB_residual_L1','RGB_residual_RMSE','quant_error_RMSE','fraction_quantized_symbols_changed','cdf_entry_bits_per_P_frame']
 summaries=[]
 for keys,g in delta.groupby(['dataset','method','QP'],dropna=False):summaries.append(dict(zip(['dataset','method','QP'],keys),n_videos=g.video.nunique(),**{k:float(g[k].mean()) for k in numeric}))
 summary=pd.DataFrame(summaries);save('alpha_summary.csv',summary);save('latent_fidelity_summary.csv',delta[join+['method','alpha','cosine','L1','RMSE','normalized_L2','delta_cos_vs_alpha0','delta_cos_vs_alpha1','delta_cos_vs_Original']]);save('coding_diagnostics.csv',df[join+['method','alpha','actual_bpp','P_frame_bpp','payload_bpp','quant_error_RMSE','fraction_quantized_symbols_changed','prequant_shift_vs_alpha0_relative_L2','skip_fraction','cdf_entry_bits_per_P_frame']])
 trends=[]
 for keys,g in df[df.method!='Original'].groupby(join):
  g=g.sort_values('alpha')
  for metric in ['actual_bpp','cosine','LPIPS','DISTS','FloLPIPS','fraction_quantized_symbols_changed']:
   y=g[metric].to_numpy();rho=float(spearmanr(g.alpha,y).statistic) if len(set(y))>1 else None
   trends.append(dict(zip(join,keys),metric=metric,Spearman=rho,n=len(g),constant_y=len(set(y))==1))
 trend=pd.DataFrame(trends);save('trend_summary.csv',trend)
 b=[];interventions=delta[delta.method!='Original']
 groupings=[([],[((),interventions)]),(['dataset'],interventions.groupby('dataset')),(['dataset','QP'],interventions.groupby(['dataset','QP'])),(['dataset','rate_band'],interventions.groupby(['dataset','rate_band'])),(['dataset','video'],interventions.groupby(['dataset','video']))]
 for cols,groups in groupings:
  for keys,g in groups:
   if not isinstance(keys,tuple):keys=(keys,)
   for m,mg in g.groupby('method'):
    for metric in ('relative_delta_bpp_vs_alpha0','delta_bpp_vs_alpha0','delta_cos_vs_alpha0'):
     b.append(dict(scope='all' if not cols else '/'.join(cols),**dict(zip(cols,keys)),method=m,metric=metric,**boot(mg,metric),fraction_rate_lower=float((mg.delta_bpp_vs_alpha0<0).mean()),fraction_teacher_worse_rate_lower=float(((mg.delta_cos_vs_alpha0>0)&(mg.delta_bpp_vs_alpha0<0)).mean())))
 bootstrap=pd.DataFrame(b);save('bootstrap_summary.csv',bootstrap)
 # System-level comparison: only normally adapted alpha1 vs Original, common measured rate support.
 matched=[]
 for (d,v),g in df.groupby(['dataset','video']):
  a=g[g.method=='B1_alpha100'].sort_values('actual_bpp');o=g[g.method=='Original'].sort_values('actual_bpp');lo=max(a.actual_bpp.min(),o.actual_bpp.min());hi=min(a.actual_bpp.max(),o.actual_bpp.max())
  for metric in ('LPIPS','DISTS','FloLPIPS'):
   row=dict(dataset=d,video=v,metric=metric,method='B1_alpha100',reference='Original',common_bpp_low=lo,common_bpp_high=hi)
   if a.actual_bpp.nunique()!=10 or o.actual_bpp.nunique()!=10 or hi<=lo:row.update(status='unavailable',reason='duplicate rates or no valid common interval',delta=None)
   else:
    grid=np.linspace(np.log(lo),np.log(hi),100);av=PchipInterpolator(np.log(a.actual_bpp),a[metric],extrapolate=False)(grid);ov=PchipInterpolator(np.log(o.actual_bpp),o[metric],extrapolate=False)(grid);row.update(status='available',delta=float((av-ov).mean()),B1_mean=float(av.mean()),Original_mean=float(ov.mean()))
   matched.append(row)
 match=pd.DataFrame(matched);save('matched_rate_B1_vs_Original.csv',match)
 ms=[]
 for d,g in [('all',match),*list(match.groupby('dataset'))]:
  for metric,h in g.groupby('metric'):
   good=h[h.status=='available'];ms.append(dict(dataset_scope=d,metric=metric,expected_videos=len(h),available_videos=len(good),**boot(good,'delta')))
 ms=pd.DataFrame(ms);save('matched_rate_summary.csv',ms)
 primary=delta[delta.method=='B1_alpha100'];global_boot=boot(primary,'relative_delta_bpp_vs_alpha0');changed=float((primary.delta_bpp_vs_alpha0!=0).mean());symbol=float(primary.fraction_quantized_symbols_changed.mean());worse_better=float(((primary.delta_cos_vs_alpha0>0)&(primary.delta_bpp_vs_alpha0<0)).mean());ds=primary.groupby('dataset').relative_delta_bpp_vs_alpha0.mean();same_direction=bool((ds<0).all() or (ds>0).all());improved=ms[(ms.dataset_scope=='all')&(ms.available_videos==ms.expected_videos)&(ms.mean_ci_high<0)]
 strong=bool(same_direction and (global_boot['mean_ci_high']<0 or global_boot['mean_ci_low']>0) and symbol>0 and len(improved)>=2 and worse_better>=.5)
 decision='CASE_A_strong_support' if strong else 'CASE_B_partial' if changed>0 or symbol>0 else 'CASE_C_not_supported'
 dump('mechanism_decision.json',dict(case=decision,complete_points=len(df),expected_points=3220,alpha1_vs_alpha0=global_boot,fraction_rate_changed=changed,fraction_rate_lower=float((primary.delta_bpp_vs_alpha0<0).mean()),fraction_teacher_worse_rate_lower=worse_better,mean_quantized_symbol_change_fraction=symbol,rate_mean_direction_consistent_across_datasets=same_direction,matched_rate_improved_metrics=improved.metric.tolist(),teacher='paper_compatible_teacher_proxy',not_Strong_No_P=True,conclusion='P changes what survives the finite-rate codec rather than merely improving teacher-latent imitation.' if strong else 'P measurably changes codec behavior; a uniform mechanism is not fully identified.' if changed>0 else 'Improved compressibility is not supported by the measured intervention.',limitations=['alpha0 is B1 receiver without source P, not Original and not a fair optimized No-P baseline','alpha !=1 quality is local intervention with unadapted receiver','teacher proxy is not verified original training teacher','CDF entry bits omit escape bypass and z; actual persisted RANS bytes remain primary','conditional features are not a defined RGB prediction; prediction residual unavailable','associations do not establish causal channel attribution','no claim of optimal representation, incorrect latent supervision, or significance beyond reported cluster CIs']))
 from figures import make
 make(df,delta,summary,trend)
 def table(data):
  return data.to_csv(index=False,float_format='%.7g')
 lines=['# v6.33 pre-bottleneck coding mechanism audit','',f'Complete: {len(df)}/3220 evaluation points, 46 videos, 64 causal frames, QP0–9. Optimizer updates: 0.','',f'1. Does P change actual bitstream rate? Fraction of alpha1/control video-QP pairs with changed total bytes/rate: {changed:.6f}.',f'2. Strength/rate: alpha1 vs alpha0 relative bpp mean {global_boot["mean"]:.6g}, median {global_boot["median"]:.6g}; video-cluster mean95% CI [{global_boot["mean_ci_low"]:.6g}, {global_boot["mean_ci_high"]:.6g}]. Fraction lower: {(primary.delta_bpp_vs_alpha0<0).mean():.6f}. Alpha trends are in trend_summary.csv.',f'3. Teacher distance: mean alpha1-minus-alpha0 cosine distance {primary.delta_cos_vs_alpha0.mean():.7g}; relative to paper_compatible_teacher_proxy only. Positive is worse.',f'4. Perceptual quality: alpha sweep values below are local interventions without receiver readaptation. System-level B1_alpha100 versus Original common-rate results follow separately.',f'5. Teacher worse but rate lower fraction: {worse_better:.6f}. This is paired by video/external QP, not an equal-rate quality comparison.',f'6. Quantization/entropy path: mean alpha1/control changed-symbol fraction {symbol:.6g}. Real raw y, dequantized y_hat, actual centered integer symbols and actual RANS CDF entries were hooked; see tensor_semantics.json. CDF entry bits exclude escape bypass and hyperlatent z.',f'7. Registered mechanism decision: {decision}. This tests whether a representation with better prescribed latent fidelity is necessarily more efficient for the fixed finite-rate codec; it does not rank all possible representations.','8. Unavailable/unsupported claims: no exact official training teacher, no fair Strong No-P baseline, no optimal representation claim, no causal channel attribution, no claim that latent supervision is wrong. RGB prediction residual is unavailable because the architecture provides conditional context rather than an explicitly defined source prediction.','', '## Alpha1 versus alpha0 by dataset (same QP; not equal-rate quality)','```csv',table(primary.groupby('dataset')[['relative_delta_bpp_vs_alpha0','delta_cos_vs_alpha0','delta_LPIPS_vs_alpha0','delta_DISTS_vs_alpha0','delta_FloLPIPS_vs_alpha0','fraction_quantized_symbols_changed']].mean().reset_index()),'```','', '## Normal B1 versus Original matched-rate comparison','100 equally spaced log-bpp samples inside each video common measured interval; PCHIP without extrapolation; video-cluster confidence intervals.','```csv',table(ms),'```','', '## Full alpha sweep','P-frame quality/fidelity means; rate includes all 64 frames and all headers. Original is separate. QP summaries remain separate in alpha_summary.csv.','```csv',table(df.groupby(['dataset','method'],sort=False)[['actual_bpp','P_frame_bpp','cosine','LPIPS','DISTS','FloLPIPS']].mean().reset_index()),'```','', 'B1_alpha0 retains the B1 bridge/generator. It is a strict source-side rate/coding intervention control, but its end-to-end quality can include receiver-interface mismatch. This experiment does not exclude insufficient baseline training.','', 'Quantizer boundaries are half-integers only after the actual learned quantization scaling and actual conditional mean centering. The scale<=0.12 skip rule and saturation are recorded separately. y_hat restores conditional means and quantization scale; it is not itself the integer transmitted symbol array. All mechanisms are measured with full causal histories, so changes include propagated changes in codec context.','', 'Teacher targets are computed from original GT and never from P_alpha(x). Native I0 bypasses P at every alpha. Actual total bytes include SPS/headers/I0; P-frame rate uses the 63 P-frame pixels as its denominator. 20fps changes only kbps display. Low/mid/high frequency energy follows the fixed v6.31a bands and is not causal attribution.','', 'Figures are provided as PNG and PDF in figures/. All points have server-side receipts, also bundled in point_receipts.jsonl.gz. Binary streams, full teacher/codec tensors and large logs remain on the server.']
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print('ANALYSIS_REPORT_PASS',len(df),decision,flush=True)
if __name__=='__main__':main()
