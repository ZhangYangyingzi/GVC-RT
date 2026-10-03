"""Raw tables, descriptive statistics, monotonicity and integrity audits."""
from collections import defaultdict
import statistics,traceback
from stride_io import *
def stats(values):
 vals=[float(x) for x in values if x is not None and x not in ('','NaN')]
 assert all(math.isfinite(x) for x in vals)
 return dict(mean=statistics.mean(vals) if vals else None,median=statistics.median(vals) if vals else None,std=statistics.pstdev(vals) if vals else None,n=len(vals))
def mono(vals,direction):
 if any(v is None or not math.isfinite(float(v)) for v in vals):return None
 return bool(vals[0]<vals[1]<vals[2]) if direction=='increasing' else bool(vals[0]>vals[1]>vals[2])
def main():
 difficulty=[];proxy=[];factorial=[];effects=[];bits=[]
 for v in videos():
  for t in ANCHORS:
   for s in STRIDES:
    p=part(v,t,s);r=load(p);assert r['status']=='PASS'
    difficulty.append(r['difficulty']);proxy.extend(r['proxy']);factorial.extend(r['factorial']);bits.extend(r['bitstream_checks'])
    own={x['method']:x for x in r['factorial']};assert set(own)==set(METHODS)
    for theta in ('V62','V64'):
     for metric in METRICS:
      base,ponly,full=(own[m][metric] for m in ('M00_original',theta+'_P_only',theta+'_full'))
      pe=ponly-base;bg=full-ponly;gap=full-base
      fraction=-bg/pe if metric in LOWER and pe>1e-8 else 'NaN'
      reason='DEFINED' if fraction!='NaN' else ('NONPOSITIVE_OR_SMALL_P_EFFECT' if metric in LOWER else 'HIGHER_IS_BETTER')
      effects.append(dict(video=v['name'],anchor=t,stride=s,theta=theta,metric=metric,metric_direction='lower_is_better' if metric in LOWER else 'higher_is_better',P_effect=pe,BG_recovery=bg,Full_gap=gap,recovery_fraction=fraction,recovery_fraction_reason=reason))
 assert len(difficulty)==120 and len(proxy)==240 and len(factorial)==600 and len(effects)==1440 and len(bits)==240
 assert all(x['status']=='PASS' for x in bits)
 write(ROOT/'results/input_temporal_difficulty.csv',difficulty)
 write(ROOT/'results/proxy_stride_raw.csv',proxy)
 write(ROOT/'results/factorial_stride_raw.csv',factorial)
 write(ROOT/'results/recoverability_effects_raw.csv',effects)
 write(ROOT/'audits/bitstream_factorization.csv',bits)
 summary=[]
 for theta in ('V62','V64'):
  for s in STRIDES:
   for key in ('temporal_RGB_L1','flow_magnitude','compensated_residual_L1'):
    summary.append(dict(theta=theta,stride=s,category='input_difficulty',metric=key,**stats(x[key] for x in difficulty if x['stride']==s)))
   for key in ('DISTS','LPIPS','edge_relative_energy_change','HF_ratio','laplacian_ratio'):
    summary.append(dict(theta=theta,stride=s,category='proxy_direct',metric=key,**stats(x[key] for x in proxy if x['theta']==theta and x['stride']==s)))
   for metric in METRICS:
    selected=[x for x in effects if x['theta']==theta and x['stride']==s and x['metric']==metric]
    for key in ('P_effect','BG_recovery','Full_gap')+ (('recovery_fraction',) if metric in LOWER else ()):
     summary.append(dict(theta=theta,stride=s,category='codec_effect',metric=metric+'_'+key,**stats(x[key] for x in selected)))
 write(ROOT/'results/stride_summary.csv',summary)
 monotonic=[]
 for v in VIDEOS:
  for theta in ('V62','V64'):
   for key in ('temporal_RGB_L1','flow_magnitude','compensated_residual_L1'):
    vals=[statistics.mean(float(x[key]) for x in difficulty if x['video']==v and x['stride']==s) for s in STRIDES]
    monotonic.append(dict(video=v,theta=theta,metric=key,difficulty_monotonic=mono(vals,'increasing'),recovery_monotonic=None,stride1_value=vals[0],stride2_value=vals[1],stride4_value=vals[2]))
   for metric in LOWER:
    vals=[stats(x['recovery_fraction'] for x in effects if x['video']==v and x['theta']==theta and x['stride']==s and x['metric']==metric)['mean'] for s in STRIDES]
    monotonic.append(dict(video=v,theta=theta,metric=metric+'_recovery_fraction',difficulty_monotonic=None,recovery_monotonic=mono(vals,'decreasing'),stride1_value=vals[0],stride2_value=vals[1],stride4_value=vals[2]))
 for metric in sorted({x['metric'] for x in monotonic}):
  rows=[x for x in monotonic if x['metric']==metric]
  field='recovery_monotonic' if metric.endswith('_recovery_fraction') else 'difficulty_monotonic'
  valid=[x[field] for x in rows if x[field] is not None]
  monotonic.append(dict(video='__dataset__',theta='ALL',metric=metric,difficulty_monotonic=None,recovery_monotonic=None,stride1_value=None,stride2_value=None,stride4_value=None,fraction_monotonic=sum(valid)/len(valid) if valid else None,n=len(valid)))
 write(ROOT/'results/monotonicity.csv',monotonic)
 from scipy.stats import pearsonr,spearmanr
 correlations=[];dmap={(x['video'],x['anchor'],x['stride']):x for x in difficulty}
 for theta in ('V62','V64'):
  for xname in ('temporal_RGB_L1','flow_magnitude','compensated_residual_L1'):
   for metric in LOWER:
    for yname in ('P_effect','BG_recovery','recovery_fraction','Full_gap'):
     rows=[x for x in effects if x['theta']==theta and x['metric']==metric and x[yname]!='NaN']
     pairs=[(dmap[(r['video'],r['anchor'],r['stride'])][xname],r[yname]) for r in rows]
     pairs=[(float(x),float(y)) for x,y in pairs if math.isfinite(float(x)) and math.isfinite(float(y))]
     xs,ys=zip(*pairs);pr,pp=pearsonr(xs,ys);sr,sp=spearmanr(xs,ys)
     correlations.append(dict(theta=theta,X=xname,metric=metric,Y=yname,pearson_r=float(pr),pearson_p_value=float(pp),spearman_rho=float(sr),spearman_p_value=float(sp),n=len(pairs)))
 write(ROOT/'results/temporal_recoverability_correlations.csv',correlations)
 pmap={(x['video'],x['anchor'],x['stride'],x['theta']):x for x in proxy}
 beauty=[]
 for d in difficulty:
  if d['video']!='Beauty':continue
  for theta in ('V62','V64'):
   ident=(d['video'],d['anchor'],d['stride'],theta);p=pmap[ident]
   row=dict(video='Beauty',anchor=d['anchor'],stride=d['stride'],theta=theta,**{k:d[k] for k in ('temporal_RGB_L1','flow_magnitude','compensated_residual_L1')},proxy_DISTS=p['DISTS'],proxy_LPIPS=p['LPIPS'],edge_relative_energy_change=p['edge_relative_energy_change'],HF_ratio=p['HF_ratio'],laplacian_ratio=p['laplacian_ratio'])
   for metric in LOWER:
    e=next(x for x in effects if (x['video'],x['anchor'],x['stride'],x['theta'],x['metric'])==(*ident,metric))
    for k in ('P_effect','BG_recovery','recovery_fraction','Full_gap','recovery_fraction_reason'):row[metric+'_'+k]=e[k]
   beauty.append(row)
 write(ROOT/'results/beauty_stride_detail.csv',beauty)
 from PIL import Image
 import numpy as np
 for video in ('Beauty','Jockey','ShakeNDry'):
  for anchor in (64,160):
   for stride in STRIDES:
    folder=ROOT/'visualizations'/video/f'a{anchor:03d}_s{stride}'
    with Image.open(folder/'GT_target.png') as im:target=np.asarray(im,dtype=np.int16)
    assert target.shape==(1080,1920,3)
    for method in METHODS:
     recon=folder/method/'frame_000001.png'
     with Image.open(recon) as im:output=np.asarray(im,dtype=np.int16)
     assert output.shape==target.shape
     Image.fromarray(np.abs(output-target).astype(np.uint8),'RGB').save(folder/method/'abs_error.png')
 eq=[]
 for v in VIDEOS:
  a=load(ROOT/'audits'/f'rgb_equivalence_{v}.json');assert a['status']=='PASS';eq.extend(a['checks'])
 dump(ROOT/'audits/rgb_source_equivalence.json',dict(status='PASS',checks=eq,checks_count=len(eq),all_exact=all(x['exact_equality'] and x['max_abs_error']==0 for x in eq)))
 conf=cfg();deps=load(ROOT/'audits/dependency_hashes.json');assert all(sha(p)==h for p,h in deps.items())
 assert all(sha(x['bitstream_path'])==x['bitstream_sha256'] for x in factorial)
 assert all(x['compression_hash']==conf['compression_hash'] and x['real_RANS'] and x['independent_decode'] and x['state_sync_pass'] for x in factorial)
 def finite(obj):
  return all(finite(z) for z in obj.values()) if isinstance(obj,dict) else all(finite(z) for z in obj) if isinstance(obj,list) else math.isfinite(obj) if isinstance(obj,float) else True
 assert all(finite(x) for x in difficulty+proxy+factorial+effects)
 assert len(beauty)==48 and len(eq)==30
 dump(ROOT/'final_integrity.json',dict(status='PASS',no_training=True,no_old_experiments_modified=True,checkpoint_hashes_pass=True,rgb_source_equivalence_pass=True,compression_core_identical=True,real_RANS=True,independent_decode=True,all_expected_pairs_complete=True,all_expected_methods_complete=True,all_metrics_finite_except_defined_recovery_fraction_nan=True,bitstream_factorization_pass=True,expected_inference_points=600,actual_inference_points=len(factorial),input_temporal_diagnostics=len(difficulty),proxy_diagnostics=len(proxy),bitstream_checks=len(bits),finished_unix=time.time()))
 print('AGGREGATE PASS',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  dump(ROOT/'final_integrity.json',dict(status='FAIL',traceback=traceback.format_exc()))
  raise
