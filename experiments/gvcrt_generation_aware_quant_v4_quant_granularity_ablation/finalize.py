"""Full artifact/math/metric audit and descriptive inference-only conclusions."""
from util import *
import math,shutil,collections,itertools
import numpy as np
METRICS=('LPIPS','DISTS','FloLPIPS','PSNR')
def csvrows(path):return list(csv.DictReader(Path(path).open()))
def audit():
 from task import valid
 assert load(P/'run_status.json')['status']=='COMPUTE_COMPLETE'
 assert load(P/'preflight.json')['status']=='PASS' and load(P/'preflight.json')['signature']==signature()
 assert load(P/'cross_mode_audit.json')['status']=='PASS'
 for f,h in load(P/'dependency_manifest.json').items():assert sha(f)==h,('changed historical source',f)
 for f,r in load(P/'weight_manifest.json').items():assert sha(f)==r['sha256'],('changed weights',f)
 rows=[];diags=[];frames=[];trans=[];receipts=[];artifacts={};maps_checked=0
 for mode,source,qp in itertools.product(MODES,SOURCES,(2,7)):
  k=key(mode,source,qp);f=P/'server_artifacts/evaluation'/f'{k}.json';assert valid(f),k;r=load(f)
  assert r['mode']==mode and r['sequence']==source and r['external_qp']==qp and r['step']==300 and r['alpha']==1 and r['training_updates']==0
  assert r['encoder_released'] and r['independent_decode'] and r['all64_reference_latent_qmap_rho_equal'] and r['all_maps_encoder_decoder_equal']
  assert r['encoder_CDF']==r['decoder_CDF']
  assert r['checkpoint_sha256']==load(P/'weight_manifest.json')[str(checkpoint(mode))]['sha256']
  for path,h in r['artifact_hashes'].items():artifacts[path]=dict(path=path,sha256=h,bytes=Path(path).stat().st_size)
  fs=csvrows(P/'server_artifacts/frames'/f'{k}.csv');ts=csvrows(P/'server_artifacts/transitions'/f'{k}.csv');ds=csvrows(P/'server_artifacts/diagnostics'/f'{k}.csv')
  assert len(fs)==64 and len(ts)==63 and len(ds)==63
  assert [int(x['actual_qp']) for x in fs]==[qp+[0,2,0,1,0,1,0,1][i%8] for i in range(64)]
  assert sum(int(x['real_bytes']) for x in fs)==r['real_total_bytes']==r['payload_bytes']+r['header_bytes']
  assert sum(int(x['payload_bytes']) for x in fs[1:])==r['P_payload_bytes']
  assert math.isclose(r['actual_bpp'],r['real_total_bytes']*8/(64*r['height']*r['width']),rel_tol=1e-13)
  assert abs(r['PSNR']+10*math.log10(sum(float(x['pixel_SSE']) for x in fs)/(64*3*r['height']*r['width'])))<1e-9
  for metric in ('LPIPS','DISTS'):assert abs(r[metric]-sum(float(x[metric]) for x in fs)/64)<1e-10
  assert abs(r['FloLPIPS']-sum(float(x['FloLPIPS']) for x in ts)/63)<1e-10
  assert all(math.isfinite(r[m]) for m in METRICS)
  st=load(P/'server_artifacts/states'/f'{k}.json');assert st['encoder']==st['decoder'] and len(st['encoder'])==64 and len(st['maps'])==63
  for i,(entry,d) in enumerate(zip(st['maps'],ds),1):
   with np.load(entry['path']) as a:
    base,q,rho,raw,var,eff=(a[n] for n in ('q_base','qmap','rho','raw_multiplier','variant_multiplier','effective_multiplier'))
    assert base.shape==q.shape==rho.shape==eff.shape and base.shape[1]==128
    assert raw.shape[1]==1 and var.shape==raw.shape
    assert all(np.isfinite(a[n]).all() for n in a.files)
    expected=np.ones_like(raw) if mode in ('A','IDENTITY') else np.broadcast_to(raw.mean(axis=(1,2,3) if mode=='FRAME' else (2,3),keepdims=True),raw.shape) if mode in ('FRAME','CHANNEL') else raw
    assert np.max(np.abs(var-expected))<1e-6,(k,i,'multiplier definition')
    expectq=np.maximum(base.astype(np.float32)*var,.5).astype(base.dtype)
    assert np.array_equal(q,expectq),(k,i,'q clamp/dtype')
    expectrho=(q.astype(np.float32)/base.astype(np.float32)).astype(base.dtype)
    assert np.array_equal(rho,expectrho) and np.array_equal(eff,rho)
    cf=float((base.astype(np.float32)*var<.5).mean());assert abs(cf-float(d['clamp_fraction']))<1e-7
    for j in (0,1):assert 0<=float(d[f'pass{j}_skip_fraction'])<=1
    for n in a.files:assert hashlib.sha256(np.ascontiguousarray(a[n]).tobytes()).hexdigest()==entry['tensor_hashes'][n]
    maps_checked+=1
  if mode in ('A','IDENTITY'):assert r['native_parity']['status']=='PASS' and r['native_parity']['all_64_states_exact']
  for name,data,combined in [('per_frame',fs,frames),('per_transition',ts,trans),('quant_diagnostics',ds,diags)]:
   combined.extend(dict(mode=mode,sequence=source,QP=qp,**{k:v for k,v in x.items() if k not in ('mode','sequence','QP')}) for x in data)
  rows.append({k:v for k,v in r.items() if k not in ('artifact_hashes','native_parity')})
  receipts.append(dict(task=k,status='PASS',frames=64,P_maps=63,native_parity=r['native_parity'],CDF_equal=True,artifact_hashes_verified=len(r['artifact_hashes'])))
  artifacts[str(f)]=dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size)
 assert len(rows)==40 and maps_checked==2520
 table(P/'evaluation.csv',rows);table(P/'per_frame.csv',frames);table(P/'per_transition.csv',trans);table(P/'quant_diagnostics.csv',diags)
 rates=[]
 for r in rows:
  est=r['estimated_P_y_bits']+r['estimated_P_z_bits'];actual=r['P_payload_bytes']*8
  rates.append(dict(mode=r['mode'],sequence=r['sequence'],QP=r['external_qp'],estimated_y_bits=r['estimated_P_y_bits'],estimated_z_bits=r['estimated_P_z_bits'],actual_P_combined_payload_bits=actual,actual_minus_estimated_bits=actual-est,relative_gap_percent=100*(actual/est-1),scope='P1-P63 only; excludes native I and all headers'))
 table(P/'estimated_vs_actual_P_rate.csv',rates)
 for f,r in load(P/'weight_manifest.json').items():artifacts[f]=dict(path=f,**{k:r[k] for k in ('sha256','bytes')})
 dump(P/'artifact_index.json',dict(status='PASS',server_only=True,records=list(artifacts.values())))
 dump(P/'delivery_audit.json',dict(status='PASS',signature=signature(),tasks=40,full64_streams=40,P_maps_checked=maps_checked,training_updates=0,immutable_weights_and_sources=True,records=receipts,all_I_payloads_same='cross_mode_audit.json',FRAME_CHANNEL_exact='cross_mode_audit.json',all_metrics_from_saved_tables=True))
 return rows
def interp(rs,x,metric,kind):
 a=sorted(rs,key=lambda r:r['actual_bpp']);lo,hi=a[0],a[-1]
 assert lo['actual_bpp']-1e-14<=x<=hi['actual_bpp']+1e-14
 if x==lo['actual_bpp']:return lo[metric]
 if x==hi['actual_bpp']:return hi[metric]
 f=math.log if kind=='log-bpp' else lambda z:z
 w=(f(x)-f(lo['actual_bpp']))/(f(hi['actual_bpp'])-f(lo['actual_bpp']))
 return lo[metric]+w*(hi[metric]-lo[metric])
def matched(rows):
 matched=[];dominance=[];ranks=[];attribution=[];effects=[]
 for source in SOURCES:
  by={m:[r for r in rows if r['sequence']==source and r['mode']==m] for m in MODES}
  for a,b in itertools.combinations(MODES,2):
   # Ordered later-minus-earlier contrasts, e.g. SPATIAL-FRAME.
   method,ref=b,a;rs,bs=by[method],by[ref]
   for u,v in itertools.product(rs,bs):
    flags={m:u['actual_bpp']<=v['actual_bpp'] and (u[m]>=v[m] if m=='PSNR' else u[m]<=v[m]) and (u['actual_bpp']<v['actual_bpp'] or u[m]!=v[m]) for m in METRICS}
    reverse={m:v['actual_bpp']<=u['actual_bpp'] and (v[m]>=u[m] if m=='PSNR' else v[m]<=u[m]) and (v['actual_bpp']<u['actual_bpp'] or v[m]!=u[m]) for m in METRICS}
    joint=u['actual_bpp']<=v['actual_bpp'] and all(u[m]>=v[m] if m=='PSNR' else u[m]<=v[m] for m in METRICS) and (u['actual_bpp']<v['actual_bpp'] or any(u[m]!=v[m] for m in METRICS))
    reverse_joint=v['actual_bpp']<=u['actual_bpp'] and all(v[m]>=u[m] if m=='PSNR' else v[m]<=u[m] for m in METRICS) and (v['actual_bpp']<u['actual_bpp'] or any(u[m]!=v[m] for m in METRICS))
    dominance.append(dict(sequence=source,method=method,reference=ref,method_QP=u['external_qp'],reference_QP=v['external_qp'],method_bpp=u['actual_bpp'],reference_bpp=v['actual_bpp'],**{m+'_delta':u[m]-v[m] for m in METRICS},**{m+'_method_dominates':x for m,x in flags.items()},**{m+'_reference_dominates':x for m,x in reverse.items()},joint_method_dominates=joint,joint_reference_dominates=reverse_joint))
   lo=max(min(r['actual_bpp'] for r in rs),min(r['actual_bpp'] for r in bs));hi=min(max(r['actual_bpp'] for r in rs),max(r['actual_bpp'] for r in bs))
   if lo>hi:matched.append(dict(sequence=source,method=method,reference=ref,status='no_observed_overlap'));continue
   for kind,position in itertools.product(('linear-bpp','log-bpp'),('low','mid','high')):
    x={'low':lo,'mid':(lo+hi)/2,'high':hi}[position]
    r=dict(sequence=source,method=method,reference=ref,status='within_observed_overlap_interpolation',interpolation=kind,position=position,actual_bpp=x,overlap_low=lo,overlap_high=hi)
    for m in METRICS:r.update({m+'_method':interp(rs,x,m,kind),m+'_reference':interp(bs,x,m,kind),m+'_delta':interp(rs,x,m,kind)-interp(bs,x,m,kind)})
    matched.append(r)
  for kind in ('linear-bpp','log-bpp'):
   modes=('FRAME','CHANNEL','SPATIAL');lo=max(min(r['actual_bpp'] for r in by[m]) for m in modes);hi=min(max(r['actual_bpp'] for r in by[m]) for m in modes)
   if lo<=hi:
    x=(lo+hi)/2
    for metric in METRICS:
     values={m:interp(by[m],x,metric,kind) for m in modes};best=(max if metric=='PSNR' else min)(values.values());winners=[m for m,v in values.items() if abs(v-best)<=1e-12]
     ranks.append(dict(sequence=source,interpolation=kind,metric=metric,actual_bpp=x,winners=winners,**values))
   modes=('A','IDENTITY','SPATIAL');lo=max(min(r['actual_bpp'] for r in by[m]) for m in modes);hi=min(max(r['actual_bpp'] for r in by[m]) for m in modes)
   if lo<=hi:
    x=(lo+hi)/2
    for metric in METRICS:
     values={m:interp(by[m],x,metric,kind) for m in modes};sign=1 if metric=='PSNR' else -1
     total=sign*(values['SPATIAL']-values['A']);gen=sign*(values['IDENTITY']-values['A']);extra=total-gen
     attribution.append(dict(sequence=source,interpolation=kind,metric=metric,actual_bpp=x,A=values['A'],IDENTITY=values['IDENTITY'],SPATIAL=values['SPATIAL'],SPATIAL_total_improvement=total,IDENTITY_improvement=gen,SPATIAL_increment_over_IDENTITY=extra,identity_fraction_of_positive_total=gen/total if total>1e-8 else '',majority_explained=gen>=.5*total if total>1e-8 else 'not_applicable_nonpositive_total'))
   modes=('IDENTITY','FRAME','SPATIAL');lo=max(min(r['actual_bpp'] for r in by[m]) for m in modes);hi=min(max(r['actual_bpp'] for r in by[m]) for m in modes)
   if lo<=hi:
    x=(lo+hi)/2
    for metric in METRICS:
     values={m:interp(by[m],x,metric,kind) for m in modes};sign=1 if metric=='PSNR' else -1
     frame=sign*(values['FRAME']-values['IDENTITY']);spatial=sign*(values['SPATIAL']-values['FRAME'])
     effects.append(dict(sequence=source,interpolation=kind,metric=metric,actual_bpp=x,FRAME_over_IDENTITY_improvement=frame,SPATIAL_over_FRAME_improvement=spatial,absolute_space_to_frame_effect_ratio=abs(spatial/frame) if abs(frame)>1e-8 else '',space_magnitude_smaller_than_frame_shift=abs(spatial)<abs(frame)))
 table(P/'matched-rate_comparisons.csv',matched);table(P/'observed_dominance.csv',dominance);table(P/'granularity_rankings.csv',ranks);table(P/'identity_attribution.csv',attribution)
 table(P/'allocation_effects.csv',effects)
 return matched,dominance,ranks,attribution,effects
def mdtable(rs,keys):
 lines=['| '+' | '.join(keys)+' |','| '+' | '.join('---' for k in keys)+' |']
 for r in rs:lines.append('| '+' | '.join(f'{r[k]:.7g}' if isinstance(r.get(k),float) else str(r.get(k,'')) for k in keys)+' |')
 return '\n'.join(lines)+'\n'
def report(rows,matched,dom,ranks,attrs,effects):
 lines=['# v4 quantization granularity ablation','','## Result and code facts','','This completed inference-only ablation uses 40 new native 64-frame streams, QP2/7, alpha1, step300, zero training updates. All encoder/independent-decoder hashes and original A/IDENTITY native-state/stream checks pass.','','The existing H11 emits one multiplier channel shared across128 latent channels. FRAME and CHANNEL are therefore exactly the same operation, not independent spatial/channel interventions. Their eight pairs of measured streams, states and metrics are identical. This is a code and measured equality; it does not establish that a separately trained channel-dependent H would be ineffective.','','A is the requested v1 A step300 generator with the original codec; IDENTITY keeps H11 G but sets m=1. They share compressed streams/states, while their RGB outputs can differ. No untouched official-generator baseline is added.','','## Measured points and dominance','','All table values below are measured native RANS points. Lower LPIPS/DISTS/FloLPIPS and higher PSNR are better. The complete table also records actual bytes, P payload, estimated y/z bits, time, memory and GPU. Actual y/z RANS payload is combined.','',mdtable(sorted(rows,key=lambda r:(SOURCES.index(r['sequence']),MODES.index(r['mode']),r['external_qp'])),['mode','sequence','external_qp','actual_bpp','PSNR','LPIPS','DISTS','FloLPIPS']),'','[observed_dominance.csv](observed_dominance.csv) records every cross-QP pair in both directions, including unfavorable differences. Observed dominance is rate no higher and quality no worse, with at least one strict improvement; interpolated rankings below are distinct from this measured-point claim.','']
 counts=[]
 for a,b in [('SPATIAL','FRAME'),('SPATIAL','CHANNEL'),('SPATIAL','IDENTITY'),('IDENTITY','A')]:
  rs=[r for r in dom if r['method']==a and r['reference']==b]
  counts.append(dict(comparison=a+'-'+b,measured_pairs=len(rs),method_joint_dominance=sum(r['joint_method_dominates'] for r in rs),reference_joint_dominance=sum(r['joint_reference_dominates'] for r in rs)))
 lines.append(mdtable(counts,['comparison','measured_pairs','method_joint_dominance','reference_joint_dominance']))
 lines+=['','## Who is best at similar actual rates?','','The following rankings are interpolated quality at the midpoint of the common FRAME/CHANNEL/SPATIAL observed rate overlap for each sequence. Both interpolation definitions use the same actual-bpp query coordinate. They are descriptive, not new measured encodings or BD-rate.','',mdtable(ranks,['sequence','interpolation','metric','actual_bpp','winners','FRAME','CHANNEL','SPATIAL'])]
 lines+=['','## Is SPATIAL stably better than FRAME and CHANNEL?','']
 stable={};stats=[]
 for metric in METRICS:
  rs=[r for r in matched if r['method']=='SPATIAL' and r['reference'] in ('FRAME','CHANNEL') and r['status']=='within_observed_overlap_interpolation']
  sign=1 if metric=='PSNR' else -1;good=sum(sign*r[metric+'_delta']>1e-12 for r in rs);bad=sum(sign*r[metric+'_delta']< -1e-12 for r in rs)
  stable[metric]=len(rs)==48 and good==len(rs)
  stats.append(dict(metric=metric,checked_sequence_method_interpolation_coordinates=len(rs),favorable=good,unfavorable=bad,ties=len(rs)-good-bad,strictly_favorable_everywhere=stable[metric]))
 lines.append(mdtable(stats,['metric','checked_sequence_method_interpolation_coordinates','favorable','unfavorable','ties','strictly_favorable_everywhere']))
 lines.append('SPATIAL is strictly favorable across all tested overlap coordinates, sequences, both comparisons and interpolation choices for every metric.' if all(stable.values()) else 'SPATIAL is not consistently better across all tested metrics, sequences and interpolation coordinates. These fixed-checkpoint results do not support a stable, general spatial-allocation advantage.')
 wins=[]
 for metric in METRICS:
  rs=[r for r in ranks if r['metric']==metric]
  wins.append(dict(metric=metric,sequence_interpolation_cases=len(rs),SPATIAL_best=sum('SPATIAL' in r['winners'] for r in rs),FRAME_CHANNEL_best=sum('FRAME' in r['winners'] for r in rs)))
 lines+=['',mdtable(wins,['metric','sequence_interpolation_cases','SPATIAL_best','FRAME_CHANNEL_best']),'','### Global-scale versus additional spatial change','', 'Both comparisons below share one actual rate coordinate inside the IDENTITY/FRAME/SPATIAL overlap, with favorable changes positive. They quantify descriptive magnitudes, not a learned-mechanism decomposition.','',mdtable(effects,['sequence','interpolation','metric','FRAME_over_IDENTITY_improvement','SPATIAL_over_FRAME_improvement','absolute_space_to_frame_effect_ratio'])]
 smaller=sum(r['space_magnitude_smaller_than_frame_shift'] for r in effects)
 lines.append(f'The absolute additional SPATIAL-over-FRAME change is smaller than the FRAME-over-IDENTITY change in {smaller}/{len(effects)} sequence/metric/interpolation cases. Where FRAME already matches or beats SPATIAL, there is no measured-interpolation need for spatial positioning to explain that quality/rate performance. This supports a scale-based explanation only for those observed cases; it is not an equivalence test or proof of absence of spatial effects.')
 lines+=['','FRAME and CHANNEL equality is required by the architecture. Whether they approximate SPATIAL must be read from the numerical differences; no arbitrary equivalence threshold is introduced. A failure to see stable SPATIAL superiority is not proof that spatial allocation has no causal effect.','', '## Does IDENTITY explain the main gain?','','Positive improvement is defined in the favorable direction. On the common A/IDENTITY/SPATIAL observed overlap, the identity fraction is (IDENTITY−A improvement)/(SPATIAL−A improvement). It is reported only for a positive total above1e-8, can exceed1 or be negative, and is not a causal decomposition.','',mdtable(attrs,['sequence','interpolation','metric','SPATIAL_total_improvement','IDENTITY_improvement','SPATIAL_increment_over_IDENTITY','identity_fraction_of_positive_total','majority_explained'])]
 total=[a for a in attrs if isinstance(a['majority_explained'],bool)];explained=sum(a['majority_explained'] for a in total)
 lines.append(f'IDENTITY accounts for at least half of the interpolated positive total improvement in {explained}/{len(total)} applicable sequence/metric/interpolation cases. Nonpositive-total cases are not evidence of explained positive gain. Inspect metric-specific rows; there is no single validated all-metric attribution percentage.')
 lines+=['','## Audits and scope','','[delivery_audit.json](delivery_audit.json) verifies40 full64-frame streams,2520 full P-frame map files, every durable artifact SHA, native I equality, A/IDENTITY native byte/state parity, decoder state and multiplier-map equality, the post-clamp/post-dtype q/rho equations, and metrics recomputed from saved per-frame tables. [cross_mode_audit.json](cross_mode_audit.json) verifies FRAME/CHANNEL equality independently at all8 video/QP pairs. [P rate accounting](estimated_vs_actual_P_rate.csv) excludes I and headers; analytic y/z estimates remain separate from the real combined RANS P payload. FID:not_computed. No training, alpha search, extra QP or legacy result modification occurred.','','The shared mode is fixed experiment configuration; the native file has no new header or encoder-only map. Raw and variant multipliers are B×1×H×W; actual qmap/rho/effective multiplier contain128 channels. Effective means the realized native-dtype ratio q_new/q_base, not an unclamped proposal.','','## Unverified causal claims and limitations','','Two-QP interpolation, one trained checkpoint/seed, inherited historical training exposure and four videos cannot establish significance, convergence, or generalization. FRAME/CHANNEL equality prevents independently attributing benefits to channelwise H. SPATIAL-vs-FRAME changes the whole causal reference trajectory as well as local scales; it does not isolate a position-only bit-budget redistribution at fixed total bits. IDENTITY-vs-A changes G as specified; interpolated gain fractions are descriptive and can be unstable near zero. Stable SPATIAL superiority would support the spatial-allocation hypothesis within this protocol, but would still not identify a unique mechanism.','','These are inference-time interventions on a spatially trained H. FRAME/CHANNEL are not separately trained optimums. Actual native q_base can vary over channels even when the multiplier is one channel. Model timing includes map saving/diagnostics, and peak memory includes metrics/native-parity work, so it is not a controlled speed benchmark.','','## Curves and fixed previews','']
 for s in SOURCES:lines += [f'![{s} measured RD](figures/RD_{s}.png)',f'![{s} fixed frame15](figures/preview_{s}.png)','']
 lines+=['![Multiplier maps](figures/fixed_maps_UVal0.png)','','[Protocol](PROTOCOL.md) · [Interpolation table](matched-rate_comparisons.csv) · [Map/stream/weight paths and SHA256](artifact_index.json)']
 frame_favorable=sum(r['FRAME_over_IDENTITY_improvement']>0 for r in effects)
 overview=f'**Observed conclusion:** FRAME and CHANNEL are exactly equivalent on this singleton-channel H. SPATIAL has no stable across-sequence/metric/interpolation advantage over them in this run. Additional spatial changes are smaller in magnitude than FRAME-versus-IDENTITY changes in {smaller}/{len(effects)} cases; this describes scale sensitivity, not guaranteed gains: FRAME improves over IDENTITY in only {frame_favorable}/{len(effects)} cases. IDENTITY explains at least half of positive SPATIAL-versus-A improvement in {explained}/{len(total)} applicable cases, and cannot be claimed to explain the main gain universally. Each case is one video/metric/interpolation combination, not an independent sample. These results do not establish spatial generation-aware bit allocation as the source of a robust gain.'
 lines[4:4]=[overview,'']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
 dump(P/'conclusions.json',dict(status='DESCRIPTIVE',FRAME_CHANNEL_exact=True,stable_SPATIAL_by_metric=stable,stable_SPATIAL_all_metrics=all(stable.values()),midpoint_best_counts=wins,additional_spatial_magnitude_smaller_cases=smaller,allocation_comparison_cases=len(effects),FRAME_over_IDENTITY_favorable_cases=frame_favorable,identity_majority_cases=explained,identity_positive_total_cases=len(total),causal_claim='not established by this fixed-checkpoint intervention alone'))
def main():
 rows=audit();a,b,c,d,e=matched(rows);report(rows,a,b,c,d,e)
 print('FINAL_AUDIT_REPORT_PASS',len(rows),flush=True)
if __name__=='__main__':main()
