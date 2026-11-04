"""Tables and plots of measured values only; no model selection or interpretation."""
from ab_common import *
from gtflow import write,METRICS
import csv,collections,gzip
def read(p):return list(csv.DictReader(Path(p).open()))
def main():
 final=read(ROOT/'evaluation/final/metrics.csv');dev=read(ROOT/'evaluation/development/metrics.csv');parent=[dict(r,model_state='parent'+r['step'],split='historical_reference') for r in read(ROOT/'evaluation/parent_gtflow/metrics.csv')]
 allrows=[dict(r,evaluation_kind=k) for k,rs in [('final',final),('development',dev),('parent_gtflow',parent)] for r in rs]
 write(ROOT/'metrics.csv',allrows)
 lookup={(r['split'],r['domain'],r['sequence'],r['QP'],r['model_state']):r for r in final};pairs=[]
 for target,reference in [('B_final','A_final'),('A_final','parent5000'),('B_final','parent5000'),('A_final','parent0'),('B_final','parent0'),('parent5000','parent0')]:
  for r in final:
   if r['model_state']!=target:continue
   a=lookup[(r['split'],r['domain'],r['sequence'],r['QP'],reference)]
   for metric in METRICS:
    av,bv=float(a[metric]),float(r[metric]);delta=bv-av;direction=1 if metric in ('PSNR','MS_SSIM') else -1
    pairs.append(dict(split=r['split'],domain=r['domain'],sequence=r['sequence'],QP=int(r['QP']),target=target,reference=reference,metric=metric,reference_value=av,target_value=bv,absolute_delta=delta,relative_delta=delta/abs(av) if av else None,relative_denominator='abs(reference_value); undefined when zero',numerical_improvement=direction*delta>0,numerical_tie=delta==0,preferred_direction='higher' if direction==1 else 'lower'))
 write(ROOT/'paired_deltas.csv',pairs)
 aggregates=[]
 for keys,rs in group(allrows,('evaluation_kind','split','domain','QP','model_state')):
  aggregates.append(dict(zip(('evaluation_kind','split','domain','QP','model_state'),keys),points=len(rs),unique_video_count=len({r['sequence'] for r in rs}),weighting='equal sequence weight within fixed QP/split/domain',**{m:sum(float(r[m]) for r in rs)/len(rs) for m in METRICS}))
 write(ROOT/'summary_by_domain_QP_split.csv',aggregates)
 counts=[]
 for keys,rs in group(pairs,('split','domain','QP','target','reference','metric')):
  counts.append(dict(zip(('split','domain','QP','target','reference','metric'),keys),numerical_improvement_points=sum(r['numerical_improvement'] for r in rs),tie_points=sum(r['numerical_tie'] for r in rs),denominator_paired_points=len(rs),unique_videos=len({r['sequence'] for r in rs}),unit='sequence at a fixed QP; QPs are not independent video identities'))
 write(ROOT/'numerical_point_counts.csv',counts)
 paired_summary=[]
 for scope,fields in [('sequence',('split','domain','sequence','target','reference','metric')),('domain_QP',('split','domain','QP','target','reference','metric')),('split_QP',('split','QP','target','reference','metric')),('domain_all_three_QPs',('split','domain','target','reference','metric'))]:
  for keys,rs in group(pairs,fields):
   relative=[r['relative_delta'] for r in rs if r['relative_delta'] is not None]
   paired_summary.append(dict(scope=scope,**dict(zip(fields,keys)),paired_point_denominator=len(rs),unique_videos=len({(r['domain'],r['sequence']) for r in rs}),QP_values=sorted({r['QP'] for r in rs}),reference_mean=sum(r['reference_value'] for r in rs)/len(rs),target_mean=sum(r['target_value'] for r in rs)/len(rs),mean_absolute_delta=sum(r['absolute_delta'] for r in rs)/len(rs),mean_point_relative_delta=sum(relative)/len(relative) if relative else None,relative_delta_denominator=len(relative),undefined_relative_points=len(rs)-len(relative),numerical_improvement_points=sum(r['numerical_improvement'] for r in rs),tie_points=sum(r['numerical_tie'] for r in rs),weighting='equal sequence/QP points in the displayed scope; multiple QPs are not independent videos'))
 write(ROOT/'paired_summary.csv',paired_summary)
 streams=[]
 for keys,rs in group(final,('split','domain','sequence','QP')):
  assert len(rs)==4
  streams.append(dict(zip(('split','domain','sequence','QP'),keys),model_states=[r['model_state'] for r in rs],stream_sha256=[r['bitstream_sha256'] for r in rs],real_bytes=[int(r['real_bytes']) for r in rs],all_stream_bytes_identical=len({r['bitstream_sha256'] for r in rs})==1,source='independently persisted fresh RANS streams; observed rates never replaced'))
 diagnoses=[]
 for observation in streams:
  if observation['all_stream_bytes_identical']:continue
  rs=[r for r in final if all(r[k]==str(observation[k]) for k in ('split','domain','sequence','QP'))];reference=next(r for r in rs if r['model_state']=='A_final');a=read(ROOT/'evaluation/final/encoder_reference'/(Path(reference['bitstream_file']).stem+'.csv'))
  for r in rs:
   if r['model_state']=='A_final' or r['bitstream_sha256']==reference['bitstream_sha256']:continue
   b=read(ROOT/'evaluation/final/encoder_reference'/(Path(r['bitstream_file']).stem+'.csv'));assert len(a)==len(b)==64 and all(x['actual_qp']==y['actual_qp'] for x,y in zip(a,b))
   differences=[i for i,(x,y) in enumerate(zip(a,b)) if x['payload_sha256']!=y['payload_sha256']]
   assert differences and differences[0]>0,'UNEXPECTED_I_FRAME_STREAM_DIFFERENCE'
   first=differences[0];changed=[k for k in ('reference_frame_sha256','reference_feature_sha256') if a[first-1][k]!=b[first-1][k]]
   diagnoses.append(dict(split=r['split'],domain=r['domain'],sequence=r['sequence'],QP=r['QP'],target=r['model_state'],reference='A_final',first_payload_difference_frame=first,preceding_reference_fields_different=changed,engineering_record='Frozen encoder parameters and matching QPs checked; differing reconstructed/reference state precedes the changed P payload' if changed else 'Unresolved payload difference under matching logged reference state; retain failure for engineering investigation',resolved=bool(changed)))
 dump(ROOT/'evaluation/bitstream_engineering_audit.json',dict(status='PASS' if all(r['resolved'] for r in diagnoses) else 'FAILED',comparisons=diagnoses,no_rate_overwrite=True))
 assert all(r['resolved'] for r in diagnoses),'UNRESOLVED_BITSTREAM_DIFFERENCE; see logged encoder reference state'
 dump(ROOT/'evaluation/stream_comparison.json',dict(records=streams,different_stream_pairs=[r for r in streams if not r['all_stream_bytes_identical']],encoder_source=dict(file='src/models/video_model_gvcrt.py',sha256=sha(REPO/'src/models/video_model_gvcrt.py')),note='TCG outputs and reference reconstruction are produced by the actual fixed codec. Differences are retained; detailed per-frame bit and state receipts provide engineering evidence. No equality of rates is imposed.'))
 traces=[]
 for arm in ('A_clean_old_pool','B_expanded_ulong'):
  for f in sorted((ROOT/arm/'receipts').glob('*.json.gz')):
   r=json.loads(gzip.decompress(f.read_bytes()));traces.append(dict(arm=arm,additional_step=r['additional_step'],absolute_optimizer_step=r['absolute_optimizer_step'],gradient_norm=r['rank_audits'][0]['reduced_preclip_gradient_norm'],seconds=max(x['seconds'] for x in r['rank_audits']),**r['mean_losses']))
 write(ROOT/'training_loss.csv',traces)
 write(ROOT/'warping_error.csv',[{k:r[k] for k in ('evaluation_kind','split','domain','sequence','QP','model_state','WarpingError','WarpingError_ReconFlow_PWC','WarpingError_GTFlow_PWC') if k in r} for r in allrows])
 plot(final,dev,traces,aggregates)
 protocol=load(ROOT/'protocol.json');summary=dict(status='PASS',protocol_revision=REVISION,N_A=protocol['N_A'],N_B=protocol['N_B'],S=protocol['additional_global_updates'],absolute_final_optimizer_step=protocol['absolute_final_optimizer_step'],removed_old_training_identities=protocol['removed_old_training_identities'],quarantined_source_groups=protocol['quarantined_source_groups'],arms={a:load(ROOT/f'{a}/training_summary.json') for a in ('A_clean_old_pool','B_expanded_ulong')},evaluation_counts=dict(parent_GTFlow_reused=156,development=len(dev),final=len(final)),primary_comparison='B_final versus A_final',primary_split='new_holdout_test',unique_new_test_videos=32,QP_points_per_model_per_new_test_video=3,science_interpretation='NOT_PERFORMED')
 lines=['# TCG Stage 1 group-isolated paired data expansion v2','',f"Execution: A/B {protocol['additional_global_updates']} committed additional updates per arm; absolute optimizer step{protocol['absolute_final_optimizer_step']}.",'', '```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','', 'Protocol: EXECUTION_PROTOCOL.md, revision_diff.json and protocol.json. Loss/precision, source exclusions, parent optimizer/RNG provenance and independent encoding are recorded in the referenced manifests/audits.','Source grouping includes reserves before exclusion and conservatively quarantines complete components. Parent prior exposure is recorded in historical_exposure.json and is not erased by this filtering.','Claim: 在所记录来源和筛查规则下，训练/开发/测试来源组无交集。 Sparse screening does not exhaust all shifts/crops/reencodes or establish foundation-pretraining exclusion.','Historical13 exposure flags are in evaluation/historical_reference_exposure.json; these points remain historical references, separate from the primary new_holdout_test summary.','Metrics are from raw decoded tensors. JPEG samples are explicitly lossy local viewing previews.','GT-flow is PWC-based with the recorded existing sampler; no exact S2VC implementation reproduction claim. Width11/dx1 samples1.1pixels: existing-implementation compatibility, not strict one-pixel translation.','No FVD/FID or BD-rate was computed. One paired initialization; four ranks form one model, not four seeds.','Coverage passes are identity visits; overlapping windows, exact repeats, QP counts and weights are recorded in compressed plan coverage statistics.','Parent156 rescore points are hash-verified E1 reuse (9984 frames/9828 transitions); they are not new A/B evaluations. New final/development points have separate fresh streams and independent decoders.','Engineering exceptions and stage/launch histories remain in audits and bounded logs/summary. Source/cache lock contention was corrected before training; no formal optimizer update was involved.','', '| Split | Domain | Sequence | QP | Model | '+' | '.join(METRICS)+' |','|---|---|---|---:|---|'+'|'.join(['---:']*len(METRICS))+'|']
 for r in final:lines.append('| '+' | '.join(str(r[k]) for k in ('split','domain','sequence','QP','model_state'))+' | '+' | '.join(str(r[m]) for m in METRICS)+' |')
 lines+=['','Development trajectory full table: evaluation/development/metrics.csv.','Parent156 GT-flow full table: evaluation/parent_gtflow/metrics.csv.','Paired differences: paired_deltas.csv. Counts/weights/denominators: numerical_point_counts.csv and summary_by_domain_QP_split.csv.','Full frames/transitions/stream receipts: evaluation/. Complete compressed per-update rank receipts: A_clean_old_pool/receipts and B_expanded_ulong/receipts.','Final engineering acceptance: final_integrity.json. Publication acceptance: publication_status.json.']
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 dump(ROOT/'summary.json',summary)
def group(rows,fields):
 d=collections.defaultdict(list)
 for r in rows:d[tuple(r.get(f,'') for f in fields)].append(r)
 return sorted(d.items())
def plot(final,dev,traces,aggregates):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=ROOT/'figures';out.mkdir(exist_ok=True)
 quality=[m for m in METRICS if m not in ('bpp','kbps','WarpingError')]
 for keys,rs in group(final,('split','domain','sequence')):
  for m in quality:
   fig,ax=plt.subplots()
   for state,points in group(rs,('model_state',)):
    points.sort(key=lambda r:float(r['bpp']));ax.plot([float(r['bpp']) for r in points],[float(r[m]) for r in points],marker='o',label=state[0])
   ax.set(xlabel='Measured bpp',ylabel=m,title=' / '.join(keys));ax.legend();fig.tight_layout();fig.savefig(out/f'{keys[-1]}_{m}.png',dpi=120);plt.close(fig)
 for keys,rs in group([r for r in aggregates if r['evaluation_kind']=='final'],('split','domain')):
  for m in quality:
   fig,ax=plt.subplots()
   for state,points in group(rs,('model_state',)):
    points.sort(key=lambda r:r['bpp']);ax.plot([r['bpp'] for r in points],[r[m] for r in points],marker='o',label=state[0])
   ax.set(xlabel='Mean measured bpp (equal video weight)',ylabel=m,title=' / '.join(keys));ax.legend();fig.tight_layout();fig.savefig(out/f'{keys[0]}_{keys[1]}_{m}.png',dpi=120);plt.close(fig)
 for m in ('total_loss','frame_loss','LPIPS','DISTS','motion_loss','reconstruction_L1','gradient_norm'):
  fig,ax=plt.subplots()
  for arm,rs in group(traces,('arm',)):ax.plot([r['additional_step'] for r in rs],[r[m] for r in rs],label=arm[0],linewidth=.6)
  ax.set(xlabel='Committed additional global update',ylabel=m);ax.legend();fig.tight_layout();fig.savefig(out/f'training_{m}.png',dpi=120);plt.close(fig)
 for q in (0,5,9):
  for m in quality:
   fig,ax=plt.subplots()
   for arm in ('A_clean_old_pool','B_expanded_ulong'):
    selected=[r for r in dev if int(r['QP'])==q and (r['model_state']=='parent5000' or r['model_state'].startswith(arm))]
    curve=[]
    for step,rs in group(selected,('additional_step',)):curve.append((int(step[0]),sum(float(r[m]) for r in rs)/len(rs)))
    curve.sort();ax.plot([r[0] for r in curve],[r[1] for r in curve],marker='o',label=arm)
   ax.set(xlabel='Additional update',ylabel=m,title=f'Development32 equal video mean QP{q}');ax.legend();fig.tight_layout();fig.savefig(out/f'development_QP{q}_{m}.png',dpi=120);plt.close(fig)
if __name__=='__main__':main()
