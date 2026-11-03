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
 streams=[]
 for keys,rs in group(final,('split','domain','sequence','QP')):
  assert len(rs)==4
  streams.append(dict(zip(('split','domain','sequence','QP'),keys),model_states=[r['model_state'] for r in rs],stream_sha256=[r['bitstream_sha256'] for r in rs],real_bytes=[int(r['real_bytes']) for r in rs],all_stream_bytes_identical=len({r['bitstream_sha256'] for r in rs})==1,source='independently persisted fresh RANS streams; observed rates never replaced'))
 dump(ROOT/'evaluation/stream_comparison.json',dict(records=streams,different_stream_pairs=[r for r in streams if not r['all_stream_bytes_identical']],encoder_source=dict(file='src/models/video_model_gvcrt.py',sha256=sha(REPO/'src/models/video_model_gvcrt.py')),note='TCG outputs and reference reconstruction are produced by the actual fixed codec. Differences are retained; detailed per-frame bit and state receipts provide engineering evidence. No equality of rates is imposed.'))
 traces=[]
 for arm in ('A_old_pool','B_expanded_ulong'):
  for f in sorted((ROOT/arm/'receipts').glob('*.json.gz')):
   r=json.loads(gzip.decompress(f.read_bytes()));traces.append(dict(arm=arm,additional_step=r['additional_step'],absolute_optimizer_step=r['absolute_optimizer_step'],gradient_norm=r['rank_audits'][0]['reduced_preclip_gradient_norm'],seconds=max(x['seconds'] for x in r['rank_audits']),**r['mean_losses']))
 write(ROOT/'training_loss.csv',traces)
 write(ROOT/'warping_error.csv',[{k:r[k] for k in ('evaluation_kind','split','domain','sequence','QP','model_state','WarpingError','WarpingError_ReconFlow_PWC','WarpingError_GTFlow_PWC') if k in r} for r in allrows])
 plot(final,dev,traces,aggregates)
 lines=['# TCG Stage 1 paired data expansion','',f"Execution: A/B {load(ROOT/'protocol.json')['additional_global_updates']} additional updates per arm.",'','Protocol: EXECUTION_PROTOCOL.md and protocol.json. Loss/precision, source exclusions, parent optimizer/RNG provenance and independent encoding are recorded in the referenced manifests/audits.','Metrics are from raw decoded tensors. JPEG samples are explicitly lossy local viewing previews.','GT-flow is PWC-based with the recorded existing sampler; no exact S2VC implementation reproduction claim.','No FVD/FID or BD-rate was computed. One paired initialization; four ranks form one model, not four seeds.','Coverage passes are identity visits; overlapping windows and exact repeats are recorded in plan statistics.','Fingerprint screening is finite; unconfirmed same-source relationships and exclusion reasons remain in inventory/.','', '| Split | Domain | Sequence | QP | Model | '+' | '.join(METRICS)+' |','|---|---|---|---:|---|'+'|'.join(['---:']*len(METRICS))+'|']
 for r in final:lines.append('| '+' | '.join(str(r[k]) for k in ('split','domain','sequence','QP','model_state'))+' | '+' | '.join(str(r[m]) for m in METRICS)+' |')
 lines+=['','Development trajectory full table: evaluation/development/metrics.csv.','Parent156 GT-flow full table: evaluation/parent_gtflow/metrics.csv.','Paired differences: paired_deltas.csv. Counts/weights/denominators: numerical_point_counts.csv and summary_by_domain_QP_split.csv.','Full frames/transitions/stream receipts: evaluation/. Complete compressed per-update rank receipts: A_old_pool/receipts and B_expanded_ulong/receipts.','Final engineering acceptance: final_integrity.json. Publication acceptance: publication_status.json.']
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
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
 for m in ('total_loss','LPIPS','DISTS','motion_loss','reconstruction_L1','gradient_norm'):
  fig,ax=plt.subplots()
  for arm,rs in group(traces,('arm',)):ax.plot([r['additional_step'] for r in rs],[r[m] for r in rs],label=arm[0],linewidth=.6)
  ax.set(xlabel='Committed additional global update',ylabel=m);ax.legend();fig.tight_layout();fig.savefig(out/f'training_{m}.png',dpi=120);plt.close(fig)
 for q in (0,5,9):
  for m in quality:
   fig,ax=plt.subplots()
   for arm in ('A_old_pool','B_expanded_ulong'):
    selected=[r for r in dev if int(r['QP'])==q and (r['model_state']=='parent5000' or r['model_state'].startswith(arm))]
    curve=[]
    for step,rs in group(selected,('additional_step',)):curve.append((int(step[0]),sum(float(r[m]) for r in rs)/len(rs)))
    curve.sort();ax.plot([r[0] for r in curve],[r[1] for r in curve],marker='o',label=arm)
   ax.set(xlabel='Additional update',ylabel=m,title=f'Development32 equal video mean QP{q}');ax.legend();fig.tight_layout();fig.savefig(out/f'development_QP{q}_{m}.png',dpi=120);plt.close(fig)
if __name__=='__main__':main()
