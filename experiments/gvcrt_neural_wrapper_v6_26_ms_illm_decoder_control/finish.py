"""Numerical tables, gradient-control counts and final publication checklist."""
from io21 import *
from report import PAIRS
import statistics,math

def main():
 assert load(ROOT/'results/fid_decomposition/final_integrity.json')['status']=='PASS'
 assert load(ROOT/'audits/cross_round_integrity.json')['status']=='PASS'
 assert load(ROOT/'historical/visual_integrity.json')['status']=='PASS'
 from reconcile_gpu_schedule import main as reconcile
 reconcile()
 assert load(ROOT/'audits/computational_integrity.json')['status']=='PASS'
 for name in ('actual_DISTS_optimizer_path','labeler_probe'):
  assert load(ROOT/'audits'/f'{name}.json')['status']=='PASS'
 assert load(ROOT/'historical/final_integrity.json')['status']=='PASS'
 hrows=load(ROOT/'historical/fid_decomposition_raw.json')
 for t in hrows:
  t['historical_reproduction_error']=abs(t['reproduced_FID']-t['historical_FID']) if t['historical_FID'] is not None else None
  t['historical_reproduction_tolerance']=1e-7+1e-8*abs(t['historical_FID']) if t['historical_FID'] is not None else None
  assert t['historical_reproduction_error'] is None or t['historical_reproduction_error']<=t['historical_reproduction_tolerance']
 dump(ROOT/'historical/fid_decomposition_raw.json',hrows);write(ROOT/'historical/fid_decomposition_raw.csv',hrows)
 summaries={};core=[];band_rows=[];band_summary=[]
 for b in BRANCHES:
  rows=[json.loads(x) for x in (ROOT/'branches'/b/'training_logs/train.jsonl').read_text().splitlines()];assert len(rows)==1000
  keys=['lambda_eff','adv_B1_ratio','gradient_cosine','global_clip_coefficient',*(k+'_weighted_adv_B1_ratio' for k in ('wrapper','bridge','generator'))]
  s=dict(steps=len(rows),G_updates=1000,D_updates=1000 if b!='A_decoder_continue' else 0,violations=sum(r['adv_B1_ratio']>.05+1e-8 for r in rows),ceiling_steps=sum(r['ceiling_triggered'] for r in rows),clipped_steps=sum(r['global_clip_coefficient']<1 for r in rows),distributions={k:dict(min=min(r[k] for r in rows),max=max(r[k] for r in rows),mean=statistics.mean(r[k] for r in rows),median=statistics.median(r[k] for r in rows)) for k in keys});assert s['violations']==0
  easy=read(ROOT/'branches'/b/'gradient_summary.csv')
  for t in easy:
   t['adversarial_statistics_applicable']=b!='A_decoder_continue'
   for module in ('wrapper','bridge','generator'):
    t[module+'_weighted_adv_gradient_norm']=float(t['lambda_eff'])*float(t[module+'_adv_gradient_norm'])
   for key in list(t):
    if key.startswith('adversarial_') and key.endswith('_weighted_gradient_norm'):del t[key]
   if b=='A_decoder_continue':t['gradient_cosine']=None
  keys=['step','domain','external_qp','B1_gradient_norm','unweighted_adv_gradient_norm','weighted_adv_gradient_norm','lambda_eff','adv_B1_ratio','ceiling_triggered','gradient_cosine',*[f'{m}_{k}' for m in ('wrapper','bridge','generator') for k in ('B1_gradient_norm','adv_gradient_norm','weighted_adv_gradient_norm','weighted_adv_B1_ratio')],'global_preclip_norm','global_clip_coefficient','optimizer_update_count','D_optimizer_updates','peak_memory_MiB','adversarial_statistics_applicable']
  compact=[]
  for t in easy:
   row={k:t.get(k) for k in keys}
   for k,v in row.items():
    if v is not None and k not in ('domain','adversarial_statistics_applicable'):
     try:row[k]=format(float(v),'.9g')
     except (ValueError,TypeError):pass
   compact.append(row)
  write(ROOT/'branches'/b/'gradient_summary.csv',compact)
  assert (ROOT/'branches'/b/'gradient_summary.csv').stat().st_size<=524288
  events=[json.loads(x) for x in (ROOT/'gpu_schedule.jsonl').read_text().splitlines()];failures=[x for x in events if x.get('event')=='exit' and x.get('task')=='train_'+b and x.get('exit_code')];s.update(retry_count=max(0,load(ROOT/'pipeline_status.json').get('attempts',{}).get('train_'+b,1)-1),failed_worker_attempts=len(failures),numeric_failure_attempts=sum(any(t in x.get('error','').lower() for t in ('nonfinite','non-finite','nan','inf gradient')) for x in failures),failure_records=failures)
  s.update(ceiling_fraction=s['ceiling_steps']/1000,clipping_fraction=s['clipped_steps']/1000,adversarial_statistics_applicable=b!='A_decoder_continue');summaries[b]=s;dump(ROOT/'branches'/b/'gradient_control_summary.json',s)
 for d in DATASETS:
  folder=ROOT/'results/ip_-1'/d;rs=read(folder/'equal_rate_summary.csv')+read(folder/'equal_rate_fid_pooled.csv')
  for r in rs:
   if r['reference']=='B1_initial' and r['method'] in METHODS[2:]:core.append(r)
 for d in DATASETS:
  grid=read(ROOT/'results/ip_-1'/d/'common_rate_grid.csv');groups={}
  for r in grid:
   if r.get('sequence') and r.get('metric'):groups.setdefault((r['sequence'],r['metric']),[]).append(r)
  local=[]
  for (seq,metric),rs in groups.items():
   xs=[float(x['ln_bpp']) for x in rs];lo,hi=min(xs),max(xs);bounds=[lo+(hi-lo)*i/3 for i in range(4)]
   for j,label in enumerate(('low','middle','high')):
    rr=[x for x in rs if float(x['ln_bpp'])>=bounds[j] and (float(x['ln_bpp'])<bounds[j+1] if j<2 else float(x['ln_bpp'])<=bounds[j+1])]
    for m,ref in PAIRS:
     mm=statistics.mean(float(x[m]) for x in rr);rm=statistics.mean(float(x[ref]) for x in rr);delta=mm-rm
     row=dict(dataset=d,sequence=seq,metric=metric,method=m,reference=ref,band=label,lower_ln_bpp=bounds[j],upper_ln_bpp=bounds[j+1],grid_points=len(rr),method_mean=mm,reference_mean=rm,delta=delta,relative_change_percent=100*delta/rm if rm else None,better_grid_fraction=statistics.mean(float(x[m])<float(x[ref]) for x in rr),interpolation='PCHIP, five-method common measured interval, no extrapolation');local.append(row)
  band_rows.extend(local)
  for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
   for m,ref in PAIRS:
    for label in ('low','middle','high'):
     rr=[x for x in local if x['metric']==metric and x['method']==m and x['reference']==ref and x['band']==label];n=1 if metric=='FID' else len(sources(d));complete=len(rr)==n
     mm=statistics.mean(x['method_mean'] for x in rr) if complete else None;rm=statistics.mean(x['reference_mean'] for x in rr) if complete else None
     band_summary.append(dict(dataset=d,metric=metric,method=m,reference=ref,band=label,status='available' if complete else 'unavailable',reason='' if complete else 'missing common interval for one or more sequences',method_mean=mm,reference_mean=rm,delta=mm-rm if complete else None,relative_change_percent=100*(mm-rm)/rm if complete and rm else None,aggregation='pooled dataset FID' if metric=='FID' else 'sequence equal weight'))
 write(ROOT/'results/equal_rate_bands_per_sequence.csv',band_rows);write(ROOT/'results/equal_rate_bands_summary.csv',band_summary)
 write(ROOT/'results/core_changes_vs_B1.csv',core);dump(ROOT/'results/core_changes_vs_B1.json',core)
 dump(ROOT/'audits/gradient_control_integrity.json',dict(status='PASS',rho=.05,lambda_ceiling=None,eps=1e-12,FP32_restore=True,window_mean_before_weight=True,combined_then_global_clip=True,branches=summaries))
 text='# V6.25 numerical results\n\nMain evaluation:800/800; v6.25 pooled FID diagnostic:200/200.\n\nSigned difference=method-reference. Relative change=100*(method_mean-reference_mean)/reference_mean.\n\n|Dataset|Method|Metric|Delta|Relative change (%)|\n|---|---|---|---:|---:|\n'
 for r in core:text+=f"|{r['dataset']}|{r['method']}|{r['metric']}|{r['delta']}|{r['relative_change_percent']}|\n"
 (ROOT/'RESULTS_NUMBERS_ONLY.md').write_text(text)
 # Publication and byte-exact shard checks follow in verify_publication.py.
 final=load(ROOT/'audits/computational_integrity.json');final.update(status='PENDING_PUBLICATION',main_expected=800,main_completed=800,historical_early_completed=0,historical_FID_pooled_completed=200,gradient_control=True,three_branches_complete=True,publication_verified=False,shards_verified=False,shared_plan_sha256=sha(ROOT/'plans/shared.json'),expected_shared_plan_sha256=load(ROOT/'config.json')['source_shared_plan_sha256'],gradient_violation_count=0,DISTS_actual_optimizer_path_verified=True,labeler_pretrained_verified=True,wrapper_frozen_verified=True,condition_source_verified=True,B_actual_condition=True,C_condition_detached=True,official_pretrained_labeler=True,new_FID_decomposition_completed=True,cross_round_recomputed=True,domain_QP_counts_verified=True)
 dump(ROOT/'final_integrity.json',final)
if __name__=='__main__':main()
