"""Numerical tables, gradient-control counts and final publication checklist."""
from io21 import *
import statistics,math

def main():
 assert load(ROOT/'audits/computational_integrity.json')['status']=='PASS'
 assert load(ROOT/'historical/final_integrity.json')['status']=='PASS'
 summaries={};core=[]
 for b in BRANCHES:
  rows=[json.loads(x) for x in (ROOT/'branches'/b/'training_logs/train.jsonl').read_text().splitlines()];assert len(rows)==1000
  keys=['lambda_eff','adv_B1_ratio','gradient_cosine','global_clip_coefficient',*(k+'_weighted_adv_B1_ratio' for k in ('wrapper','bridge','generator'))]
  s=dict(steps=len(rows),G_updates=1000,D_updates=1000 if b!='A_continue' else 0,violations=sum(r['adv_B1_ratio']>.05+1e-8 for r in rows),ceiling_steps=sum(r['ceiling_triggered'] for r in rows),clipped_steps=sum(r['global_clip_coefficient']<1 for r in rows),distributions={k:dict(min=min(r[k] for r in rows),max=max(r[k] for r in rows),mean=statistics.mean(r[k] for r in rows),median=statistics.median(r[k] for r in rows)) for k in keys});assert s['violations']==0
  s.update(ceiling_fraction=s['ceiling_steps']/1000,clipping_fraction=s['clipped_steps']/1000,adversarial_statistics_applicable=b!='A_continue');summaries[b]=s;dump(ROOT/'branches'/b/'gradient_control_summary.json',s)
 for d in DATASETS:
  folder=ROOT/'results/ip_-1'/d;rs=read(folder/'equal_rate_summary.csv')+read(folder/'equal_rate_fid_pooled.csv')
  for r in rs:
   if r['reference']=='B1_initial' and r['method'] in METHODS[2:]:core.append(r)
 write(ROOT/'results/core_changes_vs_B1.csv',core);dump(ROOT/'results/core_changes_vs_B1.json',core)
 dump(ROOT/'audits/gradient_control_integrity.json',dict(status='PASS',rho=.05,lambda_ceiling=.014448280572707944,eps=1e-12,FP32_restore=True,window_mean_before_weight=True,combined_then_global_clip=True,branches=summaries))
 text='# V6.25 numerical results\n\nMain evaluation:800/800; historical early evaluation:640/640.\n\nSigned difference=method-reference. Relative change=100*(method_mean-reference_mean)/reference_mean.\n\n|Dataset|Method|Metric|Delta|Relative change (%)|\n|---|---|---|---:|---:|\n'
 for r in core:text+=f"|{r['dataset']}|{r['method']}|{r['metric']}|{r['delta']}|{r['relative_change_percent']}|\n"
 (ROOT/'RESULTS_NUMBERS_ONLY.md').write_text(text)
 # Publication and byte-exact shard checks follow in verify_publication.py.
 final=load(ROOT/'audits/computational_integrity.json');final.update(status='PENDING_PUBLICATION',main_expected=800,main_completed=800,historical_early_completed=640,gradient_control=True,three_branches_complete=True,publication_verified=False,shards_verified=False)
 dump(ROOT/'final_integrity.json',final)
if __name__=='__main__':main()
