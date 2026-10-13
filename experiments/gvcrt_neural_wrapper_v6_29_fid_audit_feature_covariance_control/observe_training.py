"""Readable live covariance gradient summary; never edits training or checkpoints."""
from io21 import *
def main():
 last=0
 while True:
  p=ROOT/'branches/B_covstats/training_logs/train.jsonl'
  if p.exists():
   lines=p.read_text().splitlines();rows=[]
   for x in lines:
    try:rows.append(json.loads(x))
    except json.JSONDecodeError:break
   if len(rows)!=last:
    table=[]
    for r in rows:
     table.append(dict(step=r['adaptation_step'],domain=r['domain'],QP=r['external_qp'],B1_gradient_norm=r['B1_gradient_norm'],unit_stat_gradient_norm=r['stat_gradient_norm'],lambda_stat=r['lambda_stat'],weighted_stat_gradient_norm=r['weighted_stat_gradient_norm'],weighted_stat_B1_ratio=r['stat_B1_ratio'],gradient_cosine=r['stat_cosine'],L_stat=r['stat_loss'],weighted_L_stat=r['lambda_stat']*r['stat_loss'],global_preclip_norm=r['global_preclip_norm'],clip_coefficient=r['global_clip_coefficient'],**{k:r[k] for k in ('bridge_B1_gradient_norm','generator_B1_gradient_norm','bridge_adv_gradient_norm','generator_adv_gradient_norm')}))
    write(ROOT/'branches/B_covstats/stat_gradient_summary.csv',table);last=len(rows);print('SUMMARY',last,flush=True)
   if len(rows)==1000:return
  time.sleep(30)
if __name__=='__main__':main()
