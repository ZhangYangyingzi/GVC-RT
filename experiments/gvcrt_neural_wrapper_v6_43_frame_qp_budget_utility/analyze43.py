"""Fixed-window descriptive statistics and preregistered oracle transfer selection."""
from common43 import *
import argparse,math,itertools,statistics
METRICS=('LPIPS','DISTS','L1')
def tables():
 plan=load(ROOT/'intervention_plan.json');assert all(validate_result(t) for t in plan)
 summaries=[];deltas=[];transition_deltas=[]
 for t in plan:
  if t['kind']=='baseline':continue
  base=ROOT/'tasks'/t['dataset']/t['sequence']/'baseline'
  b=read(base/'frames.csv');x=read(result_path(t).parent/'frames.csv');target=t['target']
  db=[float(a['real_bits'])-float(z['real_bits']) for a,z in zip(x,b)]
  common=dict(dataset=t['dataset'],sequence=t['sequence'],variant=t['variant'],target=target,label=t['label'])
  for m in METRICS:
   gains=[float(z[m])-float(a[m]) for a,z in zip(x,b)]
   row=dict(**common,metric=m,G_now=gains[target],G_future15=sum(gains[target+1:target+16]),G_16=sum(gains[target:target+16]),delta_bits_now=db[target],delta_bits_future15=sum(db[target+1:target+16]),delta_bits_16=sum(db[target:target+16]),G_tail=sum(gains[target:]),delta_bits_tail=sum(db[target:]),G_full_sum=sum(gains),G_full_mean=sum(gains)/64,delta_bits_full=sum(db))
   for k in (0,4,8,15):row.update({f'G_cumulative_{k}':sum(gains[target:target+k+1]),f'delta_bits_cumulative_{k}':sum(db[target:target+k+1])})
   cost=row['delta_bits_16'];eligible=cost>0 and row['delta_bits_full']>0
   row.update(U_now=1000*row['G_now']/cost if eligible else None,U_16=1000*row['G_16']/cost if eligible else None,U_now_own_cost=1000*row['G_now']/row['delta_bits_now'] if row['delta_bits_now']>0 else None,positive_cost_eligible=eligible,cost_type='positive_full_and_window' if eligible else 'nonpositive_window' if cost<=0 else 'nonpositive_full',saves_bits_and_improves=cost<0 and row['G_16']>0)
   summaries.append(row)
  for i,(a,z) in enumerate(zip(x,b)):
   deltas.append(dict(**common,frame=i,delta_bits=db[i],**{f'gain_{m}':float(z[m])-float(a[m]) for m in (*METRICS,'pixel_SSE')},delta_PSNR_dB=float(a['PSNR'])-float(z['PSNR'])))
  for a,z in zip(read(result_path(t).parent/'transitions.csv'),read(base/'transitions.csv')):
   transition_deltas.append(dict(**common,from_frame=int(a['from_frame']),to_frame=int(a['to_frame']),gain_FloLPIPS=float(z['FloLPIPS'])-float(a['FloLPIPS'])))
 write(ROOT/'impulse_summary.csv',summaries);write(ROOT/'frame_differences.csv',deltas);write(ROOT/'transition_differences.csv',transition_deltas)
 write(ROOT/'saving_improving_impulses.csv',[r for r in summaries if r['saves_bits_and_improves']],fields=list(summaries[0]))
 write(ROOT/'nonpositive_cost_impulses.csv',[r for r in summaries if not r['positive_cost_eligible']],fields=list(summaries[0]))
 ranking=[];comparison=[]
 from scipy.stats import spearmanr,rankdata
 for seq in ('ParkScene','Jockey'):
  for m in METRICS:
   ss=[r for r in summaries if r['sequence']==seq and r['metric']==m and r['positive_cost_eligible']]
   now=sorted(ss,key=lambda r:(-r['U_now'],r['target'],r['label']));future=sorted(ss,key=lambda r:(-r['U_16'],r['target'],r['label']))
   nr=rankdata([-r['U_now'] for r in ss],method='average');fr=rankdata([-r['U_16'] for r in ss],method='average')
   for r,n,f in zip(ss,nr,fr):ranking.append(dict(**r,rank_now=float(n),rank_16=float(f)))
   inversions=[dict(first=a['variant'],second=b['variant']) for a,b in itertools.combinations(ss,2) if (a['U_now']-b['U_now'])*(a['U_16']-b['U_16'])<0]
   corr=float(spearmanr([r['U_now'] for r in ss],[r['U_16'] for r in ss]).statistic) if len(ss)>=3 and len(set(r['U_now'] for r in ss))>1 and len(set(r['U_16'] for r in ss))>1 else None
   comparison.append(dict(sequence=seq,metric=m,eligible_count=len(ss),now_order=[r['variant'] for r in now],future_order=[r['variant'] for r in future],top_two_set_same=set(r['variant'] for r in now[:2])==set(r['variant'] for r in future[:2]) if len(ss)>=2 else None,top_two_order_same=[r['variant'] for r in now[:2]]==[r['variant'] for r in future[:2]] if len(ss)>=2 else None,pairwise_reversals=inversions,descriptive_spearman=corr))
 write(ROOT/'utility_rankings.csv',ranking,fields=list(ranking[0]) if ranking else ['sequence','metric','variant','U_now','U_16','rank_now','rank_16']);dump(ROOT/'ranking_comparison.json',comparison)
 return summaries

def select(summaries):
 selection=[];plan=[]
 for seq in ('ParkScene','Jockey'):
  rows=[r for r in summaries if r['sequence']==seq and r['metric']=='LPIPS'];dataset=rows[0]['dataset']
  receivers=[r for r in rows if r['label']=='qplus2' and r['delta_bits_now']>0 and r['delta_bits_16']>0]
  donors=[r for r in rows if r['label']=='qminus2' and r['delta_bits_now']<0 and r['delta_bits_16']<0]
  choices=[]
  for r,d in itertools.product(receivers,donors):
   if abs(r['target']-d['target'])>=16:
    A=r['G_16']/r['delta_bits_16'];C=-d['G_16']/(-d['delta_bits_16'])
    choices.append(dict(r=r['target'],d=d['target'],A=A,C=C,score=A-C))
  choices.sort(key=lambda x:(-x['score'],x['r'],x['d']))
  if not choices:
   selection.append(dict(sequence=seq,status='NO_LEGAL_PAIR',reason='No pair satisfies actual-cost signs and separation >=16',receiver_candidates=receivers,donor_candidates=donors));continue
  chosen=choices[0];selection.append(dict(sequence=seq,status='SELECTED',selected=chosen,all_pairs=choices))
  base=next(t for t in load(ROOT/'intervention_plan.json') if t['sequence']==seq and t['kind']=='baseline')
  for direction,sign in [('forward',1),('reverse',-1)]:
   for a,b in ((1,1),(1,2),(2,1),(2,2)):
    r,d=chosen['r'],chosen['d'];q=base['actual_qps'].copy();q[r]+=sign*a;q[d]-=sign*b
    assert all(0<=v<10 for v in q)
    plan.append(dict(dataset=dataset,sequence=seq,variant=f'{direction}_r{r}_d{d}_a{a}_b{b}',kind='transfer',direction=direction,r=r,d=d,a=a,b=b,changes={str(r):sign*a,str(d):-sign*b},actual_qps=q))
 if (ROOT/'transfer_selection.json').exists():assert load(ROOT/'transfer_selection.json')==selection and load(ROOT/'transfer_plan.json')==plan
 else:dump(ROOT/'transfer_selection.json',selection);dump(ROOT/'transfer_plan.json',plan)
 print('TRANSFER_PLAN',len(plan),flush=True)

def transfers():
 plan=load(ROOT/'transfer_plan.json');assert all(validate_result(t) for t in plan)
 rows=[];chosen=[]
 for t in plan:
  r=load(result_path(t))['result'];base=next(x for x in load(ROOT/'intervention_plan.json') if x['kind']=='baseline' and x['sequence']==t['sequence']);br=load(result_path(base))['result'];ratio=r['total_bits']/br['total_bits']
  rows.append(dict(**{k:t[k] for k in ('dataset','sequence','variant','direction','r','d','a','b')},**{k:r[k] for k in ('LPIPS','DISTS','L1','PSNR','FloLPIPS','total_bits','actual_bpp')},baseline_bits=br['total_bits'],baseline_LPIPS=br['LPIPS'],delta_bits=r['total_bits']-br['total_bits'],budget_ratio=ratio,strict_feasible=ratio<=1,near_budget=.99<=ratio<=1,saved_percent=100*(1-ratio),delta_LPIPS=r['LPIPS']-br['LPIPS'],delta_DISTS=r['DISTS']-br['DISTS'],delta_L1=r['L1']-br['L1'],delta_PSNR_dB=r['PSNR']-br['PSNR'],delta_FloLPIPS=r['FloLPIPS']-br['FloLPIPS']))
 for seq in ('ParkScene','Jockey'):
  for direction in ('forward','reverse'):
   candidates=[r for r in rows if r['sequence']==seq and r['direction']==direction and r['near_budget']]
   if candidates:chosen.append(dict(status='SELECTED',**min(candidates,key=lambda r:(r['LPIPS'],r['total_bits'],r['a'],r['b']))))
   else:chosen.append(dict(sequence=seq,direction=direction,status='NO_NEAR_BUDGET_MATCH',reason='No candidate in fixed finite grid has total bits in [0.99 B0,B0]'))
 write(ROOT/'transfer_candidates.csv',rows,fields=list(rows[0]) if rows else ['sequence','direction','variant','total_bits','LPIPS','strict_feasible','near_budget']);write(ROOT/'budget_feasible_comparisons.csv',chosen)
 return rows,chosen
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['select','finish']);a=p.parse_args();ss=tables();select(ss)
 if a.phase=='finish':transfers()
