"""Analytic constant-offset check for new8-model contrasts; no codec/model calls."""
from io32 import *
import math

def main():
 import compare33 as c
 fixture=ROOT/'cache/analytic_comparison';fixture.mkdir(parents=True,exist_ok=True);c.ROOT=fixture
 manifest={d:[dict(video_index=0,name='synthetic_only_not_video')] for d in c.DATASETS if d!='mcljcv29'};dump(fixture/'eval_manifest.json',manifest)
 offsets={'original':0.,'B1_initial':.5}
 for b,x,y in zip(BRANCHES,(1.,2.,3.),(4.,6.,7.)):
  offsets[f'{b}_1000']=x;offsets[f'{b}_2000']=y;offsets[f'{b}_1500']=(x+y)/2
 records=[];pools=[]
 for d in manifest:
  for m,offset in offsets.items():
   for q in ([0,5,9] if m.endswith('_1500') else range(10)):
    x=.01*(q+1);y=10+math.log(x)+offset
    records.append(dict(dataset=d,video=0,method=m,QP=q,bpp=x,LPIPS=y,DISTS=y,FloLPIPS=y))
 for d in c.DATASETS:
  for m,offset in offsets.items():
   for q in ([0,5,9] if m.endswith('_1500') else range(10)):
    x=.01*(q+1);y=10+math.log(x)+offset;pools.append(dict(dataset=d,method=m,QP=q,bpp=x,FID=y,D_mu=.4*y,D_cov=.6*y,status='available'))
 summary,_=c.run(records,pools)
 checks=[]
 for r in summary:
  assert r['status']=='available'
  expected=1. if r['contrast']=='difference_of_differences' else offsets[r['method']]-offsets[r['reference']]
  if r['metric']=='D_mu':expected*=.4
  if r['metric']=='D_cov':expected*=.6
  error=abs(r['delta']-expected);assert error<=1e-12,(r,error)
  checks.append(error)
 # Deliberately disjoint measured rates must preserve unavailable rows and labelled fallback.
 for x in records:
  if x['dataset']=='ulong' and x['method']==f'{BRANCHES[1]}_2000':x['bpp']*=100
 for x in pools:
  if x['dataset']=='ulong' and x['method']==f'{BRANCHES[1]}_2000':x['bpp']*=100
 bad,_=c.run(records,pools);unavailable=[x for x in bad if x['dataset']=='ulong' and x['phase']=='main_2000' and x['metric']=='LPIPS'];assert unavailable and all(x['status']=='unavailable' for x in unavailable)
 assert any(x['contrast']=='difference_of_differences' for x in unavailable)
 fb=read(fixture/'pairwise_fallback.csv');assert fb and all('PAIRWISE ONLY' in x['interval_scope'] for x in fb)
 dump(ROOT/'audits/comparison_arithmetic.json',dict(status='PASS',synthetic_only=True,missing_common_interval_preserved=True,pairwise_fallback_explicitly_labelled=True,no_model_or_codec_calls=True,optimizer_updates=0,real_evaluation_points=0,checked_contrasts=len(checks),maximum_absolute_error=max(checks),tolerance=1e-12,fixture='constant offsets in ln(bpp); C2000-B2000 gap2 minus C1000-B1000 gap1 equals1; both all-rate and three fixed bands',fixture_tables_server_only_in_cache=True))
 print('CONTRAST ARITHMETIC PASS',len(checks),max(checks))
if __name__=='__main__':main()
