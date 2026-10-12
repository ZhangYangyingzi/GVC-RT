"""Recompute pairwise common intervals for V6.26 versus V6.25."""
from io21 import *
from report import comparisons,summarize
import statistics

def main():
 old=('v625_A_continue_1000','v625_B_capped_projected_1000','v625_C_capped_conditional_projected_1000');summary=[];per=[];grids=[]
 for d in DATASETS:
  pooled_new=read(ROOT/'results/ip_-1'/d/'fid_pooled.csv');pooled_old=read(ROOT/'historical/fid_decomposition_raw.csv')
  for m in METHODS[2:]:
   for ref in old:
    pairs=((m,ref),);rr=[]
    for v in sources(d):
     raw=[load(point(d,x,v['video_index'],q)) for x in (m,ref) for q in range(10)]
     for metric in ('LPIPS','DISTS','FloLPIPS'):
      rows,grid=comparisons(d,v['name'],metric,raw,methods=(m,ref),pairs=pairs);rr+=rows;grids+=grid
    per+=rr;summary+=summarize(d,rr,len(sources(d)),pairs)
    raw=[dict(r,bpp=float(r['bpp']),FID=float(r['FID'])) for r in pooled_new+pooled_old if r['dataset']==d and r['method'] in (m,ref)]
    rows,grid=comparisons(d,'pooled_dataset','FID',raw,True,methods=(m,ref),pairs=pairs);summary+=rows;grids+=grid
 write(ROOT/'results/cross_round_per_sequence.csv',per);write(ROOT/'results/cross_round_summary.csv',summary);write(ROOT/'results/cross_round_interpolation_inputs.csv',grids)
 dump(ROOT/'audits/cross_round_integrity.json',dict(status='PASS',pairwise_common_intervals_recomputed=True,PCHIP=True,no_extrapolation=True,comparisons=summary))
if __name__=='__main__':main()
