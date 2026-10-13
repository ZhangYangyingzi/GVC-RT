from io21 import *
from comparison import PAIRS
import statistics

def main():
 allrows=[];sums=[]
 for d in (*DATASETS,'mcljcv_exclude_overlap29'):
  folder=ROOT/'results/ip_-1'/d;grid=read(folder/'common_rate_grid.csv');groups={}
  for r in grid:
   if 'ln_bpp' in r:groups.setdefault((r['sequence'],r['metric']),[]).append(r)
  for (sequence,metric),rs in groups.items():
   lo=min(float(x['ln_bpp']) for x in rs);hi=max(float(x['ln_bpp']) for x in rs);bounds=[lo+(hi-lo)*i/3 for i in range(4)]
   for j,band in enumerate(('low','mid','high')):
    selected=[x for x in rs if float(x['ln_bpp'])>=bounds[j] and (float(x['ln_bpp'])<bounds[j+1] if j<2 else float(x['ln_bpp'])<=bounds[j+1])]
    for m,ref in PAIRS:
     mm=statistics.mean(float(x[m]) for x in selected);rm=statistics.mean(float(x[ref]) for x in selected)
     allrows.append(dict(dataset=d,sequence=sequence,metric=metric,band=band,method=m,reference=ref,method_mean=mm,reference_mean=rm,delta=mm-rm,relative_change_percent=100*(mm-rm)/rm if rm else None,lower_ln_bpp=bounds[j],upper_ln_bpp=bounds[j+1],points=len(selected),interpolation='main PCHIP; no extrapolation'))
  for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
   for band in ('low','mid','high'):
    for m,ref in PAIRS:
     rs=[r for r in allrows if (r['dataset'],r['metric'],r['band'],r['method'],r['reference'])==(d,metric,band,m,ref)]
     expected=1 if metric=='FID' else 29 if d.endswith('29') else len(sources(d))
     complete=len(rs)==expected;mm=statistics.mean(x['method_mean'] for x in rs) if complete else None;rm=statistics.mean(x['reference_mean'] for x in rs) if complete else None
     sums.append(dict(dataset=d,metric=metric,band=band,method=m,reference=ref,status='available' if complete else 'unavailable',method_mean=mm,reference_mean=rm,delta=mm-rm if complete else None,relative_change_percent=100*(mm-rm)/rm if complete and rm else None,expected_sequences=expected,available_sequences=len(rs)))
 write(ROOT/'results/equal_rate_bands_per_sequence.csv',allrows);write(ROOT/'results/equal_rate_bands_summary.csv',sums)
if __name__=='__main__':main()
