"""Numeric-only paired comparisons, 100 uniform log-rate points, no extrapolation."""
import math,statistics
import numpy as np
from scipy.interpolate import PchipInterpolator
from v68_io import *
PAIRS=(('A5000','original'),('B5000','original'),('B5000','A5000'),('A5000','v62'),('B5000','v62'))
def curve(rows,metric):
    grouped={}
    for r in rows:
        x=float(r['kbps']);y=float(r[metric])
        if not math.isfinite(x) or x<=0 or not math.isfinite(y):raise ValueError('nonfinite or nonpositive rate')
        if x in grouped and grouped[x]!=y:raise ValueError('conflicting metric values at identical bitrate; raw points preserved')
        grouped[x]=y
    if len(grouped)<2:raise ValueError('fewer than two distinct rates')
    rates=sorted(grouped);return np.log(rates),np.array([grouped[x] for x in rates])
def compare(a,b,metric):
    try:
        x,y=curve(a,metric);u,v=curve(b,metric);lo=max(x[0],u[0]);hi=min(x[-1],u[-1])
        if hi<=lo:raise ValueError('no positive-length common log-rate interval')
        grid=np.linspace(lo,hi,100);av=PchipInterpolator(x,y,extrapolate=False)(grid);bv=PchipInterpolator(u,v,extrapolate=False)(grid)
        assert np.isfinite(av).all() and np.isfinite(bv).all()
        return dict(status='available',reason='',common_min_kbps=float(np.exp(lo)),common_max_kbps=float(np.exp(hi)),grid_points=100,mean_method=float(av.mean()),mean_reference=float(bv.mean()),delta=float((av-bv).mean()))
    except ValueError as e:return dict(status='unavailable',reason=str(e),grid_points=0,delta=None)
def main(raw,fid,bootstrap):
    per=[];summary=[];pooled=[];draw_results=[]
    for d in DATASETS:
        for a,b in PAIRS:
            for metric in ('LPIPS','DISTS','FloLPIPS'):
                rows=[]
                for v in sources(d):
                    group=[r for r in raw if r['dataset']==d and r['video_index']==v['video_index']]
                    r=dict(dataset=d,method=a,reference=b,metric=metric,video_index=v['video_index'],sequence=v['name'],**compare([r for r in group if r['method']==a],[r for r in group if r['method']==b],metric))
                    per.append(r);rows.append(r)
                valid=[r for r in rows if r['status']=='available']
                summary.append(dict(dataset=d,method=a,reference=b,metric=metric,status='available' if len(valid)==len(rows) else 'unavailable',available_sequences=len(valid),expected_sequences=len(rows),delta=statistics.mean(r['delta'] for r in valid) if len(valid)==len(rows) else None,partial_available_delta=statistics.mean(r['delta'] for r in valid) if valid else None,video_weighting='equal'))
            group=[r for r in fid if r['dataset']==d]
            pooled.append(dict(dataset=d,method=a,reference=b,metric='pooled_FID',**compare([r for r in group if r['method']==a],[r for r in group if r['method']==b],'FID')))
            for repeat in range(20):
                samples=[]
                for r in bootstrap:
                    if r['dataset']!=d or int(r['repeat'])!=repeat or r['method'] not in (a,b):continue
                    axis=next(x for x in group if x['method']==r['method'] and x['QP']==int(r['QP']))
                    samples.append(dict(method=r['method'],FID=float(r['FID']),kbps=axis['kbps']))
                draw_results.append(dict(dataset=d,method=a,reference=b,repeat=repeat,metric='pooled_FID',**compare([r for r in samples if r['method']==a],[r for r in samples if r['method']==b],'FID')))
    write(ROOT/'results/equal_rate_per_sequence.csv',per);write(ROOT/'results/equal_rate_summary.csv',summary)
    write(ROOT/'results/equal_rate_pooled_fid.csv',pooled);write(ROOT/'results/equal_rate_fid_bootstrap.csv',draw_results)
    dump(ROOT/'audits/equal_rate_protocol.json',dict(status='PASS',interpolator='PCHIP',axis='natural log actual kbps',grid_points=100,no_extrapolation=True,no_monotonic_rewrite=True,duplicate_policy='exact equal points collapse; conflicting values unavailable',video_weighting='equal',FID='pooled only; original paired bootstrap indices, dataset rate axis'))
