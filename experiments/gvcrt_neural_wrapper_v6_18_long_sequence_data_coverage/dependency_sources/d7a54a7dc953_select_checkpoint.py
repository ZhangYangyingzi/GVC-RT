"""Validation-only common-interval PCHIP score; fixed 5000 remains primary."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from v614_io import *
def curve(rows,metric):
    pairs={}
    for r in rows:
        x=float(r['bpp']);y=float(r[metric]);assert x>0 and np.isfinite(y)
        if x in pairs and pairs[x]!=y:raise ValueError('conflicting values at identical bpp')
        pairs[x]=y
    if len(pairs)<2:raise ValueError('less than two distinct rates')
    x=sorted(pairs);return np.log(x),np.array([pairs[z] for z in x])
def select():
    from eval_adapter import validate
    assert load(ROOT/'training_status.json')['status']=='PASS';freeze=load(ROOT/'frozen_checkpoint_index.json');assert freeze['checkpoint_index_sha256']==sha(ROOT/'checkpoint_index.json')
    v=next(v for v in sources() if split(v)=='validation');curves={};raw=[]
    try:
        for s in STEPS:
            rows=[load(point('uvg',name(s),v['video_index'],q)) for q in range(10)]
            for q,r in enumerate(rows):validate(r,v,q,name(s))
            raw.extend(rows);curves[s]={metric:curve(rows,metric) for metric in ('LPIPS','DISTS')}
        lo=max(x[0] for c in curves.values() for x,y in c.values());hi=min(x[-1] for c in curves.values() for x,y in c.values())
        if hi<=lo:raise ValueError('all-candidate common interval is empty')
        grid=np.linspace(lo,hi,100);scores=[];gridrows=[]
        for s,c in curves.items():
            val={k:PchipInterpolator(x,y,extrapolate=False)(grid) for k,(x,y) in c.items()};total=val['LPIPS']+val['DISTS'];assert np.isfinite(total).all()
            scores.append(dict(adaptation_step=s,method=name(s),score=float(total.mean())))
            gridrows.extend(dict(adaptation_step=s,grid_index=i,ln_bpp=float(g),bpp=float(np.exp(g)),LPIPS=float(val['LPIPS'][i]),DISTS=float(val['DISTS'][i]),score=float(total[i])) for i,g in enumerate(grid))
        chosen=min(scores,key=lambda r:(r['score'],r['adaptation_step']))
        result=dict(status='available',primary_step=5000,selected_step=chosen['adaptation_step'],selected_method=name(chosen['adaptation_step']),scores=scores,common_min_bpp=float(np.exp(lo)),common_max_bpp=float(np.exp(hi)),ln_bpp_grid=grid.tolist(),points=100,criterion='mean LPIPS+DISTS over common ln(bpp) grid; exact tie earliest',dataset='Bosphorus validation only',holdout_used=False,frozen_checkpoint_index_sha256=sha(ROOT/'frozen_checkpoint_index.json'))
        write(ROOT/'results/validation/checkpoint_selection_grid.csv',gridrows);write(ROOT/'results/validation/checkpoint_selection_scores.csv',scores)
    except ValueError as e:result=dict(status='unavailable',reason=str(e),primary_step=5000,selected_step=None,selected_method=None,holdout_used=False)
    dump(ROOT/'validation_selection.json',result);write(ROOT/'results/validation/checkpoint_history_raw.csv',raw)
    return result
def final_methods():
    methods=['original','v62_initial',name(5000)];s=load(ROOT/'validation_selection.json')
    if s['status']=='available' and s['selected_step'] not in (0,5000):methods.append(name(s['selected_step']))
    return methods
if __name__=='__main__':select()
