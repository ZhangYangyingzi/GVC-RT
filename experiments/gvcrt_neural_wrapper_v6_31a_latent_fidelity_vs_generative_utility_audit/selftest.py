"""Analytical checks for matching, metrics, clustered uncertainty and empty evidence."""
import ast
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import audit_io
from audit_io import *


def main():
    from analysis import match_rates,correlate,cluster_ci,residuals,enrich,LATENTS,QUALITY
    # Every candidate must be included; a lower-rate exact tie wins without reading quality.
    rows=[dict(dataset='d',video='v',method=m,QP=q,actual_bpp=r,quality=quality)
          for m,q,r,quality in [('Original',0,1.,99),('Original',1,2.,-99),('B1',0,1.,-10),('B1',1,2.,10),('B_covstats',0,1.015,0)]]
    points=pd.DataFrame(rows);allpairs,primary=match_rates(points)
    assert len(allpairs)==3 and len(primary)==1
    assert primary.iloc[0].original_QP==primary.iloc[0].method_QP==0
    assert allpairs[allpairs.method=='B_covstats'].iloc[0].rate_match_level=='2pct'
    points.quality=-points.quality
    assert match_rates(points)[1].equals(primary)
    far=points.copy();far.loc[far.method!='Original','actual_bpp']=10
    assert match_rates(far)[0].empty and match_rates(far)[1].empty
    assert correlate([1,2,3],[3,2,1])['Spearman']==-1
    assert correlate([1,1,1],[2,3,4])['Pearson'] is None
    clusters=pd.DataFrame(dict(video=['a']*3+['b']*3,x=[1,2,3,4,5,6],y=[2,4,6,8,10,12]))
    ci=cluster_ci(clusters,'x','y',50)
    assert ci['video_clusters']==2 and ci['Pearson_low']>.999
    rate=pd.DataFrame(dict(video=['a']*10+['b']*10,method=['Original']*20,actual_bpp=np.exp(np.tile(np.arange(10)/10,2))))
    rate['x']=2*np.log(rate.actual_bpp)+np.repeat([0,3],10);rate['y']=-4*np.log(rate.actual_bpp)+np.repeat([0,7],10)
    rx,ry=residuals(rate,'x','y');assert np.max(np.abs(rx))<1e-12 and np.max(np.abs(ry))<1e-12
    import subprocess
    # Use the existing separate GPU and CPU analysis environments without changing either.
    code='''import sys
sys.path.insert(0,sys.argv[1])
import torch
from proxy_runtime import latent_metrics
torch.manual_seed(5);target=torch.randn(1,18,68,120)
metrics=latent_metrics(target,target)
assert metrics['L1']==metrics['RMSE']==metrics['normalized_L2']==0 and abs(metrics['cosine'])<1e-12
assert abs(latent_metrics(-target,target)['cosine']-2)<1e-12
'''
    subprocess.run(['/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python','-B','-c',code,str(ROOT)],check=True)
    # Exercise report generation when the pre-registered matching yields no evidence.
    import plots_report
    actual_root=audit_io.ROOT
    with tempfile.TemporaryDirectory(dir=ROOT,prefix='selftest_') as tmp:
        tmp=Path(tmp);(tmp/'figures/selected_counterexamples').mkdir(parents=True)
        audit_io.ROOT=tmp;plots_report.ROOT=tmp
        try:
            for scope in ('P','all','I'):
                for metric in (*LATENTS,*QUALITY):far[scope+'_'+metric]=.5
            empty=enrich(match_rates(far)[1],far)
            inv=pd.DataFrame(columns=[*empty.columns,'latent_metric','quality_metric','frame_scope','is_inversion'])
            sums=pd.DataFrame(columns=['population','frame_scope','latent_metric','groupby','quality_metric','group','inversion_fraction'])
            plots_report.finish(far,empty,inv,sums,{m:0 for m in LATENTS})
            assert load(tmp/'conclusion.json')['case']=='Not assessable'
            positive=[]
            for i in range(4):
                for method in ('Original','B1'):
                    r=dict(dataset='d'+str(i//2),video='v'+str(i),method=method,QP=i%2,actual_bpp=.01+i*.001)
                    for scope in ('P','all','I'):
                        for metric in (*LATENTS,*QUALITY):r[scope+'_'+metric]=(.6 if metric in LATENTS else .4) if method=='B1' else .5
                    positive.append(r)
            positive=pd.DataFrame(positive);pp=enrich(match_rates(positive)[1],positive)
            ii=pp.copy();ii['latent_metric']='cosine';ii['quality_metric']='LPIPS';ii['frame_scope']='P';ii['is_inversion']=True
            plots_report.finish(positive,pp,ii,sums,{m:0 for m in LATENTS})
            assert load(tmp/'conclusion.json')['case']=='Strong'
            assert load(tmp/'counterexample_selection.json')['count']==4
        finally:audit_io.ROOT=actual_root;plots_report.ROOT=actual_root
    for p in ROOT.glob('*.py'):ast.parse(p.read_text())
    dump('selftest.json',dict(status='PASS',checks=['quality-blind matching','all qualifying pairs','1% vs 2% separation',
         'lower-rate tie breaking','no-match empty case','analytic cosine/L1/RMSE','constant correlation undefined',
         'video cluster bootstrap','log-rate residual fixed effects','no-evidence figures/report','all Python parses']))
    print('SELFTEST_PASS',flush=True)


if __name__=='__main__':main()
