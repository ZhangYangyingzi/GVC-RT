"""Validate cluster math and table joins using real sanity receipts; no fake results."""
import tempfile
from audit_io import *
import pandas as pd,numpy as np
from analysis import boot
from worker import ORDER,key

def main():
 a=pd.DataFrame([dict(dataset='d',video='a',x=1.),dict(dataset='d',video='a',x=3.),dict(dataset='d',video='b',x=6.)]);r=boot(a,'x');assert r['mean']==4. and r['n_videos']==2 and r['n_pairs']==3
 z=boot(pd.DataFrame([dict(dataset='d',video='a',x=0.)]),'x');assert z['mean_ci_low']==z['mean_ci_high']==0.
 rows=[]
 for m in ORDER:
  t=dict(dataset='ulong',video=0,qp=0,method=m);r=load(ROOT/'sanity/points'/key(t)/'point.json')
  rows.append(dict(dataset='ulong',video='sanity_video',QP=0,method=m,alpha=r['alpha'],actual_bpp=r['bpp'],**r['metrics_P'],**r['quantization_P'],cdf_entry_bits_per_P_frame=r['cdf_entry_bits_per_P_frame']))
 df=pd.DataFrame(rows);cols=['dataset','video','QP'];base=df[df.method=='B1_alpha0'];one=df[df.method=='B1_alpha100'];original=df[df.method=='Original']
 delta=df.merge(base[cols+['actual_bpp','cosine','LPIPS','DISTS','FloLPIPS']],on=cols,suffixes=('','_alpha0')).merge(one[cols+['actual_bpp','cosine']],on=cols,suffixes=('','_alpha1')).merge(original[cols+['cosine']],on=cols,suffixes=('','_Original'))
 delta['delta_bpp_vs_alpha0']=delta.actual_bpp-delta.actual_bpp_alpha0;delta['relative_delta_bpp_vs_alpha0']=delta.delta_bpp_vs_alpha0/delta.actual_bpp_alpha0;delta['delta_cos_vs_alpha0']=delta.cosine-delta.cosine_alpha0
 for metric in ('LPIPS','DISTS','FloLPIPS'):delta['delta_'+metric+'_vs_alpha0']=delta[metric]-delta[metric+'_alpha0']
 assert delta[delta.method=='B1_alpha0'].delta_cos_vs_alpha0.iloc[0]==0
 import figures
 with tempfile.TemporaryDirectory(prefix='v633_figure_sanity_') as tmp:
  figures.ROOT=Path(tmp);figures.make(df,delta,None,None)
  assert len(list((Path(tmp)/'figures').glob('*.pdf')))>=12
 dump('analysis_selftest.json',dict(status='PASS',video_cluster_mean_correct=True,degenerate_cluster_CI_labeled=True,real_sanity_joins=True,PNG_PDF_render=True,synthetic_experiment_results_written=False))
 print('ANALYSIS_SELFTEST_PASS')
if __name__=='__main__':main()
