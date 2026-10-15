"""Fixed figures from all points, without visual example selection."""
from audit_io import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def make(df,delta,summary,trend):
 folder=ROOT/'figures';folder.mkdir(exist_ok=True)
 def finish(fig,name):
  fig.tight_layout()
  for ext in ('png','pdf'):fig.savefig(folder/(name+'.'+ext),dpi=170,bbox_inches='tight')
  plt.close(fig)
 sweep=df[df.method!='Original']
 for col,name,label in [('actual_bpp','alpha_vs_actual_bpp','Actual total bpp (headers + I/P)'),('cosine','alpha_vs_teacher_cosine','Teacher proxy cosine distance (P frames)'),('LPIPS','alpha_vs_lpips','LPIPS (P frames)'),('DISTS','alpha_vs_dists','DISTS (P frames)'),('FloLPIPS','alpha_vs_flolpips','FloLPIPS (P transitions)'),('quant_error_RMSE','quantization_error_vs_alpha','Raw analysis-y quantization RMSE'),('fraction_quantized_symbols_changed','symbol_change_fraction_vs_alpha','Centered integer symbols changed vs alpha0'),('cdf_entry_bits_per_P_frame','entropy_bits_vs_alpha','Actual RANS CDF entry bits / P frame\n(excludes escape bypass and z)')]:
  fig,ax=plt.subplots(figsize=(7,4.5))
  for d,g in sweep.groupby('dataset'):
   s=g.groupby('alpha')[col].mean();ax.plot(s.index,s.values,'o-',label=d)
  ax.set(xlabel='Source P strength alpha',ylabel=label,title='Fixed video pool and QP0–9; video/point means')
  ax.grid(alpha=.25);ax.legend(fontsize=8)
  if col in ('LPIPS','DISTS','FloLPIPS'):fig.text(.5,-.02,'alpha !=1: local intervention, B1 receiver not readapted',ha='center',fontsize=8)
  finish(fig,name)
 for y,name,x,label in [('delta_bpp_vs_alpha0','delta_teacher_vs_delta_bpp','delta_cos_vs_alpha0','Delta actual bpp'),('delta_LPIPS_vs_alpha0','delta_teacher_vs_delta_lpips','delta_cos_vs_alpha0','Delta LPIPS'),('delta_DISTS_vs_alpha0','delta_teacher_vs_delta_dists','delta_cos_vs_alpha0','Delta DISTS'),('delta_FloLPIPS_vs_alpha0','delta_teacher_vs_delta_flolpips','delta_cos_vs_alpha0','Delta FloLPIPS'),('delta_LPIPS_vs_alpha0','delta_bpp_vs_delta_lpips','delta_bpp_vs_alpha0','Delta LPIPS')]:
  fig,ax=plt.subplots(figsize=(7,5));s=delta[(delta.method!='Original')&(delta.alpha!=0)]
  for d,g in s.groupby('dataset'):ax.scatter(g[x],g[y],s=8,alpha=.3,label=d)
  ax.axhline(0,color='k',lw=.7);ax.axvline(0,color='k',lw=.7);ax.set(xlabel='Delta teacher cosine vs alpha0 (positive=worse)' if x=='delta_cos_vs_alpha0' else 'Delta actual bpp vs alpha0',ylabel=label+' vs alpha0',title='Same video + external QP; all nonzero alpha interventions')
  if y=='delta_bpp_vs_alpha0':ax.text(.98,.03,'Teacher worse / rate better\n(+ cosine, - bpp)',transform=ax.transAxes,ha='right',fontsize=9)
  else:fig.text(.5,-.02,'Association; alpha !=1 receiver is not independently optimized',ha='center',fontsize=8)
  ax.legend(fontsize=8);ax.grid(alpha=.2);finish(fig,name)
 primary=delta[delta.method=='B1_alpha100'];g=primary.groupby('dataset').relative_delta_bpp_vs_alpha0.mean();fig,ax=plt.subplots(figsize=(7,4));ax.bar(g.index,g.values*100);ax.axhline(0,color='k',lw=.7);ax.set(ylabel='Mean relative total bpp change (%)',title='alpha1 vs alpha0, same external QP');ax.tick_params(axis='x',rotation=20);finish(fig,'rate_change_by_dataset')
 values=[('P strength','0, .25, .5, .75, 1, 1.25'),('Source change',f"alpha1 RGB L1={primary.RGB_residual_L1.mean():.4g}"),('Discrete coding',f"symbols changed={primary.fraction_quantized_symbols_changed.mean():.2%}"),('Actual RANS bits',f"relative bpp={primary.relative_delta_bpp_vs_alpha0.mean():+.2%}"),('Teacher proxy',f"delta cosine={primary.delta_cos_vs_alpha0.mean():+.4g}"),('Generation quality',f"delta LPIPS={primary.delta_LPIPS_vs_alpha0.mean():+.4g}")]
 fig,axes=plt.subplots(2,3,figsize=(13,5))
 for ax,(title,value) in zip(axes.flat,values):
  ax.axis('off');ax.text(.5,.6,title,ha='center',fontsize=13,weight='bold');ax.text(.5,.35,value,ha='center',fontsize=11,bbox=dict(boxstyle='round',fc='#e8f0fa',ec='#486080'))
 fig.suptitle('Source shaping → codec/quantization → real bits → proxy fidelity → quality\nPaired alpha1 vs alpha0 diagnostics; not causal channel attribution',fontsize=13);finish(fig,'mechanism_summary')
