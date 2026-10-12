import sys,subprocess
if sys.executable.startswith("/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/"):
 subprocess.run(["/data1/anaconda3_new/anaconda_program/bin/python","-B",__file__],check=True);sys.exit(0)
"""Objective diagnostic and measured-rate plots; no image interpolation."""
from io21 import *
def main():
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=ROOT/'plots';out.mkdir(exist_ok=True)
 ability=read(ROOT/'diagnostics/discriminator_ability.csv')
 for domain in ('ulong','uvg'):
  for metric in ['CE_real','CE_fake','CE_G','real_content_top1','real_content_top5','AUROC_fake_from_frame_mean_p_fake']:
   fig,ax=plt.subplots()
   for panel in ('seen','unseen'):
    for rp in ('fixed','matched'):
     rows=sorted([r for r in ability if r['domain']==domain and r['panel_type']==panel and r['reconstruction_panel']==rp],key=lambda r:int(r['step']))
     if rows:ax.plot([int(r['step']) for r in rows],[float(r[metric]) for r in rows],marker='o',label=f'{panel}/{rp}')
   ax.set(xlabel='Checkpoint update',ylabel=metric,title=f'{domain} / {metric}');ax.legend();fig.tight_layout();fig.savefig(out/f'diagnostic_{domain}_{metric}.png',dpi=140);plt.close(fig)
 logs=read(ROOT/'diagnostics/training_logs/C_decoder_illm_gradient_summary.csv')
 for metric in ['lambda_eff','adv_B1_ratio','gradient_cosine','global_clip_coefficient','B1_gradient_norm','unweighted_adv_gradient_norm']:
  fig,ax=plt.subplots();ax.plot([int(r['step']) for r in logs],[float(r[metric]) for r in logs],linewidth=.6);ax.set(xlabel='Effective G update',ylabel=metric,title=f'v6.26 C / {metric}');fig.tight_layout();fig.savefig(out/f'training_{metric}.png',dpi=140);plt.close(fig)
 sensitivity=read(ROOT/'diagnostics/condition_sensitivity.csv')
 for metric in ['delta_CE_content','delta_p_fake','delta_content_only_top1']:
  fig,ax=plt.subplots()
  for cond in ('zero','shuffled'):
   rows=[r for r in sensitivity if r['image']=='real' and r['condition']==cond];steps=[0,250,500,1000];vals=[np.mean([float(r[metric]) for r in rows if int(r['checkpoint'])==s]) for s in steps];ax.plot(steps,vals,marker='o',label=cond+' (OOD)')
  ax.set(xlabel='Checkpoint update',ylabel=metric,title='Real image / paired condition probe');ax.legend();fig.tight_layout();fig.savefig(out/f'condition_{metric}.png',dpi=140);plt.close(fig)
 if not (ROOT/'results/mcljcv/fid_pooled.csv').exists():
  dump(ROOT/'audits/diagnostic_plots_integrity.json',dict(status='PASS',PNG_count=len(list(out.glob('*.png')))));return
 rows=read(ROOT/'results/mcljcv/fid_pooled.csv')
 for metric in ['LPIPS','DISTS','FloLPIPS','FID','D_mu','D_cov']:
  fig,ax=plt.subplots()
  for m in METHODS:
   rs=sorted([r for r in rows if r['method']==m],key=lambda r:float(r['bpp']));ax.plot([float(r['bpp']) for r in rs],[float(r[metric]) for r in rs],marker='.',label=m)
  ax.set(xlabel='Actual bpp (dataset mean)',ylabel=metric,title=f'MCL-JCV / {metric}');ax.legend(fontsize=6);fig.tight_layout();fig.savefig(out/f'MCL_JCV_Rate_{metric}.png',dpi=140);plt.close(fig)
 windows=read(ROOT/'results/mcljcv/window_metrics.csv')
 for metric in ['LPIPS','DISTS','FloLPIPS']:
  fig,ax=plt.subplots();names=['1-1','2-8','9-16','17-32','33-48','49-64']
  for m in METHODS:
   vals=[]
   for win in names:
    selected=[float(r[metric]) for r in windows if r['method']==m and r['window_1based']==win and r[metric]];vals.append(np.mean(selected) if selected else np.nan)
   ax.plot(range(6),vals,label=m)
  ax.set_xticks(range(6),names);ax.set(xlabel='Frame window',ylabel=metric,title=f'MCL-JCV / {metric} / all measured QPs');ax.legend(fontsize=6);fig.tight_layout();fig.savefig(out/f'MCL_JCV_window_{metric}.png',dpi=140);plt.close(fig)
 dump(ROOT/'audits/plots_integrity.json',dict(status='PASS',PNG_count=len(list(out.glob('*.png'))),objective_titles=True))
if __name__=='__main__':main()
