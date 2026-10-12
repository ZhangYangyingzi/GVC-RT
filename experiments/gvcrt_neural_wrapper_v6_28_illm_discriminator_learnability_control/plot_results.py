from io21 import *
import numpy as np
BR=("A_raw_current","B_rms_current","C_zero_current","D_raw_official_optimizer")
curve=read(ROOT/"results/learning_curves.csv")
for r in curve:
 for k,v in list(r.items()):
  if k not in ("branch","stage","panel"):r[k]=float(v)
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
for stage in ['TINY','MAIN']:
 for panel in ['tiny_fit','seen','unseen']:
  for metric in ['D_loss','real_CE_content','fake_CE_fake','CE_G_readonly','AUROC','real_content_top1','real_content_top5','paired_p_fake_difference_mean','paired_correct_order_fraction']:
   plt.figure(figsize=(7,4))
   for b in BR:
    rs=[r for r in curve if r['branch']==b and r['stage']==stage and r['panel']==panel]
    if rs:plt.plot([r['checkpoint'] for r in rs],[r[metric] for r in rs],label=b)
   if not any(r['stage']==stage and r['panel']==panel for r in curve):plt.close();continue
   plt.xlabel('Effective D updates');plt.ylabel(metric);plt.title(stage+' / '+panel+' / '+metric);plt.legend(fontsize=7);plt.tight_layout();plt.savefig(ROOT/'plots'/f'{stage}_{panel}_{metric}.png',dpi=140);plt.close()
for stage in ['TINY','MAIN']:
 plt.figure(figsize=(7,4))
 for b in BR:
  rs=read(ROOT/'branches'/b/stage/'gradient_summary.csv');values=np.array([float(r['D_loss']) for r in rs]);n=25;x=np.arange(n,len(values)+1,n);plt.plot(x,[values[i-n:i].mean() for i in x],label=b)
 plt.xlabel('Effective D updates');plt.ylabel('D_loss (25-step average)');plt.title(stage+' training');plt.legend(fontsize=7);plt.tight_layout();plt.savefig(ROOT/'plots'/f'{stage}_training_loss.png',dpi=140);plt.close()

scales=read(ROOT/'results/input_activation_scales.csv')
for feature in ('raw_condition','transformed_condition','condition_adapter','early_unet'):
 for metric in ('RMS','std'):
  plt.figure(figsize=(7,4))
  for branch in BR:
   rs=[r for r in scales if r['branch']==branch and r['stage']=='MAIN' and r['panel']=='unseen' and r['feature']==feature]
   steps=sorted(set(int(r['checkpoint']) for r in rs));ys=[]
   for step in steps:
    ss=[r for r in rs if int(r['checkpoint'])==step];ys.append(np.average([float(r[metric]) for r in ss],weights=[int(r['frames']) for r in ss]))
   if steps:plt.plot(steps,ys,label=branch)
  plt.xlabel('Effective D updates');plt.ylabel(metric);plt.title('MAIN / unseen / '+feature+' / '+metric);plt.legend(fontsize=7);plt.tight_layout();plt.savefig(ROOT/'plots'/('MAIN_unseen_'+feature+'_'+metric+'.png'),dpi=140);plt.close()
