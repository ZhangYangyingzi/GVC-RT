"""Publication artifacts from recorded values only; CPU, no model changes."""
from util import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
METHODS=['N0','N1','Q0','Q1'];SEQUENCES=['UVal0','UVal1','ParkScene','Jockey'];METRICS=['LPIPS','DISTS','FloLPIPS','PSNR']
COLORS=dict(N0='#0072B2',N1='#56B4E9',Q0='#D55E00',Q1='#CC79A7')
def rows(n):
 with (P/n).open(newline='') as f:return list(csv.DictReader(f))
def smooth(x,n=15):return np.convolve(np.asarray(x,float),np.ones(n)/n,mode='valid')
def save(fig,name):
 fig.tight_layout();fig.savefig(P/'figures'/name,dpi=150);plt.close(fig)
def main():
 (P/'figures').mkdir(exist_ok=True)
 train=rows('training_log.csv');ev=rows('evaluation.csv')
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 for ax,key in zip(axes.flat,['loss','LPIPS','L1','estimated_bpp']):
  for b in METHODS:
   rs=[r for r in train if r['branch']==b];ys=smooth([float(r[key]) for r in rs]);ax.plot(np.arange(15,len(rs)+1),ys,color=COLORS[b],ls='--' if b in ('N0','Q0') else '-',label=b+(' reused' if b in ('N0','Q0') else ' new'))
  ax.set(xlabel='Committed optimizer update',ylabel=key,title='15-update moving mean');ax.grid(alpha=.2)
 axes[0,0].legend();save(fig,'training.png')
 fig,axes=plt.subplots(3,3,figsize=(15,11))
 keys=['H_multiplier','actual_rho','q_base','q_new','raw_quant_step']
 for ax,key in zip(axes.flat,keys):
  for b in ['N1','Q1']:
   rs=[r for r in train if r['branch']==b];x=np.arange(1,301);ax.plot(x,[float(r[key+'_mean']) for r in rs],color=COLORS[b],label=b+' mean')
   ax.fill_between(x,[float(r[key+'_p05']) for r in rs],[float(r[key+'_p95']) for r in rs],color=COLORS[b],alpha=.15,label=b+' p05–p95')
  ax.set(title=key,xlabel='Update');ax.grid(alpha=.2)
 for ax,key in zip(list(axes.flat)[5:],['q_base_lower_clamp_fraction','q_lower_clamp_fraction','pass0_skip_fraction','pass1_skip_fraction']):
  for b in ['N1','Q1']:
   rs=[r for r in train if r['branch']==b];ax.plot(np.arange(1,301),[float(r[key]) for r in rs],color=COLORS[b],label=b)
  ax.set(title=key,xlabel='Update',ylabel='Fraction');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8);fig.suptitle('Supervised-frame statistics; p05/p95 averaged over four frames per update',fontsize=12);save(fig,'multiplier_statistics.png')
 for seq in SEQUENCES:
  fig,axes=plt.subplots(2,2,figsize=(11,8))
  for ax,metric in zip(axes.flat,METRICS):
   for b in METHODS:
    rs=sorted([r for r in ev if r['branch']==b and int(r['step'])==300 and r['sequence']==seq],key=lambda r:float(r['actual_bpp']))
    x=[float(r['actual_bpp']) for r in rs];y=[float(r[metric]) for r in rs]
    ax.plot(x,y,marker='o',ls='--' if b in ('N0','Q0') else '-',color=COLORS[b],label=b+(' reused' if b in ('N0','Q0') else ' new'))
    for xx,yy,r in zip(x,y,rs):ax.annotate('q'+r['external_qp'],(xx,yy),xytext=(2,3),textcoords='offset points',fontsize=6)
   ax.set(xlabel='Actual RANS total bpp',ylabel=metric,title=seq+' step300');ax.grid(alpha=.2)
  axes[0,0].legend(fontsize=8);save(fig,'actual_RD_'+seq+'.png')
 # Three checkpoints, common validation QPs only, no interpolation or best selection.
 fig,axes=plt.subplots(2,3,figsize=(13,7))
 for col,key in enumerate(['actual_bpp','LPIPS','PSNR']):
  for row,seq in enumerate(SEQUENCES[:2]):
   ax=axes[row,col]
   for b in METHODS:
    for q in ('2','7'):
     rs=sorted([r for r in ev if r['branch']==b and r['sequence']==seq and r['external_qp']==q],key=lambda r:int(r['step']))
     ax.plot([int(r['step']) for r in rs],[float(r[key]) for r in rs],color=COLORS[b],ls='--' if q=='7' else '-',marker='o',label=b+' q'+q+(' reused' if b in ('N0','Q0') else ' new'))
   ax.set(title=seq,ylabel=key,xlabel='Update');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=6,ncol=2);save(fig,'validation_checkpoints.png')
 # Fixed point/crop chosen in protocol before inspecting output.
 for seq in SEQUENCES:
  files=[P/'previews'/f'{seq}_gt_frame15.png']+[P/'previews'/f'{b}_s300_{seq}_q2_frame15.png' for b in METHODS]
  assert all(f.is_file() for f in files),files
  canvas=Image.new('RGB',(256*5,286),'white');draw=ImageDraw.Draw(canvas)
  labels=['GT','N0 (reused)','N1 (new)','Q0 (reused)','Q1 (new)']
  for i,(file,label) in enumerate(zip(files,labels)):
   img=Image.open(file).convert('RGB');assert img.size==(256,256)
   canvas.paste(img,(i*256,30));draw.text((i*256+8,8),label,fill='black')
  canvas.save(P/'figures'/f'fixed_preview_{seq}.png')
 dump(P/'figure_index.json',dict(status='PASS',source_tables={n:sha(P/n) for n in ['training_log.csv','evaluation.csv']},preview_protocol=dict(step=300,QP=2,frame=15,crop_y=200,crop_x=400,size=256),files=[dict(path=str(f.relative_to(P)),sha256=sha(f),bytes=f.stat().st_size) for f in sorted((P/'figures').glob('*.png'))]))
 print('PLOTS_PASS',flush=True)
if __name__=='__main__':main()
