"""Fixed RD/effect/visualization figures with no result-based example selection."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
METHODS={'A00':'O_original','A01':'G0body_GFadagn_z0','A10':'GFbody_G0adagn_z0',
 'A11':'Y_originalBridge_fullG','G_fullLoss_Gonly_500':'G_fullLoss_Gonly_500','F_BG500':'F_BG500'}
COLORS={'A00':'#333333','A01':'#d55e00','A10':'#0072b2','A11':'#009e73',
 'G_fullLoss_Gonly_500':'#cc79a7','F_BG500':'#e69f00'}
PANELS={'A':('A00','A01','A10','A11'),'B':('A00','G_fullLoss_Gonly_500','F_BG500')}
def read(p):
 with (HERE/p).open(newline='') as f:return list(csv.DictReader(f))
def save(fig,name):fig.tight_layout();fig.savefig(FIG/name,dpi=140);plt.close(fig)
def main():
 points=read('results/all_rd_points.csv');pooled=read('results/pooled_fid.csv');summary=read('results/equal_rate_summary.csv')
 manifest=json.loads((HERE.parent/'gvcrt_neural_wrapper_v6_37_loss_bridge_generator_mechanism_500/eval_manifest.json').read_text())
 made=[]
 for panel,methods in PANELS.items():
  for ds,videos in manifest.items():
   for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
    fig,ax=plt.subplots(figsize=(9,5))
    if metric=='FID':
     for m in methods:
      rr=sorted((x for x in pooled if x['dataset']==ds and x['method']==m),key=lambda x:float(x['mean_actual_bpp']))
      if len(rr)==3:ax.plot([float(x['mean_actual_bpp']) for x in rr],[float(x['FID']) for x in rr],'-o',label=m,color=COLORS[m])
     ax.set_xlabel('Dataset mean actual complete RANS bpp')
    else:
     for j,v in enumerate(videos):
      for m in methods:
       rr=sorted((x for x in points if x['dataset']==ds and int(x['video'])==v['video_index'] and x['method']==m),key=lambda x:float(x['actual_bpp']))
       if len(rr)==3:ax.plot([float(x['actual_bpp']) for x in rr],[float(x[metric]) for x in rr],'-o',
        alpha=.4 if len(videos)>2 else .8,label=m if j==0 else None,color=COLORS[m],markersize=3)
     ax.set_xlabel('Actual complete RANS bpp; each line is one video')
    ax.set_xscale('log');ax.set_ylabel(metric+' ↓');ax.set_title(f'{panel}: {ds} {metric}; '+', '.join(methods))
    ax.grid(alpha=.2);ax.legend(fontsize=7)
    name=f'{panel}_{ds}_{metric.lower()}_RD.png';save(fig,name);made.append(name)
  for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
   rr=[x for x in summary if x['panel']==panel and x['metric']==metric]
   labels=[f"{x['dataset']}\n{x['contrast']}" for x in rr]
   fig,ax=plt.subplots(figsize=(max(8,len(rr)*.65),4.5));ax.bar(range(len(rr)),[float(x['delta']) for x in rr],color='#0072b2')
   ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(len(rr)),labels,rotation=55,ha='right',fontsize=7)
   ax.set_ylabel(metric+' delta; negative improves');ax.set_title(f'{panel} conditional effects on panel-common actual-bpp intervals')
   name=f'{panel}_conditional_{metric.lower()}.png';save(fig,name);made.append(name)
 old=HERE.parent/'gvcrt_neural_wrapper_v6_37_loss_bridge_generator_mechanism_500'
 for ds,videos in manifest.items():
  video=videos[0]['video_index']
  for qp in (0,5,9):
   key=f'{ds}_{video}_q{qp}'
   p=next((x for x in points if (x['dataset'],int(x['video']),int(x['QP']))==(ds,video,qp)),None)
   if p is None:continue
   for frame in (16,32,48,63):
    imgs=[];names=('GT',*METHODS)
    for name in names:
     if name=='F_BG500':path=old/'artifacts/examples'/key/'F_fullLoss_noP_500'/f'{frame}.png'
     elif name=='G_fullLoss_Gonly_500':path=HERE/'artifacts/examples'/key/name/f'{frame}.png'
     else:path=HERE/'artifacts/examples'/key/name/f'{frame}.png'
     if not path.exists():break
     imgs.append(Image.open(path).convert('RGB').crop((832,384,1088,640)))
    if len(imgs)!=len(names):continue
    canvas=Image.new('RGB',(256*len(names),290),'white');draw=ImageDraw.Draw(canvas)
    for i,(name,img) in enumerate(zip(names,imgs)):
     canvas.paste(img,(256*i,25));draw.text((256*i+2,4),name,fill='black')
    draw.text((4,268),f'{ds} v{video} q{qp} frame{frame}; shared actual bpp {float(p["actual_bpp"]):.8f}; fixed crop (832,384)-(1088,640)',fill='black')
    name=f'fixed_{key}_frame{frame}.png';canvas.save(FIG/name);made.append(name)
 (HERE/'FIGURE_INDEX.md').write_text('# v6.41 fixed figures\n\nMethods: '+', '.join(f'{k}={v}' for k,v in METHODS.items())+
  '. All use measured complete-stream bpp. Lower distance is better. Lines never connect different videos. Fixed examples use manifest-first video, QP 0/5/9, frames 16/32/48/63 and one preregistered crop.\n\n'+
  '\n'.join(f'- `{x}`' for x in made)+'\n')
 print('FIGURES41',len(made),flush=True)
if __name__=='__main__':main()
