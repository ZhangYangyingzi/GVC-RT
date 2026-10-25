"""Figures from audited measured points and fixed preview/map coordinates."""
from util import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
COLORS=('#777777','#0072B2','#E69F00','#009E73','#CC79A7')
def main():
 assert load(P/'delivery_audit.json')['status']=='PASS'
 out=P/'figures';out.mkdir(exist_ok=True)
 rows=list(csv.DictReader((P/'evaluation.csv').open()))
 for source in SOURCES:
  fig,axes=plt.subplots(2,2,figsize=(11,8))
  for ax,metric in zip(axes.flat,('PSNR','LPIPS','DISTS','FloLPIPS')):
   for mode,color in zip(MODES,COLORS):
    rs=sorted([r for r in rows if r['mode']==mode and r['sequence']==source],key=lambda r:float(r['actual_bpp']))
    ax.plot([float(r['actual_bpp']) for r in rs],[float(r[metric]) for r in rs],label=mode,color=color,ls='--' if mode=='CHANNEL' else '-',marker='x' if mode=='CHANNEL' else 'o',alpha=.9)
    if mode!='CHANNEL':
     for r in rs:ax.annotate('q'+r['external_qp'],(float(r['actual_bpp']),float(r[metric])),xytext=(3,3),textcoords='offset points',fontsize=7)
   ax.set(xlabel='Actual native RANS total bpp',ylabel=metric,title=source);ax.grid(alpha=.2)
  axes[0,0].legend(fontsize=8);fig.suptitle('Measured step300 points; FRAME and CHANNEL coincide');fig.tight_layout();fig.savefig(out/f'RD_{source}.png',dpi=130);plt.close(fig)
  canvas=Image.new('RGB',(256*6,286),'white');draw=ImageDraw.Draw(canvas)
  files=[P/'previews'/f'{source}_GT_frame15.png']+[P/'previews'/f'{key(m,source,2)}_frame15.png' for m in MODES]
  for i,(file,label) in enumerate(zip(files,('GT',*MODES))):
   with Image.open(file) as im:assert im.size==(256,256);canvas.paste(im.convert('RGB'),(i*256,30))
   draw.text((i*256+8,8),label,fill='black')
  canvas.save(out/f'preview_{source}.png')
 fig,axes=plt.subplots(5,3,figsize=(10,13))
 for row,mode in enumerate(MODES):
  with np.load(P/'server_artifacts/maps'/key(mode,'UVal0',2)/'frame_15.npz') as maps:
   for col,name in enumerate(('raw_multiplier','variant_multiplier','rho')):
    ax=axes[row,col];a=maps[name].astype('float32').mean(1)[0];im=ax.imshow(a,cmap='viridis',vmin=.5,vmax=2);ax.set_title(mode+' '+name);ax.axis('off');fig.colorbar(im,ax=ax,fraction=.04)
 fig.suptitle('Fixed UVal0 QP2 frame15, raw/variant/actual ratio; spatial axes are latent coordinates');fig.tight_layout();fig.savefig(out/'fixed_maps_UVal0.png',dpi=120);plt.close(fig)
 files=sorted(out.glob('*.png'))
 dump(P/'figure_audit.json',dict(status='PASS',evaluation_sha256=sha(P/'evaluation.csv'),fixed_preview_frame=15,fixed_preview_QP=2,fixed_crop=[200,400,256,256],files=[dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size) for f in files]))
 print('FIGURES_PASS',len(files),flush=True)
if __name__=='__main__':main()
