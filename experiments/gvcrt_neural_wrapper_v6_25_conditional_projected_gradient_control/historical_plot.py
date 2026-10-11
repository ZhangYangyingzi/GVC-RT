from io21 import *
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=read(ROOT/'historical/fid_decomposition_raw.csv')
for d in DATASETS:
 for metric in ('FID','D_mu','D_cov','LPIPS','DISTS','FloLPIPS'):
  fig,ax=plt.subplots(figsize=(10,6))
  for m in sorted({r['method'] for r in rows}):
   rs=sorted([r for r in rows if r['dataset']==d and r['method']==m],key=lambda r:float(r['bpp']));ax.plot([float(r['bpp']) for r in rs],[float(r[metric]) for r in rs],'o-',ms=3,label=m)
  ax.set(xlabel='Actual bpp',ylabel=metric,title=d+' '+metric);ax.grid(alpha=.2);ax.legend(fontsize=7);fig.tight_layout();p=ROOT/'historical/plots'/f'{d}_{metric}.png';p.parent.mkdir(exist_ok=True);fig.savefig(p,dpi=150);plt.close(fig)
