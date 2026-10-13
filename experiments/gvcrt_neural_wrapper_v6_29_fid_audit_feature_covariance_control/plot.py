from io21 import *
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot(folder,curves,labels,title):
    folder.mkdir(parents=True,exist_ok=True)
    for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
        fig,ax=plt.subplots(figsize=(9,6))
        for label,rows in zip(labels,curves):
            rows=sorted(rows,key=lambda r:float(r['bpp']))
            assert len(rows)==10
            ax.plot([float(r['bpp']) for r in rows],[float(r[metric]) for r in rows],'o-',markersize=3,label=label)
        ax.set(xlabel='Bitrate (dataset mean actual bpp)',ylabel=metric+' (lower is better)',title=title);ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(folder/f'Rate_{metric}.png',dpi=180);plt.close(fig)
for ip in IPS:
    for d in DATASETS:
        folder=ROOT/'results'/f'ip_{ip}'/d;rows=read(folder/'all_qp_summary.csv')
        plot(folder,[[r for r in rows if r['method']==m] for m in METHODS],METHODS,f'{d} IP={ip}')
print('PLOTS PASS',len(IPS)*len(DATASETS)*4,flush=True)
