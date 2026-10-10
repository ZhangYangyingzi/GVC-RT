from io21 import *
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
for d in DATASETS:
    for m in METHODS:
        curves=[[r for r in read(ROOT/'results'/f'ip_{ip}'/d/'all_qp_summary.csv') if r['method']==m] for ip in IPS]
        plot(ROOT/'results/IP_comparison'/d/m,curves,[f'IP={ip}' for ip in IPS],f'{d} {m}')
print('PLOTS PASS 128',flush=True)

