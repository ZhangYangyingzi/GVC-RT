from io18 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    for d in DATASETS:
        folder=ROOT/'results'/d
        for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
            rows=read(folder/('fid_pooled.csv' if metric=='FID' else 'all_qp_summary.csv'));fig,ax=plt.subplots(figsize=(9,6))
            for m in METHODS:
                rs=sorted([r for r in rows if r['method']==m],key=lambda r:float(r['bpp']));assert len(rs)==10
                ax.plot([float(r['bpp']) for r in rs],[float(r[metric]) for r in rs],'o-',markersize=3,label=m)
            ax.set(xlabel='Bitrate (dataset mean actual bpp)',ylabel=metric+' (lower is better)',title=d);ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(folder/f'Rate_{metric}.png',dpi=180);plt.close(fig)
    print('PLOTS PASS 16',flush=True)
if __name__=='__main__':main()
