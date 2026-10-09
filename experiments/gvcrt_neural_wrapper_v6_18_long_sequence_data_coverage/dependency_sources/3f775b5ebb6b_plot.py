import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from parallel_utils import *
def main():
    rows=read(A/'virat_480p20_all_qp_summary.csv');out=A/'rd_curves';out.mkdir(exist_ok=True)
    for metric in METRICS:
        fig,ax=plt.subplots(figsize=(8,5));lines=[]
        for j,m in enumerate(('original','clip8')):
            own=sorted([r for r in rows if r['method']==m],key=lambda r:int(r['external_qp']))
            x=[float(r['kbps']) for r in own];y=[float(r[metric]) for r in own]
            ax.plot(x,y,'o-',label=m);i=min(range(len(x)),key=lambda i:x[i])
            ax.annotate(f"QP{own[i]['external_qp']}\n{x[i]:.1f} kbps",(x[i],y[i]),xytext=(15,15-j*40),textcoords='offset points',arrowprops=dict(arrowstyle='-'))
            lines.append(f'{m}: {x[i]:.3f} kbps')
        ax.set_xscale('log');ax.set_xlabel('Real kbps @20fps');ax.set_ylabel(metric);ax.set_title('VIRAT 854x480');ax.grid(alpha=.25);ax.legend()
        fig.text(.12,.02,'Minimum measured bitrate | '+' | '.join(lines),fontsize=9);fig.tight_layout(rect=(0,.07,1,1))
        for ext in ('png','pdf'):fig.savefig(out/f'Rate_{metric}.{ext}',dpi=180)
        plt.close(fig)
if __name__=='__main__':main()
