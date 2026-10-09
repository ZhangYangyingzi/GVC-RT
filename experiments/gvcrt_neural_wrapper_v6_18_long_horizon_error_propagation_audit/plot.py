"""Dataset-specific raw and original-relative measured frame curves."""
from io18 import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
STYLES={'original':('#222222','-'),'v62_initial':('#0072B2','--'),'mixed_1000':('#009E73','-'),'dists_fixed_1000':('#D55E00','-.'),'dists_w05_1000':('#CC79A7',':')}
def main():
    rows=read(ROOT/'results/frame_curve_dataset_means.csv');generated=[]
    for d in DATASETS:
        for q in QPS:
            for k in METRICS:
                for delta in (False,True):
                    fig,ax=plt.subplots(figsize=(10,6))
                    for m in METHODS:
                        rs=sorted([r for r in rows if (r['dataset'],int(r['external_QP']),r['metric'],r['method'])==(d,q,k,m)],key=lambda r:int(r['frame_index']));assert len(rs)==64
                        y=[float(r['delta_vs_original_mean' if delta else 'mean']) for r in rs]
                        assert all(np.isfinite(y[i]) for i in range(1,64))
                        color,style=STYLES[m];ax.plot(range(1,65),y,label=m,color=color,linestyle=style,linewidth=1.8)
                    if delta:ax.axhline(0,color='#777777',linewidth=.7,zorder=0)
                    ax.set(xlabel='Frame index',ylabel=k+' − original' if delta else k,xlim=(1,64),title=f'{d} | external QP {q} | {k}'+(' | delta vs original' if delta else ''))
                    ax.set_xticks([1,8,16,24,32,40,48,56,64]);ax.grid(alpha=.25);ax.legend(fontsize=9);fig.text(.5,.015,'Equal sequence weights | Diagnostic same-external-QP comparison',ha='center',fontsize=9);fig.tight_layout(rect=(0,.035,1,1))
                    target=ROOT/'plots'/f'{d}_qp{q}_{k.lower()}_{"delta_vs_original" if delta else "vs_frame"}.png';fig.savefig(target,dpi=180);plt.close(fig);generated.append(dict(path=str(target.relative_to(ROOT)),SHA256=sha(target)))
    assert len(generated)==72;dump(ROOT/'audits/plot_integrity.json',dict(status='PASS',expected_plots=72,completed_plots=72,plots=generated));print('PLOTS PASS 72',flush=True)
if __name__=='__main__':main()
