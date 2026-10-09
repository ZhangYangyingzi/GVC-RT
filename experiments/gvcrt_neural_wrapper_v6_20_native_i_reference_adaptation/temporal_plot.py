from io20 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statistics
for d in DATASETS:
    rows=read(ROOT/'evaluation/per_frame_metrics.csv');rows=[r for r in rows if r['dataset']==d];windows=read(ROOT/'results'/d/'window_metrics.csv')
    for q in (0,4,9):
        for metric in ('LPIPS','DISTS','FloLPIPS'):
            for mode in ('frame','window'):
                fig,ax=plt.subplots(figsize=(10,6))
                for m in METHODS:
                    if mode=='frame':
                        xs=list(range(2 if metric=='FloLPIPS' else 1,65));ys=[statistics.mean(float(r[metric]) for r in rows if r['method']==m and int(r['external_QP'])==q and int(r['frame_index'])==i) for i in xs];ax.plot(xs,ys,label=m)
                    else:
                        xs=['1','2-8','9-16','17-32','33-48','49-64','2-64'];xs=xs[1:] if metric=='FloLPIPS' else xs;ys=[statistics.mean(float(r['mean']) for r in windows if r['method']==m and int(r['external_QP'])==q and r['metric']==metric and r['window']==w) for w in xs];ax.plot(xs,ys,'o-',label=m)
                ax.set(xlabel='Frame index' if mode=='frame' else 'Frame window',ylabel=metric,title=f'{d} QP{q}');ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout();dest=ROOT/'plots'/mode/f'{d}_qp{q}_{metric}.png';dest.parent.mkdir(parents=True,exist_ok=True);fig.savefig(dest,dpi=150);plt.close(fig)
print('TEMPORAL PLOTS PASS 72',flush=True)
