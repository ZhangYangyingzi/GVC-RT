import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
from diag_utils import *

def main():
    rows=read(PARENT/'ulong_all_qp_summary.csv');mins=[]
    for m in old.METHODS:
        r=min((r for r in rows if r['method']==m),key=lambda r:float(r['mean_real_kbps']))
        mins.append(dict(method=m,external_qp=int(r['external_qp']),mean_real_kbps=float(r['mean_real_kbps'])))
    write('rd_curves/ulong/minimum_bitrates.csv',mins)
    lo=min(r['mean_real_kbps'] for r in mins);hi=max(float(r['mean_real_kbps']) for r in rows)
    ticks=[lo]+[v for v in (100,200,500,1000,2000,5000,10000) if lo<=v<=hi and abs(np.log(v/lo))>.18]
    audits=[]
    for metric in old.METRICS:
        fig,ax=plt.subplots(figsize=(11,7))
        for j,m in enumerate(old.METHODS):
            own=sorted((r for r in rows if r['method']==m),key=lambda r:int(r['external_qp']))
            assert len(own)==10
            x=[float(r['mean_real_kbps']) for r in own];y=[float(r[metric]) for r in own]
            line,=ax.plot(x,y,'o-',label=LABELS[m],linewidth=1.8,markersize=5)
            minimum=next(r for r in mins if r['method']==m);i=x.index(minimum['mean_real_kbps'])
            ax.annotate(f'QP{minimum["external_qp"]}\n{x[i]:.1f} kbps',xy=(x[i],y[i]),
                xytext=(.03+j*.245,1.015),textcoords='axes fraction',ha='left',va='bottom',fontsize=9,
                color=line.get_color(),arrowprops=dict(arrowstyle='-',color=line.get_color(),lw=.8),
                bbox=dict(boxstyle='round,pad=.25',fc='white',ec=line.get_color(),alpha=.95))
        box='Minimum measured bitrate\n'+'\n'.join(f'{LABELS[r["method"]]}: {r["mean_real_kbps"]:.1f} kbps' for r in mins)
        ax.text(.98,.96,box,transform=ax.transAxes,va='top',ha='right',fontsize=10,
            bbox=dict(boxstyle='round,pad=.5',facecolor='white',edgecolor='.6',alpha=.95))
        ax.set_xscale('log');ax.set_xlim(lo*.9,hi*1.05)
        ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:.1f}' if abs(v-lo)<1e-6 else f'{v:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter());ax.grid(True,alpha=.25)
        ax.set_xlabel('Mean real bitrate (kbps, log scale)');ax.set_ylabel(metric+' (lower is better)')
        ax.legend(loc='lower left',fontsize=9);fig.suptitle('Fresh U-Long | all external QP 0–9',y=.98)
        fig.subplots_adjust(top=.83,left=.1,right=.97,bottom=.12)
        for ext in ('png','pdf'):fig.savefig(ROOT/'rd_curves/ulong'/f'Rate_{metric}.{ext}',dpi=180)
        plt.close(fig)
        audits.append(dict(metric=metric,status='PASS',minimum_values=mins,source_sha256=sha(PARENT/'ulong_all_qp_summary.csv'),
            major_ticks=ticks,xlim=[lo*.9,hi*1.05],scale='log',annotations='actual minimum point callouts plus minimum bitrate box'))
    dump('rd_plot_audit.json',dict(status='PASS',plots=audits,minimum_bitrate_values_loaded_from_csv=True))
    table=[]
    for m in ('v41_20000','clip8'):
        for q in range(10):
            a=next(r for r in rows if r['method']=='original' and int(r['external_qp'])==q)
            c=next(r for r in rows if r['method']==m and int(r['external_qp'])==q)
            table.append(dict(method=m,external_qp=q,candidate_kbps=float(c['mean_real_kbps']),original_kbps=float(a['mean_real_kbps']),
                rate_ratio=float(c['mean_real_kbps'])/float(a['mean_real_kbps']),seen_during_wrapper_training=q<4,
                **{metric+'_same_QP_delta':float(c[metric])-float(a[metric]) for metric in old.METRICS},use='diagnostic only; not an RD gate'))
    write('ulong_per_qp_diagnostic.csv',table)
    table=[]
    for i,v in enumerate(old.videos('uvg')):
        for q in range(10):
            a=old.load(old.point_path('uvg','original',i,q))
            for m in METHODS:
                c=old.load(old.point_path('uvg',m,i,q))
                table.append(dict(sequence=v['sequence_name'],video_index=i,method=m,external_qp=q,
                    **{k:float(c[k]) for k in ('kbps','LPIPS','DISTS','FloLPIPS')},
                    **{k+'_delta_vs_Original':float(c[k])-float(a[k]) for k in ('kbps','LPIPS','DISTS','FloLPIPS')},
                    rate_ratio=float(c['kbps'])/float(a['kbps'])))
    write('uvg_per_sequence_diagnostic.csv',table)
    print('PLOTS AND DIAGNOSTIC TABLES PASS',flush=True)

if __name__=='__main__':main()
