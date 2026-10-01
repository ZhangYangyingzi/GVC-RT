"""Plot every measured QP, without smoothing, re-evaluation or metric edits."""
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from PIL import Image

ROOT=Path(__file__).resolve().parent
DATASETS={'ulong':'U-Long final test', 'uvg':'UVG',
          'virat720':'VIRAT 720p (1280 × 720)', 'virat480':'VIRAT 480p (854 × 480)'}
METHODS={'original':('Original GVC-RT','#454545','o'),
         'v41':('Legacy V4.1','#D55E00','s'),
         'v62':('V6.2 (s=1.0, step 1000)','#0072B2','^')}
METRICS=('LPIPS','DISTS','FloLPIPS','FID','PSNR','SSIM','MS_SSIM')
HIGHER={'PSNR','SSIM','MS_SSIM'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':14,
        'axes.labelsize':12,'legend.fontsize':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.8,'savefig.facecolor':'white'})
    final=json.loads((ROOT/'final_integrity.json').read_text());assert final['status']=='PASS'
    preserved={ROOT/f:h for f,h in final['artifacts'].items()}
    for dataset in DATASETS:
        folder=ROOT/'results'/dataset;report=json.loads((folder/'report_integrity.json').read_text())
        preserved.update({folder/f:h for f,h in report['files'].items()})
    assert all(sha(p)==h for p,h in preserved.items()),'Existing result integrity mismatch'
    generated=[]
    for dataset,title in DATASETS.items():
        folder=ROOT/'results'/dataset;source=folder/'all_qp_summary.csv'
        with source.open(newline='') as f:rows=list(csv.DictReader(f))
        assert len(rows)==30 and {r['method'] for r in rows}==set(METHODS)
        fps=20 if dataset.startswith('virat') else 30
        assert all(float(r['rate_accounting_fps'])==fps for r in rows)
        series={}
        for method in METHODS:
            own=sorted([r for r in rows if r['method']==method],key=lambda r:int(r['QP']))
            assert [int(r['QP']) for r in own]==list(range(10))
            assert all(float(r['kbps'])>0 and all(math.isfinite(float(r[k])) for k in ('kbps',*METRICS)) for r in own)
            series[method]=own
        for metric in METRICS:
            fig,ax=plt.subplots(figsize=(8.4,6.2))
            for method,(label,color,marker) in METHODS.items():
                own=series[method]
                ax.plot([float(r['kbps']) for r in own],[float(r[metric]) for r in own],
                    label=label,color=color,marker=marker,linewidth=1.8,markersize=5,
                    markerfacecolor='white',markeredgewidth=1.2)
            display='MS-SSIM' if metric=='MS_SSIM' else metric
            direction='higher is better' if metric in HIGHER else 'lower is better'
            unit=' (dB)' if metric=='PSNR' else ''
            ax.set_xlabel(f'Bitrate (kbps, {fps} fps rate accounting)')
            ax.set_ylabel(f'{display}{unit} — {direction}')
            ax.set_title(f'{title} | Rate–{display}',pad=13)
            ax.set_xscale('linear');ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
            formatter=ScalarFormatter(useOffset=False);formatter.set_scientific(False);ax.xaxis.set_major_formatter(formatter)
            ax.tick_params(direction='out');ax.grid(True,alpha=.2,linewidth=.7);ax.set_axisbelow(True)
            ax.margins(x=.035,y=.08);ax.legend(loc='best',frameon=True,framealpha=.94,edgecolor='#dddddd')
            fig.text(.5,.025,'Measured QP 0–9 · Real bitstream bytes · No smoothing or extrapolation',ha='center',fontsize=9,color='#555555')
            fig.tight_layout(rect=(0,.055,1,1))
            out=folder/f'Rate_{metric}.png'
            fig.savefig(out,dpi=240,metadata={'Title':f'{title} Rate–{display}', 'Description':'Measured dataset aggregates; linear kbps axis; QP0–9; no fitted curve.'})
            plt.close(fig)
            with Image.open(out) as image:
                assert image.format=='PNG' and image.width>=1800 and image.height>=1200
                dimensions=list(image.size);image.verify()
            generated.append(dict(dataset=dataset,metric=metric,path=str(out),sha256=sha(out),dimensions=dimensions,
                source_csv=str(source),source_sha256=sha(source),methods=list(METHODS),points_per_method=10,
                x_axis='Bitrate (kbps)',x_scale='linear',rate_accounting_fps=fps,y_axis=display,quality_direction=direction))
            print('SAVED',out.relative_to(ROOT),flush=True)
    assert len(generated)==28
    assert all(sha(p)==h for p,h in preserved.items()),'Numerical result changed during plotting'
    audit=dict(status='PASS',plot_count=len(generated),plots=generated,all_existing_result_hashes_unchanged=True,
        no_retesting=True,no_smoothing=True,no_extrapolation=True,point_order='External QP0–9; never metric-value sorting')
    out=ROOT/'results'/'rd_plot_audit.json';temp=out.with_suffix('.tmp');temp.write_text(json.dumps(audit,indent=2)+'\n');temp.replace(out)

if __name__=='__main__':main()
