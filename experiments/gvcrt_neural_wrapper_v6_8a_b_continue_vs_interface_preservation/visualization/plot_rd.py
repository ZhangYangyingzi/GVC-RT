"""Plot measured bitrate/quality curves; never change frozen experiment inputs."""
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
DATASETS={'uvg':'UVG','ulong':'U-Long','virat720':'VIRAT 720p','virat480':'VIRAT 480p','vimeo_heldout':'Vimeo held-out'}
METHODS={
    'B1000':('B1000','#333333','o','-'),
    'C0_1250':('C0 continue | step 1250','#56A4D8','s','--'),
    'C0_1500':('C0 continue | step 1500','#0072B2','s','-'),
    'C1_1250':('C1 preserve | step 1250','#E7A15D','^','--'),
    'C1_1500':('C1 preserve | step 1500','#D55E00','^','-'),
}
METRICS=('LPIPS','DISTS','FloLPIPS','FID','PSNR','SSIM','MS_SSIM')
LOWER={'LPIPS','DISTS','FloLPIPS','FID'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def num(row,key):
    v=float(row[key]);assert math.isfinite(v),(key,row)
    return v

def curves(rows,metric,rate_key,qps):
    assert set(r['method'] for r in rows)==set(METHODS)
    out={}
    for method in METHODS:
        own=[r for r in rows if r['method']==method]
        assert sorted(int(r['QP']) for r in own)==qps,(method,metric)
        values=[dict(QP=int(r['QP']),kbps=num(r,rate_key),quality=num(r,metric)) for r in own]
        assert all(v['kbps']>0 for v in values)
        out[method]=sorted(values,key=lambda v:v['kbps'])
    return out

def draw(dataset,metric,points,note):
    fig,ax=plt.subplots(figsize=(9,6))
    for method,values in points.items():
        label,color,marker,style=METHODS[method]
        ax.plot([v['kbps'] for v in values],[v['quality'] for v in values],
                label=label,color=color,marker=marker,linestyle=style,linewidth=1.8,markersize=5)
    metric_name='MS-SSIM' if metric=='MS_SSIM' else metric
    direction='lower is better' if metric in LOWER else 'higher is better'
    ax.set_xlabel('Bitrate (kbps)')
    ax.set_ylabel(f'{metric_name}'+(' (dB)' if metric=='PSNR' else '')+f' ({direction})')
    ax.set_title(f'{DATASETS[dataset]} | Rate–{metric_name}')
    ax.grid(alpha=.25);ax.legend(fontsize=9)
    ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    fig.text(.5,.015,note,ha='center',fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.045,1,1))
    dest=RESULTS/dataset/f'Rate_{metric}.png';dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix('.tmp.png');fig.savefig(tmp,dpi=200,format='png');plt.close(fig);tmp.replace(dest)
    with Image.open(dest) as im:
        assert im.format=='PNG' and im.size==(1800,1200);im.verify()
    return dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest),dataset=dataset,metric=metric,
                x_axis='Bitrate (kbps)',y_axis=metric,direction=direction,
                size=[1800,1200],points=points,aggregation_note=note)

def main():
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    protected=[*RESULTS.rglob('*.csv'),ROOT/'final_integrity.json',ROOT/'audits/local_protocol.json',
               ROOT/'recovery/output_validation.json']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    status=json.loads((ROOT/'final_integrity.json').read_text())['status']
    assert status in ('PASS','PASS_WITH_APPROVED_EXCEPTIONS')
    macro=read(RESULTS/'dataset_macro_rd.csv');fid=read(RESULTS/'fid_raw.csv');plots=[]
    assert len(macro)==len(fid)==200
    for dataset in list(DATASETS)[:4]:
        for metric in METRICS:
            source=fid if metric=='FID' else macro
            rows=[r for r in source if r['dataset']==dataset]
            points=curves(rows,metric,'dataset_kbps' if metric=='FID' else 'kbps',list(range(10)))
            note=('Pooled dataset FID; aggregate bitrate' if metric=='FID' else 'Dataset macro averages')+' | QP 0–9 | Measured points; no smoothing'
            plots.append(draw(dataset,metric,points,note))
    raw=read(RESULTS/'vimeo_heldout_raw.csv');held=read(RESULTS/'vimeo_heldout_progression.csv')
    assert len(raw)==480 and len(held)==15
    rows=[]
    for r in held:
        own=[v for v in raw if v['method']==r['method'] and int(v['QP'])==int(r['QP'])]
        assert len(own)==32 and all(int(v['frames'])==7 for v in own)
        bits=sum(num(v,'real_bytes')*8 for v in own);frames=sum(int(v['frames']) for v in own)
        # Vimeo has no asserted source FPS; reuse the established nominal 30fps accounting.
        kbps=bits/(frames/30)/1000
        assert math.isclose(kbps,statistics.mean(num(v,'kbps') for v in own),rel_tol=1e-10)
        assert math.isclose(bits/(frames*448*256),num(r,'bpp'),rel_tol=1e-10)
        row=dict(method=r['method'],QP=r['QP'],kbps=kbps,FID=num(r,'FID'))
        row.update({m:statistics.mean(num(v,m) for v in own) for m in METRICS if m!='FID'})
        for m in ('LPIPS','DISTS'):assert math.isclose(row[m],num(r,m),rel_tol=1e-10,abs_tol=1e-12)
        rows.append(row)
    for metric in METRICS:
        points=curves(rows,metric,'kbps',[0,4,9])
        note='32 official test clips | QP 0, 4, 9 | Nominal 30fps bitrate accounting | '+('Pooled FID' if metric=='FID' else 'Clip means')
        plots.append(draw('vimeo_heldout',metric,points,note))
    assert len(plots)==35
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected},'Protected CSV/audit was changed'
    audit=dict(status='PASS',PNG_count=35,datasets=list(DATASETS),metrics=list(METRICS),methods=list(METHODS),
               source_and_audit_files_unchanged=True,source_hashes=before,script_sha256=sha(Path(__file__)),
               training_or_evaluation_rerun=False,metric_values_unsmoothed=True,FID_uses_pooled_dataset_values=True,plots=plots)
    dest=RESULTS/'rd_plot_audit.json';tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n');tmp.replace(dest)
    print(json.dumps(dict(status='PASS',PNG_count=len(plots),directories=[str(RESULTS/d) for d in DATASETS]),indent=2))

if __name__=='__main__':main()
