"""Plot already computed values using the existing repository plotting environment."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
payload=json.loads((ROOT/"audits/plot_payload.json").read_text())
pooled=payload["pooled"];newpoints=payload["newpoints"];METHODS=payload["methods"];DATASETS=payload["datasets"]
if pooled:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for pool in sorted(set(r['pool'] for r in pooled if 'subset' not in r['pool'])):
        plt.figure(figsize=(7,5))
        for m in METHODS:
            rr=sorted([r for r in pooled if r['pool']==pool and r['method']==m],key=lambda r:r['bpp']);plt.plot([r['bpp'] for r in rr],[r['FID'] for r in rr],'-o',label=m)
        plt.xlabel('Actual bitrate (bpp)');plt.ylabel('FID (lower is better)');plt.title(pool+' — 3-QP diagnostic');plt.legend();plt.tight_layout();p=ROOT/'plots'/pool/'Rate_FID.png';p.parent.mkdir(parents=True,exist_ok=True);plt.savefig(p,dpi=150);plt.close()
    for d in DATASETS:
        for metric in ('LPIPS','DISTS','FloLPIPS'):
            rr=[r for r in newpoints if r['dataset']==d]
            if not rr:continue
            plt.figure(figsize=(7,5))
            for m in METHODS:
                xs=[];ys=[]
                for q in QPS:
                    r=[v for v in rr if v['method']==m and v['QP']==q];xs.append(float(np.mean([v['bpp'] for v in r])));ys.append(float(np.mean([v[metric] for v in r])))
                plt.plot(xs,ys,'-o',label=m)
            plt.xlabel('Actual bitrate (dataset mean bpp)');plt.ylabel(metric+' (lower is better)');plt.title(d+' new64 — 3-QP diagnostic');plt.legend();plt.tight_layout();p=ROOT/'plots'/d/'new64'/f'Rate_{metric}.png';p.parent.mkdir(parents=True,exist_ok=True);plt.savefig(p,dpi=150);plt.close()
