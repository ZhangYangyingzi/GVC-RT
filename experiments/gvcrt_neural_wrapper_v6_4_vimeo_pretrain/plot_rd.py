"""Plot measured macro RD points; no smoothing or extrapolation."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
METHODS = {
    'original': ('Original GVC-RT', '#333333', 'o'),
    'v62': ('V6.2 schedule_s1p0', '#0072B2', 's'),
    'vimeo_only': ('V6.4 Vimeo pretrain', '#009E73', 'D'),
}
DATASETS = {'ulong': 'Final U-Long', 'uvg': 'UVG', 'virat720': 'VIRAT 720p', 'virat480': 'VIRAT 480p'}
METRICS = ('LPIPS', 'DISTS', 'FloLPIPS', 'PSNR', 'SSIM', 'MS_SSIM')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert json.loads((ROOT / 'final_integrity.json').read_text())['status'] == 'PASS'
    source = ROOT / 'results/dataset_macro_rd.csv'
    before = sha(source)
    with source.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 120
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    files = {}
    for dataset, title in DATASETS.items():
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(9, 6))
            for method, (label, color, marker) in METHODS.items():
                own = [r for r in rows if r['dataset'] == dataset and r['method'] == method]
                assert len(own) == 10 and sorted(int(r['QP']) for r in own) == list(range(10))
                own.sort(key=lambda r: float(r['kbps']))
                x = np.array([float(r['kbps']) for r in own])
                y = np.array([float(r[metric]) for r in own])
                assert np.isfinite(x).all() and np.isfinite(y).all() and (x > 0).all()
                ax.plot(x, y, label=label, color=color, marker=marker, markersize=4.5, linewidth=1.7)
            lower = metric in ('LPIPS', 'DISTS', 'FloLPIPS')
            name = 'MS-SSIM' if metric == 'MS_SSIM' else metric
            units = ' (dB)' if metric == 'PSNR' else ''
            ax.set_xlabel('Bitrate (kbps)')
            ax.set_ylabel(f'{name}{units} ({"lower" if lower else "higher"} is better)')
            ax.set_title(f'{title} | Rate–{name}')
            ax.grid(alpha=.25)
            ax.legend(fontsize=9)
            ax.ticklabel_format(axis='x', style='plain', useOffset=False)
            fig.text(.5, .015, 'Dataset macro averages | QP 0–9 | Measured points connected without smoothing', ha='center', fontsize=8, color='#555555')
            fig.tight_layout(rect=(0, .04, 1, 1))
            path = ROOT / 'results' / dataset / f'Rate_{metric}.png'
            fig.savefig(path, dpi=200)
            plt.close(fig)
            files[str(path.relative_to(ROOT))] = sha(path)
    assert sha(source) == before and len(files) == 24
    audit = dict(status='PASS',source=str(source),source_sha256=before,plots=files,count=len(files),
                 x_axis='linear bitrate in kbps',y_axis='arithmetic sequence mean of each metric',
                 points_per_method=10,smoothing=False,extrapolation=False,metric_value_sorting=False)
    (ROOT / 'results/rd_plot_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('PASS: 24 RD PNGs generated')

if __name__ == '__main__':
    main()
