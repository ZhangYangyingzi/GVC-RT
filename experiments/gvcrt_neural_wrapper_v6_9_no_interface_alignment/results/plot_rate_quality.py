"""Plot measured RD tables; output PNGs directly in this results directory."""
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATASETS = {'uvg': 'UVG', 'ulong': 'U-Long', 'virat480': 'VIRAT 480p', 'virat720': 'VIRAT 720p'}
STYLES = {
    'B1000': ('#333333', 'o', '-'),
    'G50_1250': ('#D55E00', 's', '--'),
    'G50_1500': ('#E69F00', 's', '-'),
    'G25_1250': ('#009E73', 'D', '--'),
    'G25_1500': ('#669900', 'D', '-'),
    'N50_1250': ('#0072B2', '^', '--'),
    'N50_1500': ('#56B4E9', '^', '-'),
    'N25_1250': ('#88419D', 'v', '--'),
    'N25_1500': ('#CC79A7', 'v', '-'),
}
METRICS = ('LPIPS', 'DISTS', 'FloLPIPS', 'PSNR', 'SSIM', 'MS_SSIM')
PRIMARY = ('LPIPS', 'DISTS', 'FloLPIPS', 'FID')
LOWER = {'LPIPS', 'DISTS', 'FloLPIPS', 'FID'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(name):
    with (ROOT/name).open(newline='') as stream:
        return list(csv.DictReader(stream))

def curves(rows, dataset, metric, qps, rate='kbps'):
    output = {}
    for method in STYLES:
        own = [row for row in rows if row['dataset'] == dataset and row['method'] == method]
        assert sorted(int(row['QP']) for row in own) == qps, (dataset, method, metric)
        points = sorted((float(row[rate]), float(row[metric])) for row in own)
        assert all(x > 0 and math.isfinite(x) and math.isfinite(y) for x, y in points)
        output[method] = points
    return output

def draw(ax, points, title, metric, compact=False):
    for method, values in points.items():
        color, marker, linestyle = STYLES[method]
        ax.plot([x for x, _ in values], [y for _, y in values], label=method,
                color=color, marker=marker, linestyle=linestyle,
                markersize=3.5 if compact else 4.7, linewidth=1.5 if compact else 1.8)
    display = 'MS-SSIM' if metric == 'MS_SSIM' else metric
    unit = ' (dB)' if metric == 'PSNR' else ''
    direction = 'lower is better' if metric in LOWER else 'higher is better'
    ax.set_title(f'{title} | {display}')
    ax.set_xlabel('Bitrate (kbps)')
    ax.set_ylabel(f'{display}{unit} ({direction})')
    ax.grid(alpha=0.24)
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)

def main():
    sources = ('dataset_macro_rd.csv', 'fid_raw.csv', 'vimeo_heldout_raw.csv')
    before = {name: sha(ROOT/name) for name in sources}
    macro = read(sources[0]); fid = read(sources[1]); heldout = read(sources[2])
    assert len(macro) == len(fid) == 360 and len(heldout) == 864
    assert all(row['status'] == 'PASS' for row in fid + heldout)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    created = []
    for dataset, title in DATASETS.items():
        for metric in (*METRICS, 'FID'):
            rows = fid if metric == 'FID' else macro
            rate = 'dataset_kbps' if metric == 'FID' else 'kbps'
            points = curves(rows, dataset, metric, list(range(10)), rate)
            fig, ax = plt.subplots(figsize=(11, 6))
            draw(ax, points, title, metric)
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)
            fig.subplots_adjust(left=0.09, right=0.76, bottom=0.16, top=0.91)
            note = ('Dataset FID and aggregate bitrate' if metric == 'FID' else 'Macro mean per video')
            fig.text(0.5, 0.03, note+' | QP 0–9 | Measured points connected without smoothing',
                     ha='center', fontsize=8.5, color='#555555')
            target = ROOT/f'{dataset}_Rate_{metric}.png'
            fig.savefig(target, dpi=200); plt.close(fig); created.append(target)

    grouped = defaultdict(list)
    for row in heldout:
        grouped[(row['method'], int(row['QP']))].append(row)
    expected = {(method, qp) for method in STYLES for qp in (0, 4, 9)}
    assert set(grouped) == expected and all(len(rows) == 32 for rows in grouped.values())
    heldout_macro = []
    for (method, qp), rows in grouped.items():
        heldout_macro.append(dict(dataset='vimeo_heldout', method=method, QP=qp,
            **{key: statistics.mean(float(row[key]) for row in rows) for key in ('kbps', *METRICS)}))
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(11, 6))
        draw(ax, curves(heldout_macro, 'vimeo_heldout', metric, [0, 4, 9]), 'Vimeo held-out', metric)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)
        fig.subplots_adjust(left=0.09, right=0.76, bottom=0.16, top=0.91)
        fig.text(0.5, 0.03, '32 held-out clips | QP 0, 4, 9 | Measured points connected without smoothing',
                 ha='center', fontsize=8.5, color='#555555')
        target = ROOT/f'vimeo_heldout_Rate_{metric}.png'
        fig.savefig(target, dpi=200); plt.close(fig); created.append(target)

    fig, axes = plt.subplots(4, 4, figsize=(23, 18))
    for index, (dataset, title) in enumerate(DATASETS.items()):
        for column, metric in enumerate(PRIMARY):
            rows = fid if metric == 'FID' else macro
            rate = 'dataset_kbps' if metric == 'FID' else 'kbps'
            draw(axes[index, column], curves(rows, dataset, metric, list(range(10)), rate),
                 title, metric, compact=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=12)
    fig.suptitle('V6.9 | Measured bitrate–quality curves', fontsize=19, y=0.99)
    fig.text(0.5, 0.005, 'QP 0–9 | Spatial/temporal metrics: macro video means; FID: dataset feature pool and aggregate bitrate',
             ha='center', fontsize=11, color='#555555')
    fig.tight_layout(rect=(0, 0.09, 1, 0.97), h_pad=2, w_pad=1.5)
    target = ROOT/'Rate_Quality_Overview.png'
    fig.savefig(target, dpi=180); plt.close(fig); created.append(target)
    assert before == {name: sha(ROOT/name) for name in sources}
    manifest = dict(status='PASS',source_sha256=before,PNG_count=len(created),
        x_axis='linear measured bitrate in kbps',smoothing=False,extrapolation=False,
        plots={path.name:sha(path) for path in created})
    (ROOT/'rate_quality_plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Generated {len(created)} PNGs directly in {ROOT}')

if __name__ == '__main__':main()
