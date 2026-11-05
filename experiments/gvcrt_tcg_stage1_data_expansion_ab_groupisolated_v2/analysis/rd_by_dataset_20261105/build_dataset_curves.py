"""Dataset-separated and individual-video R-D plots from frozen measurements."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parents[1] / 'metrics_v3.csv'
SOURCE_SHA = 'd413190251cdfe64a17ba05826c0769b09f2c6475012822d1d2f1d500ed55b86'
STATES = ['parent0', 'parent5000', 'A_common', 'A_best_dev', 'B_common']
LABELS = ['Original-weight baseline (parent0)', 'TCG parent5000', 'A +1500 (common)', 'A +500 (dev-best)', 'B +1500 (common / dev-best)']
METRICS = ['LPIPS', 'DISTS', 'FloLPIPS', 'PSNR', 'MS_SSIM', 'WarpingError_GTFlow_PWC']
GROUPS = [
    ('ulong_new_test', 'ULong: new test subset', 'new_holdout_test', 'ulong_new', 32),
    ('ulong_historical', 'ULong: historical subset', 'historical_reference', 'ulong', 6),
    ('hevc_b', 'HEVC-B: historical subset', 'historical_reference', 'hevc_b', 5),
    ('uvg_holdout', 'UVG: held-out historical subset', 'historical_reference', 'uvg_holdout', 2),
]
COLORS = ['#475569', '#2563eb', '#d97706', '#0d9488', '#c026d3']
STYLES = ['-', '--', ':', '-.', '-']
QPS = (0, 5, 9)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def draw(rows, title, output, count, per_video):
    values = {(r['model_state'], int(r['QP'])): r for r in rows}
    assert len(values) == len(STATES)*len(QPS)
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.8))
    for ax, metric in zip(axes.flat, METRICS):
        for state, label, color, style in zip(STATES, LABELS, COLORS, STYLES):
            points = sorted([values[state, qp] for qp in QPS], key=lambda r: float(r['bpp']))
            ax.plot([float(r['bpp']) for r in points], [float(r[metric]) for r in points],
                    color=color, linestyle=style, marker='o', markersize=4.4,
                    linewidth=2 if state == 'B_common' else 1.5, label=label)
        higher = metric in ('PSNR', 'MS_SSIM')
        display = {'MS_SSIM': 'MS-SSIM', 'WarpingError_GTFlow_PWC': 'GT-flow warp error'}.get(metric, metric)
        ax.set_title(f'{display} ({"higher" if higher else "lower"} is better)', loc='left', fontsize=11, fontweight='bold')
        ax.set_ylabel('dB' if metric == 'PSNR' else display)
        ax.set_xlabel('Measured rate (bpp)' if per_video else 'Mean measured rate within this subset (bpp)')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(True, color='#e2e8f0', linewidth=.7)
        ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=.985)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .91 if per_video else .947), ncol=3, frameon=False, fontsize=9.5)
    averaging = 'Single-video results: no averaging across videos or datasets.' if per_video else f'Equal-video means over these {count} videos only; no averaging across datasets or evaluation splits.'
    note = (averaging + '\n'
            'Three measured QPs (0, 5, 9); lines only connect measurements. Matched models have identical streams per video/QP.\n'
            'Baseline: original checkpoint weights at loader precision, with zero TCG residual; no separate full-dataset original-code run.\n'
            'GT-flow error also contains reconstruction/flow/occlusion error; it is not an isolated flicker measure.')
    fig.text(.045, .018, note, fontsize=8.8, color='#475569', va='bottom')
    fig.tight_layout(rect=[.015, .12, .995, .885], h_pad=2.0, w_pad=2.1)
    fig.savefig(output, dpi=150, facecolor='white')
    plt.close(fig)


def main():
    assert sha(SOURCE) == SOURCE_SHA, 'Frozen metrics changed'
    baseline = json.loads((OUT/'baseline_identity_audit.json').read_text())
    assert baseline['status'] == 'PASS'
    rows = list(csv.DictReader(SOURCE.open(newline='')))
    assert len(rows) == 675
    matched = defaultdict(list)
    for r in rows:
        assert all(math.isfinite(float(r[k])) for k in ['bpp', 'kbps'] + METRICS)
        matched[(r['split'], r['domain'], r['sequence'], r['QP'])].append(r)
    for key, points in matched.items():
        assert len(points) == 5 and {r['model_state'] for r in points} == set(STATES), key
        for field in ('bpp', 'kbps', 'real_bytes', 'bitstream_sha256'):
            assert len({r[field] for r in points}) == 1, (key, field)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5,
                         'axes.spines.top': False, 'axes.spines.right': False})
    aggregate, changes, manifests, indexes = [], [], [], []
    seen = set()
    for group, title, split, domain, expected in GROUPS:
        subset = [r for r in rows if r['split'] == split and r['domain'] == domain]
        names = sorted({r['sequence'] for r in subset})
        assert len(names) == expected and len(subset) == expected * 15
        seen.update((r['split'], r['domain'], r['sequence']) for r in subset)
        group_rows = []
        for state in STATES:
            for qp in QPS:
                points = [r for r in subset if r['model_state'] == state and int(r['QP']) == qp]
                assert len(points) == expected
                group_rows.append(dict(dataset_group=group, split=split, domain=domain, model_state=state, QP=qp,
                                       video_count=expected, **{m: float(np.mean([float(r[m]) for r in points])) for m in ['bpp', 'kbps'] + METRICS}))
        aggregate.extend(group_rows)
        draw(group_rows, f'{title} | {expected} videos | rate-quality curves', OUT/f'{group}_rate_quality.png', expected, False)
        vals = {(r['model_state'], r['QP']): r for r in group_rows}
        for target, reference in [('B_common', 'parent0'), ('B_common', 'parent5000')]:
            for qp in QPS:
                a, b = vals[target, qp], vals[reference, qp]
                change = dict(dataset_group=group, target=target, reference=reference, QP=qp, bpp=a['bpp'])
                for m in METRICS:
                    suffix = '_delta' if m in ('PSNR', 'MS_SSIM') else '_change_pct'
                    change[m + suffix] = a[m]-b[m] if suffix == '_delta' else 100*(a[m]/b[m]-1)
                changes.append(change)
        video_dir = OUT/'per_video'/group
        video_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            assert '/' not in name and '\\' not in name
            video_rows = [r for r in subset if r['sequence'] == name]
            path = video_dir/f'{name}.png'
            draw(video_rows, f'{title}\n{name} | individual-video curves', path, 1, True)
            indexes.append(dict(dataset_group=group, sequence=name, figure=str(path.relative_to(OUT))))
        manifests.append(dict(group=group, split=split, domain=domain, videos=expected, sequences=names,
                              source_rows=len(subset), summary_plot=f'{group}_rate_quality.png'))
    assert len(seen) == 45
    write_csv(OUT/'rd_points_by_dataset.csv', aggregate)
    write_csv(OUT/'rd_points_per_video.csv', rows)
    write_csv(OUT/'same_rate_changes_by_dataset.csv', changes)
    write_csv(OUT/'per_video_index.csv', indexes)
    md = ['# 各数据集及逐视频 R-D 曲线', '',
          '这里使用已完成实验的 675 条实测记录；没有新增训练或全数据集推理。不同数据集和不同测试划分分别作图。', '',
          '数据集图在各自子集内按视频等权平均，这是数据集曲线的汇总方式。下面另附 45 张逐视频曲线，完全不跨视频平均。', '',
          '每张图包含 LPIPS、DISTS、FloLPIPS、PSNR、MS-SSIM、GT-flow warping error；横轴为实测 bpp。只有 QP 0、5、9 三点，不估算 BD-rate。', '',
          '基线标为 Original-weight baseline (parent0)：与用户确认的原版权重在加载精度下逐张量一致，TCG 残差为零。它不是原版代码全测试集独立重跑的记录。见 [基线核查](BASELINE_FINDINGS.md)。', '',
          '| 数据集/划分 | 视频数 | 曲线 |', '|---|---:|---|']
    for group, title, _, _, expected in GROUPS:
        md.append(f'| {title} | {expected} | [查看]({group}_rate_quality.png) |')
    md += ['', '这些是已评测子集，UVG 只有 HoneyBee、Jockey 两段，不能标为完整 UVG 评测。历史 ULong 6 段与新 ULong 32 段保持分开。', '',
           '表格：[按数据集数据](rd_points_by_dataset.csv) · [逐视频原始点](rd_points_per_video.csv) · [同码率变化](same_rate_changes_by_dataset.csv)', '',
           '## 逐视频曲线', '', '| 数据集/划分 | 视频 |', '|---|---|']
    for row in indexes:
        md.append(f'| {row["dataset_group"]} | [{row["sequence"]}]({row["figure"]}) |')
    (OUT/'README.md').write_text('\n'.join(md)+'\n')
    audit = dict(status='PASS', input_metrics_sha256=SOURCE_SHA, rows=675, videos=45,
                 metadata_same_video_QP_stream_rate_consistency='PASS; prior full-stream audit retained separately',
                 cross_dataset_averaging=False, cross_split_averaging=False, subset_aggregation='equal-video mean per model/QP',
                 per_video_plots=45, per_video_cross_video_averaging=False, dataset_plots=4, groups=manifests,
                 full_dataset_original_code_evaluation=False, no_training_or_full_dataset_inference=True,
                 artifacts={str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix in ('.png', '.csv', '.py', '.md', '.json') and p.name not in ('audit.json', 'publication.json')})
    (OUT/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', dataset_plots=4, individual_video_plots=45, input_rows=675)))


if __name__ == '__main__':
    main()
