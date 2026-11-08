from pathlib import Path
import csv
import hashlib
import json
import statistics
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/Huang_group/zyyz/Projects/GVC-RT')
EXP = ROOT / 'GVC-RT_worktrees/tcg_stage2_convlora_coverage_ab_20261107/experiments/gvcrt_tcg_stage2_convlora_coverage_ab_20261107'
ORIGINAL = ROOT / 'GVC-RT_worktrees/tcg_online_v3_publication_1793765286/experiments/gvcrt_original_independent_rd_20261105'
OUT = Path(__file__).resolve().parent
METRICS = ['LPIPS', 'DISTS', 'FloLPIPS', 'PSNR', 'MS_SSIM', 'WarpingError_GTFlow_PWC', 'WarpingError_ReconFlow_PWC']
MODELS = ['GVC_RT_original', 'baseline', 'A_control', 'B_expanded']
LABELS = {'GVC_RT_original': 'Original GVC-RT (independent evaluation)', 'baseline': 'TCG B+1500 matched-FP32, no LoRA', 'A_control': 'TCG + LoRA A (stage3 + head), step 2000', 'B_expanded': 'TCG + LoRA B (A + upsample.conv1), step 2000'}
COLORS = {'GVC_RT_original': '#222222', 'baseline': '#0072B2', 'A_control': '#D55E00', 'B_expanded': '#009E73'}
MARKERS = {'GVC_RT_original': 's', 'baseline': 'o', 'A_control': '^', 'B_expanded': 'D'}
TITLES = {'ULong_new_holdout_32': 'ULong held-out subset (32 videos)', 'ULong_historical_6': 'ULong historical subset (6 videos)', 'HEVC_B_5': 'HEVC-B (5 videos)', 'UVG_historical_HoneyBee_Jockey_2': 'UVG subset (HoneyBee and Jockey only)'}
DATASETS = list(TITLES)

def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def dump_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def key(row):
    return row['domain'], row['sequence'], row['split'], int(row['QP'])

def load_points(path):
    items = [json.loads(p.read_text()) for p in path.glob('*.json')]
    result = {key(p['result']): p for p in items}
    assert len(result) == len(items) == 135
    return result

def main():
    original = read_csv(ORIGINAL / 'metrics_original.csv')
    current = read_csv(EXP / 'metrics.csv')
    oi = json.loads((ORIGINAL / 'final_integrity.json').read_text())
    protocol = json.loads((ORIGINAL / 'protocol.json').read_text())
    assert oi['status'] == 'PASS'
    assert sha(ORIGINAL / 'metrics_original.csv') == oi['metrics_original_sha256']
    assert sha(ORIGINAL / 'protocol.json') == oi['protocol_sha256']
    checkpoint_checks = {name: sha(ROOT / 'checkpoints' / name) == h for name, h in protocol['checkpoint_sha256'].items()}
    assert all(checkpoint_checks.values())
    om = {key(r): r for r in original}
    assert len(om) == len(original) == 135
    dataset_map = {key(r): r['dataset'] for r in current if r['model'] == 'baseline'}
    original_points = load_points(ORIGINAL / 'points')
    expected_metric_hashes = json.loads((ORIGINAL / 'metric_runtime_rank0.json').read_text())['hashes']
    for rank in range(4):
        assert json.loads((ORIGINAL / f'metric_runtime_rank{rank}.json').read_text())['hashes'] == expected_metric_hashes
    for p in original_points.values():
        assert p['status'] == 'PASS' and p['fresh_encode'] and p['independent_decode'] and p['real_RANS']
        assert p['original_codec_modules_only'] and not p['TCG_modules_present'] and not p['adapted_checkpoint_loaded']
        assert p['model_hashes_unchanged']
    point_paths = {'baseline': EXP / 'common_baseline/evaluation/final_baseline/points', 'A_control': EXP / 'A_control/evaluation/final_lora_2000/points', 'B_expanded': EXP / 'B_expanded/evaluation/final_lora_2000/points'}
    checks = []
    for model, path in point_paths.items():
        pts = load_points(path)
        rr = [r for r in current if r['model'] == model]
        assert len(rr) == 135 and {key(r) for r in rr} == set(om) == set(pts)
        for r in rr:
            k = key(r)
            op, cp = original_points[k], pts[k]
            record = {name: r[name] for name in ['domain', 'sequence', 'split', 'QP', 'model', 'dataset']}
            for field in ['frames', 'real_bytes', 'bpp', 'kbps', 'bitstream_sha256', 'source_sha256', 'source_RGB_sha256']:
                record[field + '_equal'] = float(r[field]) == float(om[k][field]) if field in ['frames', 'real_bytes', 'bpp', 'kbps'] else r[field] == om[k][field]
            for field in ['source_frame_indices', 'actual_qps']:
                record[field + '_equal'] = cp[field] == op[field]
            record['metric_weights_equal'] = cp['metric_hashes'] == expected_metric_hashes
            record['independent_decoder_verified'] = cp['independent_decode'] and op['independent_decode']
            assert all(value for name, value in record.items() if name.endswith(('_equal', '_verified'))), record
            checks.append(record)
    dump_csv(OUT / 'point_pairing_checks.csv', checks)
    normalized = []
    for row in original + current:
        model = row.get('model', 'GVC_RT_original')
        normalized.append(dict(dataset=dataset_map[key(row)], model=model, domain=row['domain'], sequence=row['sequence'], split=row['split'], QP=int(row['QP']), LoRA_step=2000 if model in ['A_control', 'B_expanded'] else 0, frames=int(row['frames']), real_bytes=int(row['real_bytes']), bpp=float(row['bpp']), kbps=float(row['kbps']), **{m: float(row[m]) for m in METRICS}, bitstream_sha256=row['bitstream_sha256'], source_sha256=row['source_sha256'], source_RGB_sha256=row['source_RGB_sha256'], measurements_reused_this_review=True, measurement_source=str(ORIGINAL / 'metrics_original.csv') if model == 'GVC_RT_original' else str(EXP / 'metrics.csv')))
    assert Counter(r['model'] for r in normalized) == {m: 135 for m in MODELS}
    dump_csv(OUT / 'metrics_with_original.csv', normalized)
    grouped = defaultdict(list)
    for r in normalized:
        grouped[r['dataset'], r['model'], r['QP']].append(r)
    means = []
    for dataset in DATASETS:
        for model in MODELS:
            for qp in [0, 5, 9]:
                rr = grouped[dataset, model, qp]
                means.append(dict(dataset=dataset, model=model, QP=qp, videos=len(rr), **{m: statistics.mean(r[m] for r in rr) for m in ['bpp', 'kbps'] + METRICS}))
    dump_csv(OUT / 'rd_points_by_dataset.csv', means)
    bymean = {(r['dataset'], r['model'], r['QP']): r for r in means}
    deltas = []
    for dataset in DATASETS:
        for model in MODELS[1:]:
            for metric in METRICS:
                values = []
                for qp in [0, 5, 9]:
                    a, b = bymean[dataset, model, qp][metric], bymean[dataset, 'GVC_RT_original', qp][metric]
                    values.append(a - b if metric in ['PSNR', 'MS_SSIM'] else 100 * (a / b - 1))
                deltas.append(dict(dataset=dataset, model=model, reference='GVC_RT_original', metric=metric, delta=statistics.mean(values), unit='dB' if metric == 'PSNR' else ('absolute' if metric == 'MS_SSIM' else 'percent'), weighting='equal-video mean within QP; equal mean of QP 0/5/9 changes'))
    dump_csv(OUT / 'changes_vs_original.csv', deltas)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    for dataset in DATASETS:
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        for ax, metric in zip(axes.flat, METRICS):
            for model in MODELS:
                points = [bymean[dataset, model, q] for q in [0, 5, 9]]
                ax.plot([r['bpp'] for r in points], [r[metric] for r in points], label=LABELS[model], color=COLORS[model], marker=MARKERS[model], linewidth=1.7, markersize=5, linestyle='--' if model == 'GVC_RT_original' else '-')
            ax.set_xlabel('Actual bpp (equal-video mean)')
            ax.set_ylabel(metric.replace('WarpingError_', 'Warp error ') + (' (higher is better)' if metric in ['PSNR', 'MS_SSIM'] else ' (lower is better)'))
            ax.grid(alpha=.22)
        axes.flat[-1].axis('off')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].legend(handles, labels, loc='center', fontsize=9, frameon=False)
        fig.suptitle(TITLES[dataset] + ' | original GVC-RT + TCG + LoRA A/B', fontsize=15)
        fig.text(.5, .015, 'Same 64-frame windows and bitstreams | measured QP 0 / 5 / 9 | existing test set reused | no BD-rate estimate', ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .04, 1, .95))
        fig.savefig(OUT / (dataset + '_with_original.png'), dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for ax, metric in zip(axes, ['FloLPIPS', 'LPIPS', 'DISTS']):
            for model in MODELS[1:]:
                pp = [bymean[dataset, model, q] for q in [0, 5, 9]]
                ys = [100 * (p[metric] / bymean[dataset, 'GVC_RT_original', p['QP']][metric] - 1) for p in pp]
                ax.plot([p['bpp'] for p in pp], ys, label=LABELS[model], color=COLORS[model], marker=MARKERS[model], linewidth=1.6)
            ax.axhline(0, color=COLORS['GVC_RT_original'], linestyle='--', label='Original GVC-RT reference')
            ax.set_xlabel('Actual bpp (equal-video mean)')
            ax.set_ylabel(metric + ' change vs original (%)')
            ax.grid(alpha=.22)
        fig.suptitle(TITLES[dataset] + ' | relative changes vs original GVC-RT (lower is better)', fontsize=13)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8)
        fig.tight_layout(rect=(0, .13, 1, .95))
        fig.savefig(OUT / (dataset + '_relative_to_original.png'), dpi=160)
        plt.close(fig)
    audit = dict(status='PASS', date='2026-11-08', scope='CPU-only reuse and plotting; no new encoding, decoding or GPU training', original_code_commit=protocol['original_commit'], original_checkpoint_checks=checkpoint_checks, original_points=135, current_points=405, pairings=len(checks), all_pairs_same_source_frames_QP_schedule_source_RGB_bitstreams_and_metric_weights=True, metric_protocol='matched_rgb_fp32_v2', original_model_has_TCG=False, original_model_loads_adapted_checkpoint=False, current_baseline='B+1500 matched-FP32, frozen composite receiver plus TCG, no LoRA', frames_per_video=64, QPs=[0, 5, 9], dataset_counts={ds: len(grouped[ds, 'baseline', 0]) for ds in DATASETS}, datasets_pooled=False, existing_test_videos_reused=True, full_UVG_claim=False, full_ULong_claim=False, percent_changes_are_not_BDrate=True, upstream_sources={str(p): sha(p) for p in [ORIGINAL / 'metrics_original.csv', ORIGINAL / 'protocol.json', ORIGINAL / 'final_integrity.json', EXP / 'metrics.csv', EXP / 'protocol.json']}, graphs=sorted(p.name for p in OUT.glob('*.png')))
    (OUT / 'reuse_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    lines = ['# 含原始 GVC-RT 的独立数据集 R-D 对比', '', '日期：2026-11-08。仅复用已核验实测数据，未重新编码、解码或训练。', '', '原版数据来自 2026-11-05 独立评测 metrics_original.csv，加载项目 checkpoints/GVC-RT_I.pt 与 GVC-RT_P.pt；原版源码固定于 b2a0895c0cf31cf2be3ab1c2571ca07c05866c03，不包含 TCG，不加载实验适配检查点。原版权重当前 SHA 与历史协议一致。', '', '45 个视频、每视频同一段 64 帧、QP 0/5/9，共 135 点；与本轮每个模型的源文件/RGB 哈希、源帧索引、实际帧 QP 调度、码流字节/SHA/bpp、指标网络权重逐点对齐。原版历史评测为独立 fresh RANS 编解码；本次仅将其原始实测结果加入图表，不以 parent0 代替。', '', '## 四条线', '', '- 黑色虚线：原始 GVC-RT 独立评测。', '- 蓝色：TCG B+1500 matched-FP32，无 LoRA。', '- 橙色：TCG + LoRA A，stage3/head，开发集最佳 2000 步。', '- 绿色：TCG + LoRA B，A + upsample.conv1，开发集最佳 2000 步。', '', '本轮基础模型含 TCG，不能称为原版；A/B 表示本轮 LoRA 覆盖实验，不是早期数据扩充的 A/B。', '', '## 相对原版的变化', '', '按每个数据集内视频等权计算各 QP 均值，再平均三个 QP 的相对变化；各数据集不混合。负的 LPIPS/DISTS/FloLPIPS 变化表示改善，不表示同质量码率节省。', '', '|数据集|模型|FloLPIPS|LPIPS|DISTS|PSNR dB|GT-flow warp error|', '|---|---|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        for model in MODELS[1:]:
            vals = {r['metric']: r['delta'] for r in deltas if r['dataset'] == ds and r['model'] == model}
            lines.append('|' + '|'.join([ds, model] + [f'{vals[m]:+.4f}' + ('' if m == 'PSNR' else '%') for m in ['FloLPIPS', 'LPIPS', 'DISTS', 'PSNR', 'WarpingError_GTFlow_PWC']]) + '|')
    lines += ['', 'ULong 上 LoRA 相对 TCG 的 DISTS 小幅回退，与最终模型相对原版的 DISTS 改善是两个不同参照。不能称“ULong DISTS 比原版差”。同时，UVG 两视频的 FloLPIPS/LPIPS 仍差于原版，各组 PSNR 与 GT-flow warp 指标也有退化，不能声称全面胜过原版。', '', '## 图和数据', '']
    for ds in DATASETS:
        lines.append(f'- {TITLES[ds]}：[R-D 曲线]({ds}_with_original.png) · [相对原版差值]({ds}_relative_to_original.png)')
    lines += ['', '- [540 个实测点](metrics_with_original.csv)', '- [各数据集 48 个 R-D 点](rd_points_by_dataset.csv)', '- [相对原版的分数据集统计](changes_vs_original.csv)', '- [405 项配对检查](point_pairing_checks.csv)', '- [来源与复用审计](reuse_audit.json)', '', '这些结果使用既有反复评估过的测试集；ULong/UVG 标签均指明具体子集，不能声称全数据集或全新独立确认集的结果。指标处理与当前实验一致，不声称复现论文全部默认评测设置。', '', '建议下一步：固定 A2000 为轻量候选、B2000 为覆盖消融；先分析开发集按 QP 的 LPIPS/DISTS 梯度方向、裁剪与原生分辨率差异及时间一致性，再只改变一个因素开展受控实验。最终确认时补充事先保留且来源组独立的数据，不依据本轮测试集继续调参。', '']
    (OUT / 'README.md').write_text(chr(10).join(lines))
    print(json.dumps({'status': 'PASS', 'directory': str(OUT), 'pairings': len(checks), 'combined_rows': len(normalized), 'rd_rows': len(means), 'plot_count': len(audit['graphs']), 'original_checkpoint_checks': checkpoint_checks, 'deltas': deltas}, ensure_ascii=False))

if __name__ == '__main__':
    main()

