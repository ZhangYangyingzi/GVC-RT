"""Read-only audited aggregation of existing 12-video development measurements."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'gvcrt_tcg_flicker_pilot_abcd_20261109'
MODELS = [('Original', 0, 'Original GVC-RT'),
          ('A2000', 0, 'A2000 s0.5'), ('B', 300, 'B300 s0.5')]
B_SHA = '7465b57f20bfe2a8d6064a87155cf45c31a488cdeaa68111a228c28f6335ea85'
A_SHA = '8a69765befa6ed267d0a95d3932cd63d9513c1da9818fd2126d8bc1c165f5e4e'
METRICS = ['PSNR', 'MS_SSIM', 'LPIPS', 'DISTS', 'FloLPIPS',
           'WarpingError_GTFlow_PWC', 'WarpingError_ReconFlow_PWC',
           'MCResidual_RGB_RAFT_pixelcenter_v1_PtoP',
           'MCResidual_Sobel_RAFT_pixelcenter_v1_PtoP']
HIGHER = {'PSNR', 'MS_SSIM'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def dump(name, value):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.partial')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    t.replace(p)


def csv_write(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def mean(rows, key):
    return sum(float(r[key]) for r in rows) / len(rows)


def main():
    identity = {'task': 'B300_dataset_rd_existing_ULong_development_12',
                'source': str(SOURCE), 'new_training_updates': 0,
                'new_GPU_evaluation_points': 0, 'fixed_B300_SHA256': B_SHA}
    if (ROOT / 'task_identity.json').exists():
        assert load(ROOT / 'task_identity.json') == identity, 'DIFFERENT_TASK'
    else:
        dump('task_identity.json', identity)
    dump('status.json', {'status': 'RUNNING', 'complete': False, 'PID': os.getpid()})
    assert load(SOURCE / 'final_integrity.json')['status'] == 'PASS'
    protocol = load(SOURCE / 'protocol.json')
    quick = load(SOURCE / 'manifests/quick_development.json')
    sequences = {x['sequence'] for x in quick}
    assert len(sequences) == 12
    protocol_sha = sha(SOURCE / 'protocol.json')
    implementation_sha = sha(SOURCE / 'source_manifest.json')
    files = ['protocol.json', 'source_manifest.json', 'final_integrity.json',
             'development_metrics.csv', 'manifests/quick_development.json']
    protected = {str(SOURCE / f): sha(SOURCE / f) for f in files}
    with (SOURCE / 'development_metrics.csv').open() as f:
        tabulated = {(r['model'], int(r['step']), r['sequence'], int(r['QP'])): r
                     for r in csv.DictReader(f)}
    points, evidence, indexed = [], [], {}
    for model, step, label in MODELS:
        folder = SOURCE / f'evaluation/{model}/step_{step:04d}/points'
        paths = list(folder.glob('*.json'))
        assert len(paths) == 36
        for f in sorted(paths):
            p = load(f)
            r = p['result']
            assert p['status'] == 'PASS' and p['frozen_components_unchanged']
            assert p['real_RANS'] and p['codec_features_match_original']
            assert r['model'] == model and r['step'] == step
            assert r['domain'] == 'ULong_development_subset_12'
            assert r['split'] == 'development' and r['sequence'] in sequences
            assert r['frames'] == 64 and r['transitions'] == 63 and r['QP'] in [0, 5, 9]
            assert p['identity']['protocol_SHA256'] == protocol_sha
            assert p['identity']['implementation_SHA256'] == implementation_sha
            assert p['identity']['initial_A2000_SHA256'] == A_SHA
            if model == 'B':
                assert r['checkpoint_SHA256'] == B_SHA
            key = (model, step, r['sequence'], r['QP'])
            assert key not in indexed
            for k in ['bpp', 'kbps', *METRICS]:
                assert math.isfinite(r[k]) and float(tabulated[key][k]) == r[k]
            assert p['bytes_consumed'] == r['real_bytes']
            for rel, expected in p['record_SHA256'].items():
                assert sha(SOURCE / rel) == expected
            frame_path = folder.parent / 'frames' / (f'{r["sequence"]}_qp{r["QP"]}.csv')
            transition_path = folder.parent / 'transitions' / frame_path.name
            with frame_path.open() as h:
                frame_rows = list(csv.DictReader(h))
            with transition_path.open() as h:
                transition_rows = list(csv.DictReader(h))
            assert len(frame_rows) == 64 and len(transition_rows) == 63
            assert sum(int(x['real_bits']) for x in frame_rows) == r['real_bytes'] * 8
            mse = sum(float(x['pixel_SSE']) for x in frame_rows) / (64 * 1920 * 1080 * 3)
            assert abs(r['PSNR'] + 10 * math.log10(max(mse, 1e-15))) < 1e-9
            for k in ['LPIPS', 'DISTS', 'MS_SSIM']:
                assert abs(mean(frame_rows, k) - r[k]) < 1e-9
            points.append(dict(model_label=label, measurement_reused=True,
                               measurement_source=str(f), **r))
            indexed[key] = p
            evidence.append(dict(point=str(f), SHA256=sha(f), records=p['record_SHA256'],
                                 metric_network_hashes=p['metric_network_hashes']))
    assert len(points) == 108
    streams = []
    for sequence in sorted(sequences):
        for q in [0, 5, 9]:
            reference = indexed[('Original', 0, sequence, q)]
            for model, step, _ in MODELS[1:]:
                p = indexed[(model, step, sequence, q)]
                for k in ['source_RGB_SHA256', 'bitstream_SHA256', 'real_bytes',
                          'bpp', 'kbps', 'codec_trace_SHA256']:
                    assert reference['result'][k] == p['result'][k], (sequence, q, k)
                for k in ['source_row_SHA256', 'source_cache_identity']:
                    assert reference['identity'][k] == p['identity'][k]
                assert p['actual_qps'] == reference['actual_qps']
                assert p['metric_network_hashes'] == reference['metric_network_hashes']
            r = reference['result']
            streams.append(dict(sequence=sequence, external_QP=q, real_bytes=r['real_bytes'],
                                bpp=r['bpp'], stream_SHA256=r['bitstream_SHA256'],
                                source_RGB_SHA256=r['source_RGB_SHA256'],
                                codec_trace_SHA256=r['codec_trace_SHA256'], status='PASS'))
    summaries = []
    for model, step, label in MODELS:
        for q in [0, 5, 9]:
            rr = [r for r in points if r['model'] == model and r['QP'] == q]
            assert len(rr) == 12
            summaries.append(dict(dataset='ULong_development_subset_12', split='development',
                                  model=model, model_label=label, step=step, QP=q,
                                  independent_videos=12, frames_per_video=64,
                                  **{k: mean(rr, k) for k in ['bpp', 'kbps', *METRICS]}))
    gains, per_video = [], []
    for ref in ['Original', 'A2000']:
        for q in [0, 5, 9]:
            base = next(r for r in summaries if r['model'] == ref and r['QP'] == q)
            candidate = next(r for r in summaries if r['model'] == 'B' and r['QP'] == q)
            for k in METRICS:
                differences = []
                for sequence in sorted(sequences):
                    a = indexed[('B', 300, sequence, q)]['result'][k]
                    b = indexed[(ref, 0, sequence, q)]['result'][k]
                    delta = a - b
                    differences.append(delta)
                    per_video.append(dict(sequence=sequence, QP=q, candidate='B300 s0.5',
                                          reference=ref, metric=k, candidate_value=a,
                                          reference_value=b, absolute_delta=delta,
                                          relative_percent=100 * (a / b - 1) if b else None))
                gains.append(dict(dataset='ULong_development_subset_12', reference=ref,
                                  candidate='B300 s0.5', QP=q, metric=k,
                                  direction='higher_is_better' if k in HIGHER else 'lower_is_better',
                                  reference_mean=base[k], candidate_mean=candidate[k],
                                  absolute_delta=candidate[k] - base[k],
                                  relative_percent=100 * (candidate[k] / base[k] - 1),
                                  improved_videos=sum(x > 0 if k in HIGHER else x < 0 for x in differences),
                                  worsened_videos=sum(x < 0 if k in HIGHER else x > 0 for x in differences),
                                  equal_videos=sum(x == 0 for x in differences),
                                  video_denominator=12, bpp_delta=candidate['bpp'] - base['bpp']))
    csv_write('metrics_paired.csv', points)
    csv_write('rd_points_by_qp.csv', summaries)
    csv_write('gains_by_qp.csv', gains)
    csv_write('paired_deltas_by_video_qp.csv', per_video)
    csv_write('stream_identity_checks.csv', streams)
    dump('protocol.json', dict(task=identity, source_protocol_SHA256=protocol_sha,
                              models=MODELS, dataset='ULong_development_subset_12',
                              videos=12, frames=64, QP=[0, 5, 9],
                              aggregation='equal video weights within each QP; each video PSNR from total SSE',
                              relative_percent='100*(candidate_mean/reference_mean-1)',
                              no_cross_QP_average=True, no_BD_rate=True, no_extrapolation=True,
                              new_inference=False, historical_other_datasets_not_available_for_B300=True,
                              precision=protocol['precision']))
    dump('source_manifest.json', dict(protected_files=protected, point_evidence=evidence,
                                     build_script_SHA256=sha(__file__)))
    (ROOT / 'plots').mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = {'PSNR': 'PSNR (dB)', 'MS_SSIM': 'MS-SSIM',
             'MCResidual_RGB_RAFT_pixelcenter_v1_PtoP': 'RGB temporal residual (P-to-P)',
             'MCResidual_Sobel_RAFT_pixelcenter_v1_PtoP': 'Sobel temporal residual (P-to-P)'}
    styles = [('Original', '#333333', 'o'), ('A2000', '#1976d2', 's'), ('B', '#d95f02', '^')]

    def panel(ax, metric):
        for model, color, marker in styles:
            rows = sorted((r for r in summaries if r['model'] == model), key=lambda x: x['bpp'])
            ax.plot([r['bpp'] for r in rows], [r[metric] for r in rows],
                    color=color, marker=marker, label=rows[0]['model_label'])
        ax.set_xlabel('Measured bpp (equal-video mean)')
        ax.set_ylabel(names.get(metric, metric) + (' [higher is better]' if metric in HIGHER else ' [lower is better]'))
        ax.grid(alpha=.2)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, k in zip(axes.flat, METRICS):
        panel(ax, k)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle('ULong development subset: 12 videos, 64 frames each; measured QP 0/5/9\nOriginal GVC-RT vs A2000 s0.5 vs B300 s0.5; not a full-dataset test', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(ROOT / 'plots/ULong_dev12_RD_all_metrics.png', dpi=150)
    plt.close(fig)
    for k in METRICS:
        fig, ax = plt.subplots(figsize=(7, 5))
        panel(ax, k)
        ax.legend()
        ax.set_title('ULong development subset: 12 videos, QP 0/5/9')
        fig.tight_layout()
        fig.savefig(ROOT / 'plots' / (k + '_RD.png'), dpi=150)
        plt.close(fig)
    lines = ['# B300 data-set-level measured R-D comparison', '',
             'Scope: the existing ULong development subset of 12 videos. This is not the full ULong data set, the 32-video held-out set, or a cross-data-set test. No new decoding or training was performed.', '',
             'Fixed states: independent Original GVC-RT; A2000 s0.5; flicker-pilot branch B step300 s0.5 (TCG + LoRA + Bridge.mlp[3]). B300 is not the earlier expanded-pool B+1500 TCG checkpoint.', '',
             '108 reused measured points, 36 per model. Native 64 frames per video, QP0/5/9. Spatial metrics use64frames; temporal metrics use63transitions; sustained P-to-P residual metrics use62transitions. The separate person_dog/Cactus 128-frame comparison is not pooled here.', '',
             'Within each QP, all12videos have equal weights. Per-video PSNR derives from total pixel SSE; the data-set PSNR is the equal-video mean of those PSNRs. All matching points have identical source/RGB, actual QP schedule, real stream SHA/bytes/bpp, codec trace and metric networks. Curves connect only the three actual measured points. No BD-rate or extrapolation.', '',
             '|QP|Mean bpp|Model|PSNR|MS-SSIM|LPIPS|DISTS|FloLPIPS|',
             '|---:|---:|---|---:|---:|---:|---:|---:|']
    for q in [0, 5, 9]:
        for r in (x for x in summaries if x['QP'] == q):
            lines.append(f'|{q}|{r["bpp"]:.8f}|{r["model_label"]}|' + '|'.join(f'{r[k]:.7f}' for k in ['PSNR', 'MS_SSIM', 'LPIPS', 'DISTS', 'FloLPIPS']) + '|')
    lines += ['', 'B300 differences below use 100*(candidate mean/reference mean-1). Negative values mean a numerical reduction; metric direction is explicitly retained. PSNR absolute differences are in dB.', '',
              '|Reference|QP|LPIPS change %|DISTS change %|FloLPIPS change %|Sobel residual change %|PSNR difference dB|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for ref in ['Original', 'A2000']:
        for q in [0, 5, 9]:
            g = {x['metric']: x for x in gains if x['reference'] == ref and x['QP'] == q}
            lines.append(f'|{ref}|{q}|' + '|'.join(f'{g[k]["relative_percent"]:+.4f}' for k in ['LPIPS', 'DISTS', 'FloLPIPS', 'MCResidual_Sobel_RAFT_pixelcenter_v1_PtoP']) + f'|{g["PSNR"]["absolute_delta"]:+.5f}|')
    lines += ['', 'All per-video differences, including increases and decreases, are retained in paired_deltas_by_video_qp.csv. No new model selection or training is authorized by this descriptive aggregation. B300 has not been evaluated on the complete HEVC/UVG/MCL or32-video ULong held-out grids in this package.', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    for path, expected in protected.items():
        assert sha(path) == expected, 'SOURCE_CHANGED'
    dump('final_integrity.json', dict(status='PASS', reused_points=108, videos=12, models=3,
                                     QP=[0, 5, 9], stream_pairs=36, all_matched_bpp_identical=True,
                                     record_hashes_and_frame_aggregates_verified=True,
                                     source_files_unchanged=True, plots=10, time=time.time()))
    dump('status.json', dict(status='COMPLETE', complete=True, training_updates=0,
                            new_GPU_evaluation_points=0, reused_measurements=108,
                            publication_status='PENDING', time=time.time()))
    print(json.dumps(dict(status='COMPLETE', report=str(ROOT / 'REPORT.md'), rows=108)))


if __name__ == '__main__':
    main()
