"""Objective completeness checks, paired deltas, fixed preview crops and inventory."""
import csv
import hashlib
from pathlib import Path

from common_run import HERE, dump, load, sha, table

PAIRS = [('G', 'A'), ('R', 'A'), ('RQ', 'A'), ('RQ', 'R'), ('R', 'G'), ('RQ', 'G')]
METRICS = ('LPIPS', 'DISTS', 'FloLPIPS', 'PSNR')


def run():
    from PIL import Image, ImageDraw
    import numpy as np
    import torch
    (HERE / 'figures').mkdir(exist_ok=True)
    streams = load(HERE / 'stream_manifest.json')
    summaries = []; deltas = []; checks = []; frame_metrics = []; transitions_all = []
    for item in streams:
        seq, qp = item['sequence'], item['external_qp']
        receipts = {b: load(HERE / 'local_artifacts/decoded' / f'{seq}_q{qp}' / b / 'receipt.json')
                    for b in ('A', 'G', 'R', 'RQ')}
        summary = {b: load(HERE / 'local_artifacts/quality' / f'{seq}_q{qp}' / b / 'summary.json')
                   for b in ('A', 'G', 'R', 'RQ')}
        for branch in ('A', 'G', 'R', 'RQ'):
            for key, destination in (('frame_csv', frame_metrics), ('transition_csv', transitions_all)):
                with Path(summary[branch][key]).open(newline='') as stream:
                    destination.extend(csv.DictReader(stream))
        base = receipts['A']
        for branch, receipt in receipts.items():
            assert receipt['stream_sha256'] == base['stream_sha256'] == item['stream_sha256']
            assert receipt['consumed_bytes'] == base['consumed_bytes'] == item['total_native_bytes']
            assert receipt['real_total_bytes'] == base['real_total_bytes']
            assert receipt['bpp'] == base['bpp']
            assert len(receipt['states']) == 64
            for i, state in enumerate(receipt['states']):
                for key in ('reference', 'reference_rgb', 'reference_poc', 'y_hat', 'f', 'u', 'q_actual', 'q_recon',
                            'actual_qp', 'payload_bytes', 'consumed_bytes'):
                    assert state[key] == base['states'][i][key], (seq, qp, branch, i, key)
            for scope in ('all64', 'P63'):
                summaries.append(dict(sequence=seq, external_qp=qp, branch=branch,
                                      scope=scope, real_total_bytes=receipt['real_total_bytes'],
                                      bpp=receipt['bpp'], **summary[branch][scope]))
            assert receipt['states'][0]['output_rgb'] == base['states'][0]['output_rgb']
        checks.append(dict(sequence=seq, external_qp=qp, status='PASS',
                           native_stream_sha256=base['stream_sha256'],
                           total_bytes=base['real_total_bytes'], frames=64,
                           reference_yhat_f_u_qactual_identical=True))
        for scope in ('all64', 'P63'):
            for first, second in PAIRS:
                deltas.append(dict(sequence=seq, external_qp=qp, scope=scope,
                                   comparison=f'{first}-{second}',
                                   **{metric: summary[first][scope][metric]-summary[second][scope][metric]
                                      for metric in METRICS}))
        preview = HERE / 'local_artifacts/previews' / f'{seq}_q{qp}'
        for frame in (15, 31, 63):
            labels = ['GT', 'A', 'G', 'R', 'RQ']
            crops = []
            for label in labels:
                image = Image.open(preview / f'{label}_{frame:02d}.png').convert('RGB')
                width, height = image.size
                x = max(0, (width-256)//2); y = max(0, (height-256)//2)
                crops.append(image.crop((x, y, x+256, y+256)))
            board = Image.new('RGB', (5*256, 280), 'white')
            draw = ImageDraw.Draw(board)
            for index, (label, crop) in enumerate(zip(labels, crops)):
                board.paste(crop, (index*256, 24))
                draw.text((index*256+8, 6), label, fill='black')
            board.save(HERE / 'figures' / f'{seq}_q{qp}_frame{frame:02d}.png')
            map_images = []
            for branch, field in (('RQ', 'q_actual'), ('R', 'delta_u'), ('RQ', 'delta_u')):
                diag = torch.load(HERE / 'local_artifacts/decoded' / f'{seq}_q{qp}' /
                                  branch / f'diagnostics_{frame:02d}.pt',
                                  map_location='cpu', weights_only=True)
                tensor = diag[field].float()
                array = tensor[0].square().sum(0).sqrt().numpy() if field == 'delta_u' else tensor[0].mean(0).numpy()
                scaled = ((array - array.min()) / max(float(np.ptp(array)), 1e-12) * 255).round().astype('uint8')
                map_images.append(Image.fromarray(scaled).resize((256, 256)))
            maps = Image.new('RGB', (3*256, 280), 'white'); draw = ImageDraw.Draw(maps)
            for index, (label, image) in enumerate(zip(('q_actual', 'R delta norm', 'RQ delta norm'), map_images)):
                maps.paste(image.convert('RGB'), (index*256, 24))
                draw.text((index*256+8, 6), label, fill='black')
            maps.save(HERE / 'figures' / f'{seq}_q{qp}_maps_frame{frame:02d}.png')
    table(HERE / 'results_summary.csv', summaries)
    table(HERE / 'paired_differences.csv', deltas)
    table(HERE / 'frame_metrics.csv', frame_metrics)
    table(HERE / 'flolpips_transitions.csv', transitions_all)
    dump(HERE / 'consistency_checks.json', checks)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for branch in ('G', 'R', 'RQ'):
        with (HERE / f'training_{branch}.csv').open(newline='') as stream:
            training = list(csv.DictReader(stream))
        assert len(training) == 300 and [int(x['step']) for x in training] == list(range(1, 301))
        axes[0].plot([int(x['step']) for x in training], [float(x['loss']) for x in training], label=branch)
        axes[1].plot([int(x['step']) for x in training],
                     [float(x['gradient_norm_preclip']) for x in training], label=branch)
    axes[0].set_ylabel('LPIPS + 0.1383589 RGB L1')
    axes[1].set_ylabel('Preclip gradient norm')
    axes[1].set_xlabel('Effective optimizer update')
    for axis in axes:
        axis.legend(); axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(HERE / 'figures/training_curves.png', dpi=150)
    plt.close(fig)
    manifest = []
    for path in sorted((HERE / 'local_artifacts').rglob('*')):
        if path.is_file():
            manifest.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
    dump(HERE / 'large_file_manifest.json', manifest)
    print('FINAL_CHECKS_PASS', len(summaries), len(deltas), flush=True)


if __name__ == '__main__':
    run()
