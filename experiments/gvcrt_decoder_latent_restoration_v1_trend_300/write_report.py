"""Render factual settings, completion state and every quality point; no mechanism claims."""
import csv
from datetime import datetime, timezone
from common_run import HERE, load


def run():
    audit=load(HERE/'delivery_audit.json')
    timing=load(HERE/'output_branch_timing.json')
    with (HERE/'results_summary.csv').open(newline='') as f: results=list(csv.DictReader(f))
    lines=['# Decoder latent restoration v1: factual execution report','',
      'Status: all local training, evaluation and delivery checks passed. GitHub publication and remote SHA verification remain pending; see `publication_status.json`. No COMPLETE.json is written before publication succeeds.','',
      '## Protocol and actual execution','',
      '- Fixed native I/P codec, quantization, entropy and A reference trajectory. No extra tokens, adaptive quantizer, history restoration, temporal loss, rate search, FID or BD-rate.',
      '- A uses the original I/P weights, bridge from G_basic_100, and G from visual-condition-v1 A/step_300.pt. The specified generator file SHA256 is `17a7635ae6e4f74922b5bfe368a790bf2f48b3a7bff2b2f516c4dea24148404f`. Effective loaded component hashes are in `weight_manifest_effective.json`.',
      '- G trains only its generator. R/RQ freeze codec, bridge and G and train only the 18-channel continuous-latent residual Adapter. R uses a zero map; RQ uses log of independently decoded q_actual with all 128 channels and spatial variation preserved. q_recon remains a separate generator input; rho is identically 1.',
      '- Paired v5 plan SHA256: `b9f7c050449be8fcf4751b6559a280875cd4378fe59ad7386a36e3a0639ee2b9`. Exactly 300 plan rows, batch 1, 256×256, 16 frames: I0, P1–P11 warmup, P12–P15 supervised with their mean loss and one optimizer update. All branches share the native A decoding cache and training-stream SHA sequence.',
      '- Loss: LPIPS(alex, normalize=True) + 0.1383589248954108 × RGB L1 on clamped output mapped from [-1,1] to [0,1]. AdamW: G 1e-6; Adapter 1e-4; betas (0.9,0.999), eps 1e-8, weight decay 1e-4, clip 1.0. Seed 20261027. Smoke updates 1 and 2 are included in 300.',
      '- FP16 codec; FP32 bridge, G and Adapter in training and inference. The optional CUDA inference extension could not be imported in executed processes, which used the repository PyTorch fallback. Native entropy streams were still decoded and their byte counts verified.',
      '- Actual training GPU: G on physical 0; R/RQ on physical 4. Latest user authorization allows 0 and 4. The original GPU 4–7 restriction was superseded by explicit user instructions. No external process was terminated.',
      '- Same step count is not reported as equal compute or equal convergence. Per-update time includes GT reads and training, and excludes cache construction and full checkpoint I/O; memory is peak reserved memory.', '',
      '| Branch | Effective updates | Trainable parameters | Nonzero-gradient parameters: step 1 / 2 / 100 / 300 | Sum update seconds | Peak reserved MiB |',
      '| --- | ---: | ---: | --- | ---: | ---: |',
      '| A | 0 (fixed) | 0 | n/a | n/a | n/a |']
    for b in ('G','R','RQ'):
        t=audit['training'][b];g=t['parameters_with_nonzero_gradient']
        lines.append(f"| {b} | 300 | {t['trainable_parameters']} | {' / '.join(str(g[str(k)]) for k in (1,2,100,300))} | {t['summed_update_seconds']:.3f} | {t['max_reserved_MiB']:.1f} |")
    lines += ['', 'R/RQ have equal nominal Adapter parameter counts and identical initial model state/RNG; their counts of parameters receiving nonzero gradients differ as recorded above. Every branch has step0, step100 and step300 plus latest, optimizer state, RNG, data cursor and effective-update count. Formal quality used only step300. Checkpoint paths and SHA are in `delivery_audit.json` and `large_file_manifest.json`.', '',
      '## Required checks','',
      '- `preflight.json`: zero-initialized R/RQ outputs equal A; disabling restoration restores A; actual frozen G passes gradients to Adapter; earlier layers receive gradients after changing the output layer; no training optimizer step is used by this diagnostic. The 16-frame native preflight checks I-frame output, supervised-frame correspondence, quant maps and unchanged base references.',
      '- `training_status_*.json` and checkpoint audit: 300 effective updates each, matched frozen codec/bridge hashes, identical R/RQ initialization, finite recorded losses and shared training streams. R/RQ frozen G was checked during training.',
      '- `consistency_checks.json`, `decode_receipts/` and `delivery_audit.json`: 8 common complete streams, 32 separate decoder processes, exact bytes consumed, and 64-frame equality of reference feature, reference RGB/POC, y_hat, f, original u, q_actual and q_recon. I output remains identical. Enhanced output never replaces base references.',
      '- Decoders use stream, weights, fixed public dimensions and stream provenance configuration only. GT is first read in the separate quality stage after all decoding completes. Stream, checkpoint and source hashes are registered.', '',
      '## All quality points','',
      'UVal0, UVal1, ParkScene and Jockey, 64 frames each, external QP 2 and 7, IP=-1 and no extra reset. All native headers and the I frame count toward bytes/bpp. All methods for a video/QP have exactly the same rate. Metrics use uncompressed decoded tensors, not the display videos.', '',
      'The table lists all-64-frame values. `results_summary.csv` also contains all P63 summaries. RGB PSNR is calculated from aggregate SSE/MSE. LPIPS/DISTS are frame means; FloLPIPS uses all 63 adjacent transitions (I0→P1 included for both displayed scopes, each transition assigned to its destination P frame). `frame_metrics.csv` has 2048 rows; `flolpips_transitions.csv` has 2016 rows. `paired_differences.csv` retains all 96 signed rows for G−A, R−A, RQ−A, RQ−R, R−G and RQ−G, by sequence, QP and scope.', '',
      '| Sequence | QP | Method | Native bytes | bpp | LPIPS | DISTS | FloLPIPS | RGB PSNR dB |',
      '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        if r['scope']=='all64':
            lines.append(f"| {r['sequence']} | {r['external_qp']} | {r['branch']} | {r['real_total_bytes']} | {float(r['bpp']):.8f} | {float(r['LPIPS']):.6f} | {float(r['DISTS']):.6f} | {float(r['FloLPIPS']):.6f} | {float(r['PSNR']):.6f} |")
    lines += ['', '## Fixed figures, reconstruction locations and timing','',
      '- `figures/*_overview_frame{15,31,63}.png`: complete frames downscaled equally, red box marks the common central 256×256 crop. `*_frame{15,31,63}.png`: GT/A/G/R/RQ at that common crop. Frames and crop locations were fixed, not selected by quality.',
      '- `*_maps_frame{15,31,63}.png`: q_actual displayed as a channel mean for visualization only; R/RQ delta_u displayed as channel L2 norms. Each panel independently min/max scales to grayscale, so panel brightness cannot be read as an absolute cross-panel magnitude. Full channel/spatial q_actual is retained in local diagnostics and was used for conditioning without averaging.',
      '- `figures/training_curves.png` and `training_*.csv` retain training losses, components, gradients, delta RMS/relative norm, update duration and reserved memory.',
      '- `reconstruction_locations.csv` gives all 32 raw-tensor directories and display video locations. Display MP4 files were excluded from metric calculation. Local diagnostics preserve q_actual and delta_u for fixed frames.', '',
      '| Output branch | Median milliseconds | Difference from A milliseconds |',
      '| --- | ---: | ---: |']
    for b in ('A','R','RQ'):
        median=timing[b]['median_ms'];lines.append(f"| {b} | {median:.6f} | {median-timing['A']['median_ms']:.6f} |")
    lines += ['',
      'Timing used physical GPU 0, a 256×256 cached training sample, five warmup forwards and 20 synchronized samples per branch, excluding initialization, metric evaluation and file writes. These are sequential short wall-clock measurements, not isolated-device or full-resolution benchmarks. RQ−A is negative in this run; that signed observation is retained and cannot establish negative Adapter cost or a model speedup. No cross-GPU timing comparison is used.', '',
      '## Reproduction, local artifacts and limitations','',
      '- Use the existing gvc-rt Python environment and `run_pipeline.py`; `PROTOCOL.md` lists individual entrypoints. `verify_delivery.py` checks saved checkpoints, raw metric counts and aggregates, native/reference equivalence and manifest sizes on CPU. It does not rerun training or quality inference.',
      '- `large_file_manifest.json` lists absolute local paths, sizes and SHA for weights/checkpoints, cache, native training/evaluation streams, decoded tensors, previews, diagnostic maps and display videos created by this experiment. Input weights/data and metric dependencies are separately registered in their manifests. Large files remain ignored locally.',
      '- Public Python dependencies are tracked in the repository or the minimal dependency snapshot; `repository_dependency_manifest.json` and `metric_dependency_manifest.json` register remaining dependencies and external source/weight requirements. Original input data and pretrained weights must be provisioned at the registered paths or the entrypoints explicitly adapted. Hardcoded local paths are intentional execution provenance, not a portable standalone data package.',
      '- Worktree source differs from the original checkout at src/utils/video_writer.py; that module is not used by this experiment’s custom cv2 quality writer. Repository source hashes are registered. This report makes no research mechanism, bitrate-saving or convergence claim.',
      '- Operational history: GPU waiting, sandbox GPU invisibility, GitHub DNS failures and repeated approval-service HTTP 502 failures occurred before execution. The user launched the pipeline from the server terminal. GPU utilization scheduling was corrected following authorization for 0 and 4. Earlier blocker receipts are historical, superseded for local execution by pipeline_status.json and delivery_audit.json.',
      '- GitHub publication is currently blocked by automatic approval review: explicit destination/payload confirmation is requested before exporting experiment code, local-path manifests, CSVs and figures to ZhangYangyingzi/GVC-RT. No force-push or merge is used.', '',
      'Results root: `/Huang_group/zyyz/home_dir/tmp/gvcrt-decoder-latent-restoration-v1/experiments/gvcrt_decoder_latent_restoration_v1_trend_300`. Publication status is separate from completed local computation.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('FACTUAL_REPORT_WRITTEN')


if __name__ == '__main__':
    run()
