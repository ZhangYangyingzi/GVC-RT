# Decoder latent restoration v1: fixed protocol

- Baseline A uses the original I/P codec, bridge from `G_basic_100.pt`, and the generator from visual condition v1 `A/step_300.pt`. All weights are SHA checked.
- G, R and RQ start from A. G trains only the generator at 1e-6. R and RQ train identical Adapter structures at 1e-4 with paired initialization. The bridge and codec stay fixed.
- R conditions on a zero tensor with the decoded quantization map's shape. RQ conditions on `log(q_actual)`, where `q_actual` is independently reconstructed at the decoder and natively clipped to at least 0.5. No additional data is transmitted.
- Each branch consumes the same 300-row v5 training plan. Each row uses a 16-frame 256×256 window, native I/P encoding and independent A decoding. P1–P11 warm the base codec reference. P12–P15 contribute a quarter-weighted LPIPS(alex) + 0.1383589248954108 RGB L1 loss; one optimizer update follows. Smoke updates 1 and 2 count toward 300.
- AdamW betas (0.9, 0.999), eps 1e-8, weight decay 1e-4, gradient clip 1.0. Seed 20261027. Checkpoints at 0, 100 and 300. Formal quality uses only step 300.
- All four decoder processes use the same native A stream for each video and QP. The base decoder maintains the reference feature and RGB. Output enhancement runs after base decoding and does not replace reference state. Quality reads GT only after decoding finishes.
- Fixed evaluation: UVal0, UVal1, ParkScene, Jockey; 64 frames each; external QP 2 and 7; IP=-1 with no extra reset. Metrics: LPIPS, DISTS, FloLPIPS and RGB SSE based PSNR for all 64 and P63. Pair differences are signed first minus second and include every result.
- FP16 codec, FP32 bridge, generator and Adapter for training and inference. Loss and metrics use images mapped from [-1,1] to [0,1], with LPIPS `normalize=True`.

## Entrypoints

Use `/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python`. Per the latest user authorization, the current launcher is restricted to physical GPUs 0 and 4. Run `cache_training.py` once before `preflight.py`, then `train_branch.py --stop 2` for G, R and RQ, then `train_branch.py --stop 300` to resume. Run `prepare_streams.py` once. For each stream, run `decode_stream.py` in a separate process for A, G, R and RQ. Then run `evaluate_quality.py` and `finalize.py`. Every training step and cache row is atomic and resumable.

## Continuation and offline checks

`offline_audit.py` performs CPU provenance, native header parsing and synthetic Adapter connectivity checks. It never replaces `preflight.py` and performs no training optimizer updates. `repository_dependency_manifest.json` registers tracked repository Python dependencies and their relationship to the original checkout.

Run `run_pipeline.py` with the existing Python environment to schedule cache generation, GPU preflight, paired smoke/continuation training, independent decoding, quality evaluation, timing and final checks. It uses only authorized physical GPUs 0 and 4, waits when these devices are occupied, and reports driver/query failures. Logs remain under ignored `local_artifacts/logs/`. A second simultaneous pipeline instance is rejected by a local file lock. Completed training checkpoints are loaded with their optimizer, RNG, cursor and log rows; no additional updates beyond 300 are performed.

Latest GPU authorization: physical 0 and 4. Launch requires at least 24,000 MiB free memory; GPU utilization is recorded but does not block launch. This replaces the previous utilization <10% queue rule.

## Final local delivery checks

After finalize.py, run build_overviews.py and verify_delivery.py, then write_report.py. The pipeline includes the first two. verify_delivery.py checks step0/100/300 effective updates, cursor, optimizer state, R/RQ initial RNG, matching training streams, raw metric/transition counts and aggregates, native decoder reference equality, inventory sizes and checkpoint SHA. Publication is separate; a local PASS does not authorize COMPLETE.json before remote verification. See metric_dependency_manifest.json for external FloLPIPS/DISTS/LPIPS source and weight requirements.
