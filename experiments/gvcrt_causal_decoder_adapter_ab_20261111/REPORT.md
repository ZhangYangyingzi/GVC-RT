# Causal decoder adapter A/B: execution record

Status: IN_PROGRESS. ULong development only; no final test.

Experiment directory: `/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_causal_decoder_adapter_ab_20261111/experiments/gvcrt_causal_decoder_adapter_ab_20261111`
Branch: `codex/tcg-causal-decoder-adapter-ab-20261111`

## Fixed configuration

B300 SHA256: `7465b57f20bfe2a8d6064a87155cf45c31a488cdeaa68111a228c28f6335ea85`. Full recovery: Original I/P → pilot full receiver and persistent buffers → B1500 TCG → A2000 LoRA → B300 52-key last overlay. No E/F/H trained weights.
All inherited components are frozen. Existing five LoRA branches remain at s=0.5. Only the 21,696-parameter causal branch is optimized.
Adapter after ada_final, before frozen head: 320→32 1x1 no bias; depthwise 3x3x3 (groups32, temporal padding0); 32→320 1x1 with bias. Last projection weight/bias start at zero. Internal operations and residual sum are FP32.
History stores two detached raw h frames. Missing history uses left replication. Video/GOP resets clear it; continuous chunks preserve it. Training h46–63 is cached. Frame48 is rendered with current adapter weights. At each segment boundary previous RGB is detached; both frame outputs within a segment retain gradients. No backbone BPTT is claimed.

Spatial objective: LPIPS-Alex + 0.5 DISTS + 0.1383589248954108 RGB-L1.
B adds lambda=1.2676811734151341 times (RGB MC residual + 0.5 Sobel MC residual). Calibration used ten inherited training windows, finite/nonzero B-projection gradient norms, zero optimizer updates. See calibration_gradient_norms.csv.
Flow is frozen RAFT estimated from GT RGB, not annotated true flow. It is used only for supervision/scoring. Deployment never reads GT. Pixel-center bilinear warp, zero padding, align_corners=False, forward/backward consistency and bounds masks are inherited.
Each group: effective batch4, 100 updates, exact inherited400-window order once, seed20261111, AdamW lr1e-4 betas(0.9,0.999) eps1e-8 weight_decay1e-4 clip1. Checkpoints0/25/50/75/100. Supervises49–63 in segments49–52,53–56,57–60,61–63.
Training uses FP32 frozen reconstruction prefix and head; codec features follow inherited FP16. Deployment uses inherited FP16 codec/Bridge/backbone, FP32 TCG/LoRA/adapter, shared module head and FP32-first RGB. B300 uses the matching path.

## Completion and coverage

| Group | Committed updates |
|---|---:|
| A_spatial | 100 |
| B_spatial_temporal | 100 |

| State | Completed points / 96 |
|---|---:|
| Original_step0 | 35 / 96 |
| B300_step0 | 96 / 96 |
| A_spatial_step50 | 32 / 96 |
| A_spatial_step100 | 32 / 96 |
| B_spatial_temporal_step50 | 32 / 96 |
| B_spatial_temporal_step100 | 30 / 96 |

Training source isolation and QP/crop/frame/order identities are inherited from H. No new dataset inventory scan or split. No training codec tensor for these400 windows was retained at startup; missing codec/GT-flow caches were regenerated with the pinned independent Original path. Native evaluation caches are read-only where reusable.
Original is decoded using pinned independent b2a0895 original source and original I/P weights in an isolated process. Cached tensors carry independent-decode/real-RANS/source/feature hashes. Historical scalar reuse requires every RGB frame, source frame, stream, QP, precision and metric-weight identity to match. Missing metrics are computed anew.

Frozen PWC GT forward flows used internally by FloLPIPS are cached as FP32 CPU tensors within each source/QP evaluation bundle and shared across model states. A direct check found bit-exact flow and FloLPIPS equality. This cache does not change the RAFT MC supervision, any training loss, or deployment inputs. The official FloLPIPS source/weight identities are in metric_source_manifest.json.

## Development numbers

CSV files contain every completed point, equal-video QP means and candidate-minus-reference paired differences. Negative error differences and positive PSNR differences retain their numeric meaning; no subjective flicker percentage is inferred.

| Model | Step | QP | n | bpp | PSNR | LPIPS-Alex | DISTS | FloLPIPS P→P | RGB MC P→P | Sobel MC P→P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_spatial | 50 | 0 | 11 | 0.0020944416 | 24.008048 | 0.18011768 | 0.10618538 | 0.16829723 | 0.016871224 | 0.0067006134 |
| A_spatial | 50 | 5 | 11 | 0.0070522773 | 25.828331 | 0.1260142 | 0.073589969 | 0.1177899 | 0.015093066 | 0.006786417 |
| A_spatial | 50 | 9 | 10 | 0.01642079 | 27.124523 | 0.095983399 | 0.059477695 | 0.086313448 | 0.013623165 | 0.0060770181 |
| A_spatial | 100 | 0 | 11 | 0.0020944416 | 23.993814 | 0.18039269 | 0.10616441 | 0.16862927 | 0.016863684 | 0.0067017734 |
| A_spatial | 100 | 5 | 11 | 0.0070522773 | 25.797604 | 0.12627894 | 0.073600744 | 0.1180689 | 0.01508727 | 0.0067889483 |
| A_spatial | 100 | 9 | 10 | 0.01642079 | 27.085424 | 0.096287633 | 0.059549705 | 0.08665557 | 0.013615733 | 0.0060790734 |
| B300 | 0 | 0 | 32 | 0.0019158333 | 24.107245 | 0.18625056 | 0.10852167 | 0.18408114 | 0.016069916 | 0.0067526832 |
| B300 | 0 | 5 | 32 | 0.0069668447 | 26.458508 | 0.11614086 | 0.071960529 | 0.11186831 | 0.013574805 | 0.0061206268 |
| B300 | 0 | 9 | 32 | 0.015992852 | 27.357931 | 0.095882007 | 0.060900029 | 0.091178717 | 0.013017411 | 0.005995354 |
| B_spatial_temporal | 50 | 0 | 11 | 0.0020944416 | 24.010882 | 0.18005839 | 0.10617141 | 0.16827143 | 0.016853548 | 0.0066822113 |
| B_spatial_temporal | 50 | 5 | 11 | 0.0070522773 | 25.836009 | 0.12599249 | 0.073561579 | 0.11774362 | 0.015069937 | 0.0067668976 |
| B_spatial_temporal | 50 | 9 | 10 | 0.01642079 | 27.134092 | 0.095947564 | 0.059443938 | 0.086277299 | 0.013601162 | 0.0060577845 |
| B_spatial_temporal | 100 | 0 | 10 | 0.0019767253 | 23.782271 | 0.18389924 | 0.1075495 | 0.1722616 | 0.016834831 | 0.0068330483 |
| B_spatial_temporal | 100 | 5 | 11 | 0.0070522773 | 25.78701 | 0.12643488 | 0.073660245 | 0.11823433 | 0.01504668 | 0.0067561737 |
| B_spatial_temporal | 100 | 9 | 9 | 0.017105825 | 27.353547 | 0.094456042 | 0.056739404 | 0.087827919 | 0.014124291 | 0.0062772972 |
| Original | 0 | 0 | 12 | 0.0021154886 | 23.566984 | 0.19962124 | 0.12248899 | 0.19611997 | 0.020099806 | 0.0086127894 |
| Original | 0 | 5 | 12 | 0.0073782864 | 25.789856 | 0.13703684 | 0.084960474 | 0.12727102 | 0.015988345 | 0.0071066411 |
| Original | 0 | 9 | 11 | 0.017179663 | 26.955556 | 0.10437853 | 0.069453869 | 0.09345184 | 0.014281869 | 0.0062768357 |

## Checks and repaired exceptions

Mechanical preflight passed: exact zero-branch equality in both precisions; future perturbation causality; continuous/chunked equality; reset isolation and first-history replication; finite losses/gradients; frozen head input gradients; actual adapter/head-LoRA calls in training/deployment. Upstream adapter gradients were zero at zero B and became nonzero after a disposable B update.
CUDA grid_sample backward does not implement strict determinism. Preflight initially stopped on inherited strict mode. Shared training context now uses deterministic algorithms with warn_only=True; A/B seeds, order and optimizer settings match. The initial failed check made no formal optimizer update. No independent RMS diagnosis was run. The inherited H calibration list initially covered four QPs; those provisional runs (11 updates per group) and coefficient were preserved under audits/superseded_calibration_qp_coverage, excluded from the formal budget, and replaced by one fixed window for each QP0–9 followed by a fresh common-step0 restart.
Original source model hashes and stream hashes, all checkpoint receipts and byte identities are audited in integrity.json. Non-task processes were preserved.

## Artifacts

`protocol.json`, `source_manifest.json`, `dependency_imports.json`, `calibration_gradient_norms.csv`, `training_logs/`, per-group training receipts, `checkpoint_manifest.json`, `metrics.csv`, `metrics_by_QP.csv`, `paired_differences*.csv`, `plots/development/`, `media/`, `integrity.json`, `status.json`, `publication_status.json`.
Main R-D plots show both step100 candidates; separate step50 supplements use the same real-bpp axis. Metrics and plots are ULong development, never final-test aggregates.
Media cases/frames/ROI were fixed before training in manifests/media_cases.json. Native FFV1 and 1:1 ROIs use identical RGB8 rounding and fps. Weights, dataset, large caches and lossless videos remain local with hashes.

Performance status: PENDING_UNCONTENDED_GPU. Method and observations are in performance.json. No speed claim is inferred from contended training/evaluation timings. No zero-overhead claim is made.
