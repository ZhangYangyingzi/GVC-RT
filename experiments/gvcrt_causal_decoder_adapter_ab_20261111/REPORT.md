# Causal decoder adapter A/B: execution record

Status: COMPLETED_WITH_PENDING_PERFORMANCE. ULong development only; no final test.

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
| Original_step0 | 96 / 96 |
| B300_step0 | 96 / 96 |
| A_spatial_step50 | 96 / 96 |
| A_spatial_step100 | 96 / 96 |
| B_spatial_temporal_step50 | 96 / 96 |
| B_spatial_temporal_step100 | 96 / 96 |

Training source isolation and QP/crop/frame/order identities are inherited from H. No new dataset inventory scan or split. No training codec tensor for these400 windows was retained at startup; missing codec/GT-flow caches were regenerated with the pinned independent Original path. Native evaluation caches are read-only where reusable.
Original is decoded using pinned independent b2a0895 original source and original I/P weights in an isolated process. Cached tensors carry independent-decode/real-RANS/source/feature hashes. Historical scalar reuse requires every RGB frame, source frame, stream, QP, precision and metric-weight identity to match. Missing metrics are computed anew.

Frozen PWC GT forward flows used internally by FloLPIPS are cached as FP32 CPU tensors within each source/QP evaluation bundle and shared across model states. A direct check found bit-exact flow and FloLPIPS equality. This cache does not change the RAFT MC supervision, any training loss, or deployment inputs. The official FloLPIPS source/weight identities are in metric_source_manifest.json.

## Development numbers

CSV files contain every completed point, equal-video QP means and candidate-minus-reference paired differences. Negative error differences and positive PSNR differences retain their numeric meaning; no subjective flicker percentage is inferred.

| Model | Step | QP | n | bpp | PSNR | LPIPS-Alex | DISTS | FloLPIPS P→P | RGB MC P→P | Sobel MC P→P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_spatial | 50 | 0 | 32 | 0.0019158333 | 24.111764 | 0.18630049 | 0.10898 | 0.18411609 | 0.016083455 | 0.00676262 |
| A_spatial | 50 | 5 | 32 | 0.0069668447 | 26.467362 | 0.11612235 | 0.072310664 | 0.11178327 | 0.013586807 | 0.0061294325 |
| A_spatial | 50 | 9 | 32 | 0.015992852 | 27.368216 | 0.095841871 | 0.061161303 | 0.091061818 | 0.013029727 | 0.0060042698 |
| A_spatial | 100 | 0 | 32 | 0.0019158333 | 24.087807 | 0.18649463 | 0.10892689 | 0.18435377 | 0.016077376 | 0.0067636203 |
| A_spatial | 100 | 5 | 32 | 0.0069668447 | 26.426374 | 0.11636246 | 0.072286558 | 0.11213517 | 0.013583744 | 0.0061325191 |
| A_spatial | 100 | 9 | 32 | 0.015992852 | 27.319912 | 0.096082615 | 0.061193625 | 0.091467558 | 0.013025734 | 0.006007049 |
| B300 | 0 | 0 | 32 | 0.0019158333 | 24.107245 | 0.18625056 | 0.10852167 | 0.18408114 | 0.016069916 | 0.0067526832 |
| B300 | 0 | 5 | 32 | 0.0069668447 | 26.458508 | 0.11614086 | 0.071960529 | 0.11186831 | 0.013574805 | 0.0061206268 |
| B300 | 0 | 9 | 32 | 0.015992852 | 27.357931 | 0.095882007 | 0.060900029 | 0.091178717 | 0.013017411 | 0.005995354 |
| B_spatial_temporal | 50 | 0 | 32 | 0.0019158333 | 24.118142 | 0.18626512 | 0.10898593 | 0.18406886 | 0.016064902 | 0.0067455306 |
| B_spatial_temporal | 50 | 5 | 32 | 0.0069668447 | 26.478265 | 0.11610149 | 0.072312611 | 0.11174166 | 0.013563253 | 0.006110381 |
| B_spatial_temporal | 50 | 9 | 32 | 0.015992852 | 27.381007 | 0.095825311 | 0.061155582 | 0.091015239 | 0.01300506 | 0.0059847881 |
| B_spatial_temporal | 100 | 0 | 32 | 0.0019158333 | 24.079514 | 0.18657948 | 0.10898178 | 0.18443033 | 0.01604243 | 0.0067337426 |
| B_spatial_temporal | 100 | 5 | 32 | 0.0069668447 | 26.412414 | 0.11648866 | 0.072362179 | 0.11226297 | 0.013544061 | 0.0061002763 |
| B_spatial_temporal | 100 | 9 | 32 | 0.015992852 | 27.303487 | 0.096220448 | 0.06125957 | 0.091614218 | 0.012984374 | 0.0059740964 |
| Original | 0 | 0 | 32 | 0.0019158333 | 24.100589 | 0.20536066 | 0.12636838 | 0.20726631 | 0.017621831 | 0.0076982094 |
| Original | 0 | 5 | 32 | 0.0069668447 | 26.169861 | 0.13191379 | 0.08712271 | 0.12709458 | 0.014728384 | 0.0066854134 |
| Original | 0 | 9 | 32 | 0.015992852 | 26.977264 | 0.10901814 | 0.074146837 | 0.10347958 | 0.01407903 | 0.0064830991 |

## Checks and repaired exceptions

Mechanical preflight passed: exact zero-branch equality in both precisions; future perturbation causality; continuous/chunked equality; reset isolation and first-history replication; finite losses/gradients; frozen head input gradients; actual adapter/head-LoRA calls in training/deployment. Upstream adapter gradients were zero at zero B and became nonzero after a disposable B update.
CUDA grid_sample backward does not implement strict determinism. Preflight initially stopped on inherited strict mode. Shared training context now uses deterministic algorithms with warn_only=True; A/B seeds, order and optimizer settings match. The initial failed check made no formal optimizer update. No independent RMS diagnosis was run. The inherited H calibration list initially covered four QPs; those provisional runs (11 updates per group) and coefficient were preserved under audits/superseded_calibration_qp_coverage, excluded from the formal budget, and replaced by one fixed window for each QP0–9 followed by a fresh common-step0 restart.
Original source model hashes and stream hashes, all checkpoint receipts and byte identities are audited in integrity.json. Non-task processes were preserved.

## Artifacts

`protocol.json`, `source_manifest.json`, `dependency_imports.json`, `calibration_gradient_norms.csv`, `training_logs/`, per-group training receipts, `checkpoint_manifest.json`, `metrics.csv`, `metrics_by_QP.csv`, `paired_differences*.csv`, `plots/development/`, `media/`, `integrity.json`, `status.json`, `publication_status.json`.
Main R-D plots show both step100 candidates; separate step50 supplements use the same real-bpp axis. Metrics and plots are ULong development, never final-test aggregates.
Media cases/frames/ROI were fixed before training in manifests/media_cases.json. Native FFV1 and 1:1 ROIs use identical RGB8 rounding and fps. Weights, dataset, large caches and lossless videos remain local with hashes.

Performance status: PENDING_UNCONTENDED_GPU. Method and observations are in performance.json. No speed claim is inferred from contended training/evaluation timings. No zero-overhead claim is made.
