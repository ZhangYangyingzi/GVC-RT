# Corrected H-series B300 execution record

Status: RUNNING. See status.json for current stage.

All models restore original I/P → complete pilot generator/Bridge/persistent buffers → B+1500 TCG → A2000 LoRA → B30052-key overlay last. LoRA is retained without merging at s=0.5. Full Bridge801124, TCG137760, LoRA59392 are trainable; original decoder backbone, codec, I model, entropy and metric networks are frozen. H0 is a continuation control whose difference from B300 includes widening Bridge training and further updates; this report assigns no single-factor causal effect.

Base objective: LPIPS-Alex +0.5 DISTS +0.1383589248954108 RGB-L1 +0.05 RAFT-motion +2.603297563918803×(RGB MCResidual +0.5 Sobel MCResidual). Training LPIPS is Alex, evaluation records Alex and VGG separately ([0,1] RGB, normalize=True, v0.1 weights).

H1 adds hard-mask-area-normalized confidence-weighted latent differences with shared differentiable channel RMS, tau0.02. Flow is estimated by frozen RAFT on GT RGB, not a known true motion field. H2 additionally proposes four stateless paired translation losses/window; both sides disable TCG and cache updates; target is no_grad. No edge-floor, Teacher, Router, GAN, SFT, history alignment or new upsampler.

Fresh AdamW: Bridge2e-6, TCG1e-5, LoRA5e-5; betas(.9,.999), eps1e-8, weight_decay1e-4, globalclip1. Exactly four logical windows/update,64 frames/window, current-weight warmup1–48, supervised49–63 segmented4/4/4/3. Maximum100 updates; no automatic extension. GPU0 H0, GPU1 H1, GPU2 H2 if mechanically allowed, GPU3 serialized bounded cache/evaluation.

Calibration uses10 frozen training windows, no optimizer updates or development/test choice. lambda_lat=0.2953169862992426; lambda_equiv=0.10179495298276185.

H2 mechanical gate: MECHANICAL_CHECK_FAILED; median16px/8px error ratio atcrop64=0.16182493976184825. Fixed threshold0.1 operationalizes separation by more than one order of magnitude; no padding/crop/target tuning after measurements.

| Group | Added updates | Dev50 | Dev100 | Numerical screen |
|---|---:|---:|---:|---|
| H0_control | 8 | 0/96 | 0/96 | pending |
| H1_latent_self | 8 | 0/96 | 0/96 | pending |
| H2_latent_equiv | 0 | 0/96 | 0/96 | pending |

Mechanical screen compares every QP mean DISTS, LPIPS-Alex, LPIPS-VGG and FloLPIPS to≤1.005×B300 and requires meanQP P→P Sobel ratio S<1. Eligible50/100 checkpoints are selected by lowest S; differences below0.0005 prefer earlier. NOT_QUALIFIED never substitutes step0. This is a numerical screen, not a claim about human-visible flicker.

Aggregation: first64-frame video metrics, then equal video weights within dataset/split/QP. I→P/P→P temporal metrics are separate. Candidate RGB never enters the independent Original codec reference trajectory. Rates use real RANS bytes including actual I frame and stream headers. Byte counts, bpp, stream SHA and decoded codec-feature traces are paired checks.

Final test status: pending conditional numerical screen. No qualification means no new final R-D points. Existing32new-holdout/6historical ULong,5HEVC-B,2historical HoneyBee/Jockey UVG and30historical MCL-JCV remain distinct test plots when measured. No cross-dataset pooled curve, extrapolation or three-point BD-rate.

Video materials retain full-resolution single-model FFV1 and unscaled ROI, with labels, source frames, fps and SHA. Anonymous label mapping is separate. person_dog is recovered from audited original RGB frames; its MP4 is identification material only, historical exposure is retained, and it is excluded from training and model selection. No subjective experiment is claimed.

Raw development values: development_metrics.csv; paired baseline differences: development_paired_by_video_qp.csv; same-step H1−H0 and H2−H1 where available: development_matched_steps.csv. H2 missing rows are due to its mechanical gate, not replacement by another model. Edge/local energy values are diagnostics and not quality-improvement guarantees.

Measured equal-video aggregates:

| Phase | Dataset | Model | Step | QP | bpp | PSNR | Alex | VGG | DISTS | FloLPIPS | P→P Sobel |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

This report records execution, coverage and numbers only. Integrity/publication states are in final_integrity.json and publication_status.json.
