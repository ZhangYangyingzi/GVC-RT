# Corrected H-series B300 execution record

Status: RUNNING. See status.json for current stage.

All models restore original I/P → complete pilot generator/Bridge/persistent buffers → B+1500 TCG → A2000 LoRA → B30052-key overlay last. LoRA is retained without merging at s=0.5. Full Bridge801124, TCG137760, LoRA59392 are trainable; original decoder backbone, codec, I model, entropy and metric networks are frozen. H0 is a continuation control whose difference from B300 includes widening Bridge training and further updates; this report assigns no single-factor causal effect.

Base objective: LPIPS-Alex +0.5 DISTS +0.1383589248954108 RGB-L1 +0.05 RAFT-motion +2.603297563918803×(RGB MCResidual +0.5 Sobel MCResidual). Training LPIPS is Alex, evaluation records Alex and VGG separately ([0,1] RGB, normalize=True, v0.1 weights).

H1 adds hard-mask-area-normalized confidence-weighted latent differences with shared differentiable channel RMS, tau0.02. Flow is estimated by frozen RAFT on GT RGB, not a known true motion field. H2 additionally proposes four stateless paired translation losses/window; both sides disable TCG and cache updates; target is no_grad. No edge-floor, Teacher, Router, GAN, SFT, history alignment or new upsampler.

Fresh AdamW: Bridge2e-6, TCG1e-5, LoRA5e-5; betas(.9,.999), eps1e-8, weight_decay1e-4, globalclip1. Exactly four logical windows/update,64 frames/window, current-weight warmup1–48, supervised49–63 segmented4/4/4/3. Maximum100 updates; no automatic extension. Training allocation: GPU0 H0, GPU1 H1, GPU2 H2. After the three100-update runs finished, the user authorized a shared GPU0–3 pool for independent evaluation processes and bounded frozen-cache production. Each GPU runs at most one heavy process belonging to this experiment; unrelated processes are retained.

Calibration uses10 frozen training windows, no optimizer updates or development/test choice. lambda_lat=0.2953169862992426; lambda_equiv=0.10179495298276185.

H2 mechanical gate: DIAGNOSTIC_ONLY_USER_DIRECTED_TRAINING; median16px/8px error ratio atcrop64=0.16182493976184825. The assistant-added0.1 launch cutoff was withdrawn by explicit user instruction. Original controls remain unchanged and diagnostic; H2 is trained with the original calibrated objective, padding and crop.

| Group | Added updates | Dev50 | Dev100 | Numerical screen |
|---|---:|---:|---:|---|
| H0_control | 100 | 96/96 | 96/96 | NOT_QUALIFIED |
| H1_latent_self | 100 | 96/96 | 96/96 | NOT_QUALIFIED |
| H2_latent_equiv | 100 | 96/96 | 96/96 | NOT_QUALIFIED |

Mechanical screen compares every QP mean DISTS, LPIPS-Alex, LPIPS-VGG and FloLPIPS to≤1.005×B300 and requires meanQP P→P Sobel ratio S<1. Eligible50/100 checkpoints are selected by lowest S; differences below0.0005 prefer earlier. NOT_QUALIFIED never substitutes step0. This is a numerical screen, not a claim about human-visible flicker.

Aggregation: first64-frame video metrics, then equal video weights within dataset/split/QP. I→P/P→P temporal metrics are separate. Candidate RGB never enters the independent Original codec reference trajectory. Rates use real RANS bytes including actual I frame and stream headers. Byte counts, bpp, stream SHA and decoded codec-feature traces are paired checks.

Final test status: NOT_RUN_NO_QUALIFIED_CANDIDATE. No qualification means no new final R-D points. Existing32new-holdout/6historical ULong,5HEVC-B,2historical HoneyBee/Jockey UVG and30historical MCL-JCV remain distinct test plots when measured. No cross-dataset pooled curve, extrapolation or three-point BD-rate.

Video materials retain full-resolution single-model FFV1 and unscaled ROI, with labels, source frames, fps and SHA. Anonymous label mapping is separate. person_dog is recovered from audited original RGB frames; its MP4 is identification material only, historical exposure is retained, and it is excluded from training and model selection. No subjective experiment is claimed.

Raw development values: development_metrics.csv; paired baseline differences: development_paired_by_video_qp.csv; same-step H1−H0 and H2−H1 where available: development_matched_steps.csv. H2 started later after the user withdrew the assistant-added cutoff; its matched-step rows are populated from actual completed evaluations. Edge/local energy values are diagnostics and not quality-improvement guarantees.

Measured equal-video aggregates:

| Phase | Dataset | Model | Step | QP | bpp | PSNR | Alex | VGG | DISTS | FloLPIPS | P→P Sobel |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| development | ULong_development_32 | B300 | 0 | 0 | 0.0019158333 | 24.107245 | 0.18625056 | 0.32076811 | 0.10852167 | 0.18461602 | 0.0067526832 |
| development | ULong_development_32 | B300 | 0 | 5 | 0.0069668447 | 26.458508 | 0.11614086 | 0.22933891 | 0.071960529 | 0.11203413 | 0.0061206268 |
| development | ULong_development_32 | B300 | 0 | 9 | 0.015992852 | 27.357931 | 0.095882007 | 0.19868322 | 0.060900029 | 0.091131674 | 0.005995354 |
| development | ULong_development_32 | H0_control | 50 | 0 | 0.0019158333 | 24.077793 | 0.18684653 | 0.32156017 | 0.10949849 | 0.18512007 | 0.0068069401 |
| development | ULong_development_32 | H0_control | 50 | 5 | 0.0069668447 | 26.434776 | 0.11632301 | 0.2299805 | 0.072608086 | 0.11224906 | 0.0061693352 |
| development | ULong_development_32 | H0_control | 50 | 9 | 0.015992852 | 27.339126 | 0.095920255 | 0.19938781 | 0.061554744 | 0.091177817 | 0.0060411885 |
| development | ULong_development_32 | H0_control | 100 | 0 | 0.0019158333 | 24.059548 | 0.18733441 | 0.32183513 | 0.10998915 | 0.18543192 | 0.0066461692 |
| development | ULong_development_32 | H0_control | 100 | 5 | 0.0069668447 | 26.404532 | 0.1170784 | 0.229685 | 0.072812829 | 0.11291333 | 0.006027239 |
| development | ULong_development_32 | H0_control | 100 | 9 | 0.015992852 | 27.296094 | 0.096818423 | 0.19898128 | 0.061755903 | 0.092011539 | 0.0059053958 |
| development | ULong_development_32 | H1_latent_self | 50 | 0 | 0.0019158333 | 24.030214 | 0.18722065 | 0.32272809 | 0.10998269 | 0.18554076 | 0.0068134115 |
| development | ULong_development_32 | H1_latent_self | 50 | 5 | 0.0069668447 | 26.408307 | 0.1164553 | 0.23051031 | 0.072368117 | 0.11236237 | 0.0061590015 |
| development | ULong_development_32 | H1_latent_self | 50 | 9 | 0.015992852 | 27.325076 | 0.096022068 | 0.19967441 | 0.061242099 | 0.091235935 | 0.0060281254 |
| development | ULong_development_32 | H1_latent_self | 100 | 0 | 0.0019158333 | 23.94283 | 0.18799757 | 0.32326051 | 0.11072063 | 0.18623082 | 0.0066811521 |
| development | ULong_development_32 | H1_latent_self | 100 | 5 | 0.0069668447 | 26.301172 | 0.11749182 | 0.23045064 | 0.072689765 | 0.11326228 | 0.0060337411 |
| development | ULong_development_32 | H1_latent_self | 100 | 9 | 0.015992852 | 27.200756 | 0.097165237 | 0.19945519 | 0.061523378 | 0.092260423 | 0.0059064195 |
| development | ULong_development_32 | H2_latent_equiv | 50 | 0 | 0.0019158333 | 24.009165 | 0.18730648 | 0.32253534 | 0.10999077 | 0.18569145 | 0.0067943398 |
| development | ULong_development_32 | H2_latent_equiv | 50 | 5 | 0.0069668447 | 26.36794 | 0.11659052 | 0.23050187 | 0.072422911 | 0.11249453 | 0.0061440647 |
| development | ULong_development_32 | H2_latent_equiv | 50 | 9 | 0.015992852 | 27.277033 | 0.09615316 | 0.19972102 | 0.061313638 | 0.091378762 | 0.0060140518 |
| development | ULong_development_32 | H2_latent_equiv | 100 | 0 | 0.0019158333 | 23.905126 | 0.18817229 | 0.32320208 | 0.11081239 | 0.18641606 | 0.0066551113 |
| development | ULong_development_32 | H2_latent_equiv | 100 | 5 | 0.0069668447 | 26.231208 | 0.11772549 | 0.23048258 | 0.072795268 | 0.11349054 | 0.0060127968 |
| development | ULong_development_32 | H2_latent_equiv | 100 | 9 | 0.015992852 | 27.117206 | 0.097409881 | 0.19953194 | 0.061652034 | 0.092504614 | 0.0058863846 |

This report records execution, coverage and numbers only. Integrity/publication states are in final_integrity.json and publication_status.json.
