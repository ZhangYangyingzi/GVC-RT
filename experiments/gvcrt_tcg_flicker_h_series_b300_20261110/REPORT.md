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
| H0_control | 100 | 93/96 | 93/96 | pending |
| H1_latent_self | 100 | 93/96 | 93/96 | pending |
| H2_latent_equiv | 100 | 82/96 | 72/96 | pending |

Mechanical screen compares every QP mean DISTS, LPIPS-Alex, LPIPS-VGG and FloLPIPS to≤1.005×B300 and requires meanQP P→P Sobel ratio S<1. Eligible50/100 checkpoints are selected by lowest S; differences below0.0005 prefer earlier. NOT_QUALIFIED never substitutes step0. This is a numerical screen, not a claim about human-visible flicker.

Aggregation: first64-frame video metrics, then equal video weights within dataset/split/QP. I→P/P→P temporal metrics are separate. Candidate RGB never enters the independent Original codec reference trajectory. Rates use real RANS bytes including actual I frame and stream headers. Byte counts, bpp, stream SHA and decoded codec-feature traces are paired checks.

Final test status: pending conditional numerical screen. No qualification means no new final R-D points. Existing32new-holdout/6historical ULong,5HEVC-B,2historical HoneyBee/Jockey UVG and30historical MCL-JCV remain distinct test plots when measured. No cross-dataset pooled curve, extrapolation or three-point BD-rate.

Video materials retain full-resolution single-model FFV1 and unscaled ROI, with labels, source frames, fps and SHA. Anonymous label mapping is separate. person_dog is recovered from audited original RGB frames; its MP4 is identification material only, historical exposure is retained, and it is excluded from training and model selection. No subjective experiment is claimed.

Raw development values: development_metrics.csv; paired baseline differences: development_paired_by_video_qp.csv; same-step H1−H0 and H2−H1 where available: development_matched_steps.csv. H2 started later after the user withdrew the assistant-added cutoff; its matched-step rows are populated from actual completed evaluations. Edge/local energy values are diagnostics and not quality-improvement guarantees.

Measured equal-video aggregates:

| Phase | Dataset | Model | Step | QP | bpp | PSNR | Alex | VGG | DISTS | FloLPIPS | P→P Sobel |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| development | ULong_development_32 | B300 | 0 | 0 | 0.0019226089 | 24.04944 | 0.18683833 | 0.32116994 | 0.10809914 | 0.1862228 | 0.0067969738 |
| development | ULong_development_32 | B300 | 0 | 5 | 0.0070036293 | 26.393595 | 0.11674155 | 0.22994225 | 0.072073639 | 0.11287865 | 0.0061562798 |
| development | ULong_development_32 | B300 | 0 | 9 | 0.016081268 | 27.292689 | 0.096457212 | 0.19929134 | 0.061028936 | 0.09178238 | 0.0060294138 |
| development | ULong_development_32 | H0_control | 50 | 0 | 0.0019226089 | 24.020114 | 0.18741175 | 0.32195961 | 0.10904045 | 0.18668137 | 0.0068497041 |
| development | ULong_development_32 | H0_control | 50 | 5 | 0.0070036293 | 26.369365 | 0.11691285 | 0.23057783 | 0.072717423 | 0.11306967 | 0.0062035445 |
| development | ULong_development_32 | H0_control | 50 | 9 | 0.016081268 | 27.273014 | 0.096480665 | 0.19998428 | 0.061668334 | 0.091807084 | 0.0060738569 |
| development | ULong_development_32 | H0_control | 100 | 0 | 0.0019226089 | 24.002128 | 0.18791244 | 0.32222206 | 0.10948998 | 0.18701324 | 0.0066892204 |
| development | ULong_development_32 | H0_control | 100 | 5 | 0.0070036293 | 26.340398 | 0.11768903 | 0.23029057 | 0.07290946 | 0.11376154 | 0.0060618805 |
| development | ULong_development_32 | H0_control | 100 | 9 | 0.016081268 | 27.231952 | 0.097397681 | 0.19959353 | 0.061867177 | 0.092664609 | 0.0059384901 |
| development | ULong_development_32 | H1_latent_self | 50 | 0 | 0.0019226089 | 23.972858 | 0.18778745 | 0.32313333 | 0.10952609 | 0.1871084 | 0.0068562562 |
| development | ULong_development_32 | H1_latent_self | 50 | 5 | 0.0070036293 | 26.342332 | 0.1170547 | 0.23112165 | 0.072489104 | 0.11319023 | 0.0061932203 |
| development | ULong_development_32 | H1_latent_self | 50 | 9 | 0.016081268 | 27.257858 | 0.096591757 | 0.20027965 | 0.061363054 | 0.091872621 | 0.0060608223 |
| development | ULong_development_32 | H1_latent_self | 100 | 0 | 0.0019226089 | 23.886104 | 0.18858277 | 0.32365622 | 0.11023021 | 0.18783385 | 0.006724751 |
| development | ULong_development_32 | H1_latent_self | 100 | 5 | 0.0070036293 | 26.236681 | 0.11811865 | 0.23107532 | 0.072809763 | 0.11412498 | 0.0060687325 |
| development | ULong_development_32 | H1_latent_self | 100 | 9 | 0.016081268 | 27.135563 | 0.097761693 | 0.20008183 | 0.061649573 | 0.092926773 | 0.0059399224 |
| development | ULong_development_32 | H2_latent_equiv | 50 | 0 | 0.0019302537 | 23.995083 | 0.18042191 | 0.31709901 | 0.10958199 | 0.17805186 | 0.0068615398 |
| development | ULong_development_32 | H2_latent_equiv | 50 | 5 | 0.0070262489 | 26.284349 | 0.11253502 | 0.22746194 | 0.07259468 | 0.10755312 | 0.0061886636 |
| development | ULong_development_32 | H2_latent_equiv | 50 | 9 | 0.016125861 | 27.023617 | 0.096688099 | 0.20044792 | 0.062531422 | 0.091237006 | 0.0062856585 |
| development | ULong_development_32 | H2_latent_equiv | 100 | 0 | 0.0019881386 | 23.941674 | 0.18486186 | 0.32129377 | 0.10880466 | 0.1822307 | 0.0068970497 |
| development | ULong_development_32 | H2_latent_equiv | 100 | 5 | 0.0070687111 | 26.306188 | 0.11270257 | 0.2257826 | 0.07028842 | 0.10678311 | 0.0060353036 |
| development | ULong_development_32 | H2_latent_equiv | 100 | 9 | 0.015973095 | 26.961039 | 0.096642409 | 0.19728683 | 0.061520691 | 0.090415901 | 0.0059807377 |

This report records execution, coverage and numbers only. Integrity/publication states are in final_integrity.json and publication_status.json.
