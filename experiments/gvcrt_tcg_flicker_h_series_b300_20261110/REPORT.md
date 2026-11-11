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
| H0_control | 100 | 53/96 | 53/96 | pending |
| H1_latent_self | 100 | 53/96 | 53/96 | pending |
| H2_latent_equiv | 100 | 32/96 | 22/96 | pending |

Mechanical screen compares every QP mean DISTS, LPIPS-Alex, LPIPS-VGG and FloLPIPS to≤1.005×B300 and requires meanQP P→P Sobel ratio S<1. Eligible50/100 checkpoints are selected by lowest S; differences below0.0005 prefer earlier. NOT_QUALIFIED never substitutes step0. This is a numerical screen, not a claim about human-visible flicker.

Aggregation: first64-frame video metrics, then equal video weights within dataset/split/QP. I→P/P→P temporal metrics are separate. Candidate RGB never enters the independent Original codec reference trajectory. Rates use real RANS bytes including actual I frame and stream headers. Byte counts, bpp, stream SHA and decoded codec-feature traces are paired checks.

Final test status: pending conditional numerical screen. No qualification means no new final R-D points. Existing32new-holdout/6historical ULong,5HEVC-B,2historical HoneyBee/Jockey UVG and30historical MCL-JCV remain distinct test plots when measured. No cross-dataset pooled curve, extrapolation or three-point BD-rate.

Video materials retain full-resolution single-model FFV1 and unscaled ROI, with labels, source frames, fps and SHA. Anonymous label mapping is separate. person_dog is recovered from audited original RGB frames; its MP4 is identification material only, historical exposure is retained, and it is excluded from training and model selection. No subjective experiment is claimed.

Raw development values: development_metrics.csv; paired baseline differences: development_paired_by_video_qp.csv; same-step H1−H0 and H2−H1 where available: development_matched_steps.csv. H2 started later after the user withdrew the assistant-added cutoff; its matched-step rows are populated from actual completed evaluations. Edge/local energy values are diagnostics and not quality-improvement guarantees.

Measured equal-video aggregates:

| Phase | Dataset | Model | Step | QP | bpp | PSNR | Alex | VGG | DISTS | FloLPIPS | P→P Sobel |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| development | ULong_development_32 | B300 | 0 | 0 | 0.0019398429 | 24.139576 | 0.19417438 | 0.32557346 | 0.10637932 | 0.19576156 | 0.0068185055 |
| development | ULong_development_32 | B300 | 0 | 5 | 0.006922445 | 26.589023 | 0.12163719 | 0.2330681 | 0.07065662 | 0.11930427 | 0.006183921 |
| development | ULong_development_32 | B300 | 0 | 9 | 0.015870282 | 27.801319 | 0.094944902 | 0.19664695 | 0.057505183 | 0.090492792 | 0.005687279 |
| development | ULong_development_32 | H0_control | 50 | 0 | 0.0019398429 | 24.110248 | 0.19477736 | 0.32626886 | 0.10741225 | 0.19617385 | 0.0068828643 |
| development | ULong_development_32 | H0_control | 50 | 5 | 0.006922445 | 26.561568 | 0.12178925 | 0.23366826 | 0.071359057 | 0.1194216 | 0.0062390193 |
| development | ULong_development_32 | H0_control | 50 | 9 | 0.015870282 | 27.776448 | 0.094980194 | 0.19737337 | 0.058189317 | 0.090518062 | 0.0057353445 |
| development | ULong_development_32 | H0_control | 100 | 0 | 0.0019398429 | 24.090124 | 0.19538181 | 0.32646972 | 0.10778809 | 0.19661587 | 0.0067114405 |
| development | ULong_development_32 | H0_control | 100 | 5 | 0.006922445 | 26.527588 | 0.12275361 | 0.23323979 | 0.071579448 | 0.12026274 | 0.0060905699 |
| development | ULong_development_32 | H0_control | 100 | 9 | 0.015870282 | 27.72519 | 0.095997998 | 0.19682849 | 0.058367888 | 0.091412888 | 0.0056043706 |
| development | ULong_development_32 | H1_latent_self | 50 | 0 | 0.0019398429 | 24.063803 | 0.19507115 | 0.32731013 | 0.10784578 | 0.1964012 | 0.0068865554 |
| development | ULong_development_32 | H1_latent_self | 50 | 5 | 0.006922445 | 26.534535 | 0.12188796 | 0.23415182 | 0.071065055 | 0.11948555 | 0.0062275878 |
| development | ULong_development_32 | H1_latent_self | 50 | 9 | 0.015870282 | 27.761591 | 0.095052779 | 0.19767022 | 0.057860943 | 0.090575673 | 0.0057226064 |
| development | ULong_development_32 | H1_latent_self | 100 | 0 | 0.0019398429 | 23.976323 | 0.19597465 | 0.32777862 | 0.10856756 | 0.19727511 | 0.0067381863 |
| development | ULong_development_32 | H1_latent_self | 100 | 5 | 0.006922445 | 26.425212 | 0.12313928 | 0.23395942 | 0.071465024 | 0.12063057 | 0.0060919672 |
| development | ULong_development_32 | H1_latent_self | 100 | 9 | 0.015870282 | 27.626035 | 0.09633099 | 0.19732461 | 0.058169812 | 0.091724259 | 0.0056002963 |
| development | ULong_development_32 | H2_latent_equiv | 50 | 0 | 0.0015253939 | 23.985867 | 0.18304046 | 0.32081746 | 0.11406405 | 0.1801208 | 0.006758084 |
| development | ULong_development_32 | H2_latent_equiv | 50 | 5 | 0.0057169405 | 26.144765 | 0.11709069 | 0.23402221 | 0.076066199 | 0.11201449 | 0.0061557256 |
| development | ULong_development_32 | H2_latent_equiv | 50 | 9 | 0.013700889 | 27.062761 | 0.10114037 | 0.21069529 | 0.064047206 | 0.093377572 | 0.0065168666 |
| development | ULong_development_32 | H2_latent_equiv | 100 | 0 | 0.0015472261 | 24.027586 | 0.19581886 | 0.33358861 | 0.11192433 | 0.19219283 | 0.0070460847 |
| development | ULong_development_32 | H2_latent_equiv | 100 | 5 | 0.005108283 | 26.446976 | 0.11808974 | 0.23186898 | 0.069824005 | 0.11023773 | 0.0058847867 |
| development | ULong_development_32 | H2_latent_equiv | 100 | 9 | 0.012116023 | 27.231943 | 0.099886029 | 0.20367694 | 0.060248516 | 0.08869721 | 0.0058179899 |

This report records execution, coverage and numbers only. Integrity/publication states are in final_integrity.json and publication_status.json.
