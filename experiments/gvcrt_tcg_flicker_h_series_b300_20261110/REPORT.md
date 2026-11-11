# Corrected H-series B300 execution record

Status: RUNNING. See status.json for current stage.

All models restore original I/P → complete pilot generator/Bridge/persistent buffers → B+1500 TCG → A2000 LoRA → B30052-key overlay last. LoRA is retained without merging at s=0.5. Full Bridge801124, TCG137760, LoRA59392 are trainable; original decoder backbone, codec, I model, entropy and metric networks are frozen. H0 is a continuation control whose difference from B300 includes widening Bridge training and further updates; this report assigns no single-factor causal effect.

Base objective: LPIPS-Alex +0.5 DISTS +0.1383589248954108 RGB-L1 +0.05 RAFT-motion +2.603297563918803×(RGB MCResidual +0.5 Sobel MCResidual). Training LPIPS is Alex, evaluation records Alex and VGG separately ([0,1] RGB, normalize=True, v0.1 weights).

H1 adds hard-mask-area-normalized confidence-weighted latent differences with shared differentiable channel RMS, tau0.02. Flow is estimated by frozen RAFT on GT RGB, not a known true motion field. H2 additionally proposes four stateless paired translation losses/window; both sides disable TCG and cache updates; target is no_grad. No edge-floor, Teacher, Router, GAN, SFT, history alignment or new upsampler.

Fresh AdamW: Bridge2e-6, TCG1e-5, LoRA5e-5; betas(.9,.999), eps1e-8, weight_decay1e-4, globalclip1. Exactly four logical windows/update,64 frames/window, current-weight warmup1–48, supervised49–63 segmented4/4/4/3. Maximum100 updates; no automatic extension. GPU0 H0, GPU1 H1, GPU2 H2 if mechanically allowed, GPU3 serialized bounded cache/evaluation.

Calibration uses10 frozen training windows, no optimizer updates or development/test choice. lambda_lat=0.2953169862992426; lambda_equiv=0.10179495298276185.

H2 mechanical gate: DIAGNOSTIC_ONLY_USER_DIRECTED_TRAINING; median16px/8px error ratio atcrop64=0.16182493976184825. The assistant-added0.1 launch cutoff was withdrawn by explicit user instruction. Original controls remain unchanged and diagnostic; H2 is trained with the original calibrated objective, padding and crop.

| Group | Added updates | Dev50 | Dev100 | Numerical screen |
|---|---:|---:|---:|---|
| H0_control | 100 | 50/96 | 50/96 | pending |
| H1_latent_self | 100 | 50/96 | 50/96 | pending |
| H2_latent_equiv | 100 | 32/96 | 22/96 | pending |

Mechanical screen compares every QP mean DISTS, LPIPS-Alex, LPIPS-VGG and FloLPIPS to≤1.005×B300 and requires meanQP P→P Sobel ratio S<1. Eligible50/100 checkpoints are selected by lowest S; differences below0.0005 prefer earlier. NOT_QUALIFIED never substitutes step0. This is a numerical screen, not a claim about human-visible flicker.

Aggregation: first64-frame video metrics, then equal video weights within dataset/split/QP. I→P/P→P temporal metrics are separate. Candidate RGB never enters the independent Original codec reference trajectory. Rates use real RANS bytes including actual I frame and stream headers. Byte counts, bpp, stream SHA and decoded codec-feature traces are paired checks.

Final test status: pending conditional numerical screen. No qualification means no new final R-D points. Existing32new-holdout/6historical ULong,5HEVC-B,2historical HoneyBee/Jockey UVG and30historical MCL-JCV remain distinct test plots when measured. No cross-dataset pooled curve, extrapolation or three-point BD-rate.

Video materials retain full-resolution single-model FFV1 and unscaled ROI, with labels, source frames, fps and SHA. Anonymous label mapping is separate. person_dog is recovered from audited original RGB frames; its MP4 is identification material only, historical exposure is retained, and it is excluded from training and model selection. No subjective experiment is claimed.

Raw development values: development_metrics.csv; paired baseline differences: development_paired_by_video_qp.csv; same-step H1−H0 and H2−H1 where available: development_matched_steps.csv. H2 started later after the user withdrew the assistant-added cutoff; its matched-step rows are populated from actual completed evaluations. Edge/local energy values are diagnostics and not quality-improvement guarantees.

Measured equal-video aggregates:

| Phase | Dataset | Model | Step | QP | bpp | PSNR | Alex | VGG | DISTS | FloLPIPS | P→P Sobel |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| development | ULong_development_32 | B300 | 0 | 0 | 0.0018748475 | 23.94677 | 0.19812257 | 0.32934209 | 0.10866159 | 0.19990953 | 0.00692275 |
| development | ULong_development_32 | B300 | 0 | 5 | 0.0068020521 | 26.352704 | 0.12479232 | 0.23660998 | 0.072467612 | 0.12251218 | 0.0062908748 |
| development | ULong_development_32 | B300 | 0 | 9 | 0.015599934 | 27.547183 | 0.097712438 | 0.2002339 | 0.05916972 | 0.093138896 | 0.0057845199 |
| development | ULong_development_32 | H0_control | 50 | 0 | 0.0018748475 | 23.91737 | 0.19871762 | 0.33002499 | 0.1097038 | 0.20030598 | 0.0069892878 |
| development | ULong_development_32 | H0_control | 50 | 5 | 0.0068020521 | 26.324876 | 0.12491805 | 0.23719026 | 0.073181997 | 0.12259702 | 0.0063477913 |
| development | ULong_development_32 | H0_control | 50 | 9 | 0.015599934 | 27.521052 | 0.097727023 | 0.20095859 | 0.059882156 | 0.093137537 | 0.0058342419 |
| development | ULong_development_32 | H0_control | 100 | 0 | 0.0018748475 | 23.895822 | 0.19935265 | 0.33024003 | 0.11010074 | 0.2007679 | 0.0068114445 |
| development | ULong_development_32 | H0_control | 100 | 5 | 0.0068020521 | 26.290231 | 0.12592394 | 0.23675403 | 0.073402256 | 0.12346724 | 0.0061941572 |
| development | ULong_development_32 | H0_control | 100 | 9 | 0.015599934 | 27.471298 | 0.098777304 | 0.20039041 | 0.06004312 | 0.094055403 | 0.0056987067 |
| development | ULong_development_32 | H1_latent_self | 50 | 0 | 0.0018748475 | 23.871177 | 0.19902109 | 0.33109139 | 0.11014902 | 0.20053297 | 0.0069934322 |
| development | ULong_development_32 | H1_latent_self | 50 | 5 | 0.0068020521 | 26.297529 | 0.12502691 | 0.2376987 | 0.072886812 | 0.12266988 | 0.0063366163 |
| development | ULong_development_32 | H1_latent_self | 50 | 9 | 0.015599934 | 27.504189 | 0.097813048 | 0.20127804 | 0.059548669 | 0.093210414 | 0.0058217639 |
| development | ULong_development_32 | H1_latent_self | 100 | 0 | 0.0018748475 | 23.783458 | 0.19996272 | 0.33157593 | 0.11089381 | 0.20143553 | 0.0068401369 |
| development | ULong_development_32 | H1_latent_self | 100 | 5 | 0.0068020521 | 26.189069 | 0.12633213 | 0.23750944 | 0.073289501 | 0.12385716 | 0.0061968627 |
| development | ULong_development_32 | H1_latent_self | 100 | 9 | 0.015599934 | 27.371816 | 0.099137829 | 0.20091951 | 0.059841166 | 0.094396099 | 0.0056957397 |
| development | ULong_development_32 | H2_latent_equiv | 50 | 0 | 0.0015253939 | 23.985867 | 0.18304046 | 0.32081746 | 0.11406405 | 0.1801208 | 0.006758084 |
| development | ULong_development_32 | H2_latent_equiv | 50 | 5 | 0.0057169405 | 26.144765 | 0.11709069 | 0.23402221 | 0.076066199 | 0.11201449 | 0.0061557256 |
| development | ULong_development_32 | H2_latent_equiv | 50 | 9 | 0.013700889 | 27.062761 | 0.10114037 | 0.21069529 | 0.064047206 | 0.093377572 | 0.0065168666 |
| development | ULong_development_32 | H2_latent_equiv | 100 | 0 | 0.0015472261 | 24.027586 | 0.19581886 | 0.33358861 | 0.11192433 | 0.19219283 | 0.0070460847 |
| development | ULong_development_32 | H2_latent_equiv | 100 | 5 | 0.005108283 | 26.446976 | 0.11808974 | 0.23186898 | 0.069824005 | 0.11023773 | 0.0058847867 |
| development | ULong_development_32 | H2_latent_equiv | 100 | 9 | 0.012116023 | 27.231943 | 0.099886029 | 0.20367694 | 0.060248516 | 0.08869721 | 0.0058179899 |

This report records execution, coverage and numbers only. Integrity/publication states are in final_integrity.json and publication_status.json.
