# Teacher temporal noLoRA A/B execution record

Status: RUNNING.

Both groups inherit the B300 Bridge and TCG, which were previously trained with LoRA. Five LoRA wrappers were physically removed without merging their increments. The decoder backbone is frozen. G0 and G1 differ only by the latent temporal objective; Teacher is training-only.

Common step0 SHA256: `99fc7576e3afa8e6e8bfe8b612e061679283150aab090a823fe2d16ad9109279`.
B300 SHA256: `7465b57f20bfe2a8d6064a87155cf45c31a488cdeaa68111a228c28f6335ea85`.
Calibration: lambda_out=0.7921341234204811; lambda_latent=2.379767384936951.

Training: FP32 Bridge/TCG and frozen decoder computation, fresh AdamW; global batch four, two ranks each sequentially processing two windows. GPU0/1: G0; GPU2/3: G1. Independent Original codec trajectory and real RANS rate, with no candidate image fed back to codec. Deployment follows the uniform inherited FP16 backbone/Bridge and FP32 TCG policy.

Training LPIPS is VGG only. Evaluation records Alex and VGG separately, RGB [0,1], LPIPS normalize=True, v0.1 weights. Full numerical definitions and weight identities are in protocol.json, quality.py and per-point receipts. Temporal missing regions are excluded using valid denominators. No margin or extra detail loss is implemented.

Teacher is the channel-compatible audited pretrain262144 EMA; its identity is not asserted to be the exact historical author checkpoint. Channel compatibility is not a claim of temporal stability.

| Group | Real added updates | Stop reason | Selected step | Qualification | S |
|---|---:|---|---:|---|---:|
| G0_teacher_spatial | 18 | running | pending | pending | pending |
| G1_teacher_temporal | 18 | running | pending | pending | pending |

Selection uses development quality only. Missing metrics and engineering faults do not count as non-improvements. All same-step G1−G0 measurements are in development_G1_minus_G0_common_steps.csv. Final checkpoint identities are frozen in selection.json before final testing.

Final coverage:

| Dataset / split | Model | Completed | Expected |
|---|---|---:|---:|
| HEVC_B_5 | Original GVC-RT | 0 | 15 |
| HEVC_B_5 | B300 original reference (LoRA s=0.5) | 0 | 15 |
| HEVC_B_5 | B300 noLoRA step0 | 0 | 15 |
| HEVC_B_5 | G0 teacher spatial selected | 0 | 15 |
| HEVC_B_5 | G1 teacher temporal selected | 0 | 15 |
| ULong_historical_6 | Original GVC-RT | 0 | 18 |
| ULong_historical_6 | B300 original reference (LoRA s=0.5) | 0 | 18 |
| ULong_historical_6 | B300 noLoRA step0 | 0 | 18 |
| ULong_historical_6 | G0 teacher spatial selected | 0 | 18 |
| ULong_historical_6 | G1 teacher temporal selected | 0 | 18 |
| UVG_historical_HoneyBee_Jockey_2 | Original GVC-RT | 0 | 6 |
| UVG_historical_HoneyBee_Jockey_2 | B300 original reference (LoRA s=0.5) | 0 | 6 |
| UVG_historical_HoneyBee_Jockey_2 | B300 noLoRA step0 | 0 | 6 |
| UVG_historical_HoneyBee_Jockey_2 | G0 teacher spatial selected | 0 | 6 |
| UVG_historical_HoneyBee_Jockey_2 | G1 teacher temporal selected | 0 | 6 |
| ULong_new_holdout_32 | Original GVC-RT | 0 | 96 |
| ULong_new_holdout_32 | B300 original reference (LoRA s=0.5) | 0 | 96 |
| ULong_new_holdout_32 | B300 noLoRA step0 | 0 | 96 |
| ULong_new_holdout_32 | G0 teacher spatial selected | 0 | 96 |
| ULong_new_holdout_32 | G1 teacher temporal selected | 0 | 96 |
| MCL_JCV_30 | Original GVC-RT | 0 | 90 |
| MCL_JCV_30 | B300 original reference (LoRA s=0.5) | 0 | 90 |
| MCL_JCV_30 | B300 noLoRA step0 | 0 | 90 |
| MCL_JCV_30 | G0 teacher spatial selected | 0 | 90 |
| MCL_JCV_30 | G1 teacher temporal selected | 0 | 90 |

Aggregation: arithmetic mean across video-level measurements within each dataset/split/QP; each video has equal weight. Each per-video PSNR uses pixel SSE across 64 frames; spatial metrics use all 64 frames. I→P and P→P temporal measurements are separate. The main selection temporal score uses P→P Sobel MCResidual. Figures connect three actual RANS bpp points only. Dataset splits are plotted separately; UVG contains only the two historical HoneyBee/Jockey sequences. ULong historical six and new holdout 32 are separate. MCL-JCV is a historically used test collection. No BD-rate is reported.

Original source: independently restored GVC-RT_I.pt and GVC-RT_P.pt, commit b2a0895c0cf31cf2be3ab1c2571ca07c05866c03. It loads no adaptation weights. Per-point byte counts, stream hashes and Original codec feature traces are checked; LPIPS-VGG and RAFT residual metrics are newly computed for every state. Compatible older scalar components are reused only after exact reconstructed RGB and metric-weight checks.

See metrics.csv for raw values; rd_points_by_dataset.csv for equal-video aggregates; paired_deltas_by_dataset_qp.csv and paired_deltas_by_video.csv for paired differences. Local lossless media and SHA256 are recorded in video_manifest.json. Preview sizes are never used as neural rates. Videos are records, with no claim of a subjective blind test.

Integrity and publication: final_integrity.json and publication_status.json. This report records execution and measurements only; it draws no method or causal conclusion.
