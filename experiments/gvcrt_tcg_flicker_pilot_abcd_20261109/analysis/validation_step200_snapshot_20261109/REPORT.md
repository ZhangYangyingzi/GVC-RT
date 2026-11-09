# Step200 development snapshot

Snapshot local time: 2026-11-09 17:56:15 +0800.

Four branches each have complete36-point results:12 existing ULong development videos x external QP0/5/9. Reference: fixed A2000 s0.5. Equal video weights within each QP. No cross-QP mean in the table.

A: TCG+LoRA, frozen Bridge, no explicit history alignment. B: A plus Bridge mlp[3] training. C: A plus fixed causal original-reconstruction RAFT history alignment. D: C plus Bridge mlp[3] training.

|Group|QP|LPIPS change %|DISTS change %|FloLPIPS change %|Sobel temporal error mean change %|
|---|---:|---:|---:|---:|---:|
|A|0|-0.30|-0.33|+0.01|-8.92|
|A|5|-0.06|-0.80|+0.01|-7.28|
|A|9|-0.08|-1.07|-0.19|-6.76|
|B|0|-0.47|-0.59|-0.18|-9.23|
|B|5|-0.24|-1.11|-0.24|-7.71|
|B|9|-0.26|-1.45|-0.42|-7.20|
|C|0|+0.22|-0.44|+2.08|-8.38|
|C|5|+0.37|-0.66|+0.99|-7.19|
|C|9|+0.35|-0.94|+0.78|-6.76|
|D|0|+0.05|-0.67|+1.86|-8.44|
|D|5|+0.18|-0.96|+0.75|-7.49|
|D|9|+0.17|-1.25|+0.56|-7.11|

All four error metrics are lower-is-better. Relative changes are(candidate mean/reference mean-1)*100. The per-QP Sobel mean changes above differ from the preregistered median-of-video-ratios screening statistic.

Screening at200: A=ONE_VALIDATION_PASS, B=NUMERIC_SCREEN_PASS, C=ONE_VALIDATION_PASS, D=ONE_VALIDATION_PASS.

See summary_by_group_qp.csv for absolute metrics, PSNR dB changes, actual bpp/kbps, both original PWC warping metrics, and point_metrics.csv for sequence/QP detail.

This is an intermediate fixed-budget comparison. The complete300-step pilot, videos, final integrity and final result publication are pending.
