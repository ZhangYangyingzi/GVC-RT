# V6.18 numeric artifacts

Protocol: 64 causal frames; external QP 0, 4, 9; diagnostic same-external-QP comparison; FP16 codec / FP32 wrapper; force_zero_thres 0.12.

Frame windows: 1–8, 9–16, 17–32, 33–48, 49–64. Population standard deviation (ddof=0). FloLPIPS targets transitions; frame1 is undefined and excluded from window means.

Drift: late_49_64 − early_1_8. Excess drift: method drift − original drift. OLS slopes: frames2–64. Dataset aggregates: equal sequence weights.

| Dataset | Sequences | Methods | QPs | Points | Frames |
|---|---:|---:|---:|---:|---:|
| ulong | 8 | 5 | 3 | 120 | 7680 |
| uvg_holdout | 2 | 5 | 3 | 30 | 1920 |
| uvg_validation | 1 | 5 | 3 | 15 | 960 |
| hevc_b | 5 | 5 | 3 | 75 | 4800 |

| File | Rows |
|---|---:|
| results/window_metrics.csv | 3600 |
| results/temporal_drift_per_sequence.csv | 720 |
| results/temporal_drift_dataset_summary.csv | 180 |
| results/excess_drift_vs_original.csv | 720 |
| results/excess_drift_dataset_summary.csv | 180 |
| results/per_frame_delta_vs_original.csv | 46080 |
| results/frame_index_slopes.csv | 720 |
| results/frame_index_slopes_dataset_summary.csv | 180 |
| evaluation/per_frame_metrics.csv | 15360 |
| results/frame_curve_dataset_means.csv | 11520 |

Plots: `plots/<dataset>_qp<QP>_<metric>_vs_frame.png` and `plots/<dataset>_qp<QP>_<metric>_delta_vs_original.png`.
