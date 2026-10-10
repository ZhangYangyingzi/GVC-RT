# Protocol

No training. Seed 20261010. 3 methods, 16 sequences, QP 0–9. 128 canonical frames per sequence. Paired sequence-stratified sampling. FID: sample covariance; KID: unbiased cubic polynomial MMD. PCHIP on 100 uniform ln(bpp) points within the three-method common measured interval.

| Output | Count |
|---|---:|
| Real RANS points | 480 |
| V6.20 reproduction points | 120 |
| Bootstrap repeats per dataset/QP | 1000 |
| Sample-size repeats | 100 |

## Files

results/v620_fid_reproduction.csv
results/fid_sample_size_stability.csv
results/fid_sample_size_summary.csv
results/fid_bootstrap_delta.csv
results/fid_bootstrap_summary.csv
results/kid_per_qp.csv
results/kid_delta_vs_original.csv
results/kid_bootstrap_summary.csv
results/equal_rate_fid.csv
results/equal_rate_kid.csv
results/equal_rate_fid_bootstrap_summary.csv
results/equal_rate_kid_bootstrap_summary.csv
