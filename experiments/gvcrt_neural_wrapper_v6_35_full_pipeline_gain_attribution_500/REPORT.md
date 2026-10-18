# v6.35 results report

## Fixed answer and coverage

A_refLoss_noP_500 was not trained: the executable-source gate is `BLOCKED_MISSING_REFERENCE_OBJECTIVE` with zero updates. N/P were reused at their validated cumulative step 500. The O/N/P matrix is complete for the fixed 13-video, QP 0/5/9 manifest.

Completed logical RD points: 117/117 (75 verified reuse, 42 new encodes). Pooled FID points: 27/27.

## Model/source identities

N checkpoint SHA256: `ca2c66aa1d68877779c8646b66d02e4ba06e82fc4f3b3c92c11271fc7ffc19dc`.
P checkpoint SHA256: `eea01e8111799038521f357d970b21b1a9774db3a641a94ed6708477771c816e`.
Original public I/P checkpoint paths and hashes are listed in `source_hashes.json`. No model was trained in v6.35.

## Equal measured-rate tables

Values use the dataset-wide common measured-bpp interval across all included videos and O/N/P, a fixed 20-point uniform ln(bpp) grid, piecewise-linear interpolation without extrapolation, and equal video weighting. Relative change is 100×(method mean−reference mean)/reference mean; lower is better for LPIPS/DISTS/FloLPIPS. These percentages are metric changes, not BD-rate.

|Dataset|Method|Reference|Metric|Method mean|Reference mean|Delta|Relative %|Common bpp interval|
|---|---|---|---|---:|---:|---:|---:|---|
|ulong|P_oursLoss_withP_500|original|LPIPS|0.148347|0.158032|-0.00968461|-6.128|0.0032021–0.00881987|
|ulong|P_oursLoss_withP_500|original|DISTS|0.102397|0.10894|-0.00654225|-6.005|0.0032021–0.00881987|
|ulong|P_oursLoss_withP_500|original|FloLPIPS|0.153234|0.163074|-0.00983958|-6.034|0.0032021–0.00881987|
|ulong|N_oursLoss_noP_500|original|LPIPS|0.148149|0.158032|-0.00988241|-6.253|0.0032021–0.00881987|
|ulong|N_oursLoss_noP_500|original|DISTS|0.102007|0.10894|-0.00693246|-6.364|0.0032021–0.00881987|
|ulong|N_oursLoss_noP_500|original|FloLPIPS|0.152828|0.163074|-0.0102457|-6.283|0.0032021–0.00881987|
|ulong|P_oursLoss_withP_500|N_oursLoss_noP_500|LPIPS|0.148347|0.148149|+0.000197801|+0.134|0.0032021–0.00881987|
|ulong|P_oursLoss_withP_500|N_oursLoss_noP_500|DISTS|0.102397|0.102007|+0.000390211|+0.383|0.0032021–0.00881987|
|ulong|P_oursLoss_withP_500|N_oursLoss_noP_500|FloLPIPS|0.153234|0.152828|+0.000406155|+0.266|0.0032021–0.00881987|
|uvg_holdout|P_oursLoss_withP_500|original|LPIPS|0.174183|0.181559|-0.00737645|-4.063|0.00676408–0.00996046|
|uvg_holdout|P_oursLoss_withP_500|original|DISTS|0.0935629|0.104676|-0.011113|-10.617|0.00676408–0.00996046|
|uvg_holdout|P_oursLoss_withP_500|original|FloLPIPS|0.175799|0.184887|-0.00908791|-4.915|0.00676408–0.00996046|
|uvg_holdout|N_oursLoss_noP_500|original|LPIPS|0.173728|0.181559|-0.00783104|-4.313|0.00676408–0.00996046|
|uvg_holdout|N_oursLoss_noP_500|original|DISTS|0.0936698|0.104676|-0.0110062|-10.515|0.00676408–0.00996046|
|uvg_holdout|N_oursLoss_noP_500|original|FloLPIPS|0.176151|0.184887|-0.00873655|-4.725|0.00676408–0.00996046|
|uvg_holdout|P_oursLoss_withP_500|N_oursLoss_noP_500|LPIPS|0.174183|0.173728|+0.000454596|+0.262|0.00676408–0.00996046|
|uvg_holdout|P_oursLoss_withP_500|N_oursLoss_noP_500|DISTS|0.0935629|0.0936698|-0.000106841|-0.114|0.00676408–0.00996046|
|uvg_holdout|P_oursLoss_withP_500|N_oursLoss_noP_500|FloLPIPS|0.175799|0.176151|-0.000351363|-0.199|0.00676408–0.00996046|
|hevc_b|P_oursLoss_withP_500|original|LPIPS|0.18778|0.198662|-0.0108815|-5.477|0.0042387–0.0143732|
|hevc_b|P_oursLoss_withP_500|original|DISTS|0.088298|0.0948989|-0.00660092|-6.956|0.0042387–0.0143732|
|hevc_b|P_oursLoss_withP_500|original|FloLPIPS|0.181883|0.200702|-0.0188193|-9.377|0.0042387–0.0143732|
|hevc_b|N_oursLoss_noP_500|original|LPIPS|0.187292|0.198662|-0.0113694|-5.723|0.0042387–0.0143732|
|hevc_b|N_oursLoss_noP_500|original|DISTS|0.0876115|0.0948989|-0.00728745|-7.679|0.0042387–0.0143732|
|hevc_b|N_oursLoss_noP_500|original|FloLPIPS|0.182015|0.200702|-0.0186872|-9.311|0.0042387–0.0143732|
|hevc_b|P_oursLoss_withP_500|N_oursLoss_noP_500|LPIPS|0.18778|0.187292|+0.000487927|+0.261|0.0042387–0.0143732|
|hevc_b|P_oursLoss_withP_500|N_oursLoss_noP_500|DISTS|0.088298|0.0876115|+0.000686523|+0.784|0.0042387–0.0143732|
|hevc_b|P_oursLoss_withP_500|N_oursLoss_noP_500|FloLPIPS|0.181883|0.182015|-0.000132111|-0.073|0.0042387–0.0143732|

## Pooled FID

FID is computed over the complete dataset frame pool using the fixed Inception2048 features, float64 and ddof=1. Single-video FID is not averaged. The equal-rate FID trend uses the dataset mean actual bpp and a linear diagnostic interpolation over the three measured QP anchors. D_mu and D_cov are in `results/pooled_fid.csv`; their sum matches the stored pooled FID by construction.

|Dataset|Method|Reference|FID mean|Reference mean|Delta|Relative %|Samples|
|---|---|---|---:|---:|---:|---:|---:|
|ulong|P_oursLoss_withP_500|original|30.4514|33.302|-2.8506|-8.560|384|
|ulong|N_oursLoss_noP_500|original|30.0732|33.302|-3.22877|-9.695|384|
|ulong|P_oursLoss_withP_500|N_oursLoss_noP_500|30.4514|30.0732|+0.378166|+1.257|384|
|uvg_holdout|P_oursLoss_withP_500|original|28.4576|34.0527|-5.59509|-16.431|128|
|uvg_holdout|N_oursLoss_noP_500|original|28.8598|34.0527|-5.19286|-15.249|128|
|uvg_holdout|P_oursLoss_withP_500|N_oursLoss_noP_500|28.4576|28.8598|-0.40223|-1.394|128|
|hevc_b|P_oursLoss_withP_500|original|37.1674|39.4928|-2.3254|-5.888|320|
|hevc_b|N_oursLoss_noP_500|original|36.8834|39.4928|-2.60944|-6.607|320|
|hevc_b|P_oursLoss_withP_500|N_oursLoss_noP_500|37.1674|36.8834|+0.284041|+0.770|320|

## Scope and missing evidence

The official-source objective feasibility record lists the missing original training implementation, exact per-QP weighting/normalization, exact teacher checkpoint and preprocessing, exact margin definition/interface, and discriminator training implementation. No proxy teacher or substitute objective was used. Consequently this run measures (1) O vs P as the combined pipeline difference and (2) P vs N under the same complete v6.34b training objective; the objective-only comparison remains unmeasured. This is a fixed 13-video, three-QP screen and does not establish broader generalization.

The source-identity audit found no train/evaluation path, source SHA or domain/video identity intersection in the audited plan prefix. Exhaustive content near-duplicate search and official pretraining-data exposure are not claimed.

## Artifacts

- `results/all_rd_points.csv` — all 117 video/method/QP rows with source and bitstream hashes.
- `results/pooled_fid.csv` — 27 pooled dataset/method/QP points, with D_mu and D_cov.
- `results/common_rate_grid.csv`, `results/equal_rate_summary.csv`, `results/equal_rate_bands.csv` — common-rate paired comparisons.
- `plots/` — per-video RD charts, common-rate curves, pooled FID charts, and the fixed QP5 frame panels for the first listed video in each dataset.
- `audits/historical_reuse.json` — pointwise reuse verification.
- `final_integrity.json` — mechanical completion checks.
