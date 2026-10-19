# v6.36 results report

Execution status: **PASS**. New training: **500/500** B updates. Original and F were read-only reused.

## Fixed scope

Three methods: O_original, B_basicLoss_noP_500, F_fullLoss_noP_500. The new target is the explicit nested basic objective. F reuses v6.35 N_oursLoss_noP_500 at cumulative step 500. This report does not call the basic target an official objective. The F−B comparison is the combined DISTS + legacy spatial statistic + native-resolution FD package; individual terms are not separated.

## Point coverage

RD points: 117/117 (78 verified historical O/F rows; 39 new B encodes). Pooled FID points: 27/27. The fixed data sets are U-Long (6 videos), UVG holdout (2), HEVC-B (5), at QP 0/5/9. All rates are measured from full RANS streams.

## Per-video common-rate comparison

Each video uses its own O/B/F measured-bpp intersection, 20 linear points in ln(actual bpp), no extrapolation, then video-equal weighting. Relative changes are metric changes, not bitrate savings.

|Dataset|Method|Reference|Metric|Method mean|Reference mean|Delta|Relative %|Videos|
|---|---|---|---|---:|---:|---:|---:|---:|
|ulong|F_fullLoss_noP_500|B_basicLoss_noP_500|LPIPS|0.147553|0.148062|-0.000508939|-0.344|6|
|ulong|F_fullLoss_noP_500|B_basicLoss_noP_500|DISTS|0.0999582|0.101022|-0.00106337|-1.053|6|
|ulong|F_fullLoss_noP_500|B_basicLoss_noP_500|FloLPIPS|0.152906|0.153966|-0.00106049|-0.689|6|
|ulong|B_basicLoss_noP_500|O_original|LPIPS|0.148062|0.157209|-0.00914717|-5.818|6|
|ulong|B_basicLoss_noP_500|O_original|DISTS|0.101022|0.107061|-0.00603912|-5.641|6|
|ulong|B_basicLoss_noP_500|O_original|FloLPIPS|0.153966|0.162758|-0.00879128|-5.401|6|
|ulong|F_fullLoss_noP_500|O_original|LPIPS|0.147553|0.157209|-0.00965611|-6.142|6|
|ulong|F_fullLoss_noP_500|O_original|DISTS|0.0999582|0.107061|-0.00710249|-6.634|6|
|ulong|F_fullLoss_noP_500|O_original|FloLPIPS|0.152906|0.162758|-0.00985177|-6.053|6|
|uvg_holdout|F_fullLoss_noP_500|B_basicLoss_noP_500|LPIPS|0.15447|0.156739|-0.00226905|-1.448|2|
|uvg_holdout|F_fullLoss_noP_500|B_basicLoss_noP_500|DISTS|0.0929639|0.100207|-0.00724275|-7.228|2|
|uvg_holdout|F_fullLoss_noP_500|B_basicLoss_noP_500|FloLPIPS|0.158343|0.161302|-0.00295903|-1.834|2|
|uvg_holdout|B_basicLoss_noP_500|O_original|LPIPS|0.156739|0.160971|-0.0042326|-2.629|2|
|uvg_holdout|B_basicLoss_noP_500|O_original|DISTS|0.100207|0.102546|-0.00233961|-2.282|2|
|uvg_holdout|B_basicLoss_noP_500|O_original|FloLPIPS|0.161302|0.164297|-0.00299576|-1.823|2|
|uvg_holdout|F_fullLoss_noP_500|O_original|LPIPS|0.15447|0.160971|-0.00650166|-4.039|2|
|uvg_holdout|F_fullLoss_noP_500|O_original|DISTS|0.0929639|0.102546|-0.00958235|-9.344|2|
|uvg_holdout|F_fullLoss_noP_500|O_original|FloLPIPS|0.158343|0.164297|-0.00595479|-3.624|2|
|hevc_b|F_fullLoss_noP_500|B_basicLoss_noP_500|LPIPS|0.193332|0.194215|-0.000883023|-0.455|5|
|hevc_b|F_fullLoss_noP_500|B_basicLoss_noP_500|DISTS|0.0902073|0.0962388|-0.0060315|-6.267|5|
|hevc_b|F_fullLoss_noP_500|B_basicLoss_noP_500|FloLPIPS|0.185852|0.187005|-0.00115285|-0.616|5|
|hevc_b|B_basicLoss_noP_500|O_original|LPIPS|0.194215|0.205109|-0.0108939|-5.311|5|
|hevc_b|B_basicLoss_noP_500|O_original|DISTS|0.0962388|0.0971764|-0.000937606|-0.965|5|
|hevc_b|B_basicLoss_noP_500|O_original|FloLPIPS|0.187005|0.205222|-0.0182172|-8.877|5|
|hevc_b|F_fullLoss_noP_500|O_original|LPIPS|0.193332|0.205109|-0.0117769|-5.742|5|
|hevc_b|F_fullLoss_noP_500|O_original|DISTS|0.0902073|0.0971764|-0.00696911|-7.172|5|
|hevc_b|F_fullLoss_noP_500|O_original|FloLPIPS|0.185852|0.205222|-0.01937|-9.439|5|

## Pooled FID anchors

FID is pooled across dataset frames; per-video FID is not averaged. The values below are measured QP anchors.

|Dataset|Method|QP|FID|D_mu|D_cov|Mean actual bpp|Samples|
|---|---|---:|---:|---:|---:|---:|---:|
|ulong|O_original|0|60.5403|9.61049|50.9298|0.002052811|384|
|ulong|O_original|5|23.0165|3.00316|20.0134|0.007712654|384|
|ulong|O_original|9|14.4522|1.59203|12.8602|0.01722559|384|
|ulong|B_basicLoss_noP_500|0|57.2407|8.68094|48.5597|0.002052811|384|
|ulong|B_basicLoss_noP_500|5|22.0662|2.67536|19.3908|0.007712654|384|
|ulong|B_basicLoss_noP_500|9|13.9014|1.40426|12.4972|0.01722559|384|
|ulong|F_fullLoss_noP_500|0|55.5463|8.12787|47.4184|0.002052811|384|
|ulong|F_fullLoss_noP_500|5|20.3612|2.33958|18.0217|0.007712654|384|
|ulong|F_fullLoss_noP_500|9|12.6605|1.1922|11.4683|0.01722559|384|
|uvg_holdout|O_original|0|62.6774|13.0204|49.657|0.004222367|128|
|uvg_holdout|O_original|5|22.2699|4.04057|18.2293|0.01374569|128|
|uvg_holdout|O_original|9|15.9853|3.01657|12.9688|0.02780204|128|
|uvg_holdout|B_basicLoss_noP_500|0|58.5941|12.3034|46.2906|0.004222367|128|
|uvg_holdout|B_basicLoss_noP_500|5|21.9205|3.89442|18.0261|0.01374569|128|
|uvg_holdout|B_basicLoss_noP_500|9|15.7383|2.84908|12.8892|0.02780204|128|
|uvg_holdout|F_fullLoss_noP_500|0|50.6797|9.9837|40.696|0.004222367|128|
|uvg_holdout|F_fullLoss_noP_500|5|20.097|3.36061|16.7364|0.01374569|128|
|uvg_holdout|F_fullLoss_noP_500|9|14.5721|2.58702|11.9851|0.02780204|128|
|hevc_b|O_original|0|62.3451|6.90418|55.441|0.002624144|320|
|hevc_b|O_original|5|31.4712|2.79991|28.6713|0.00941856|320|
|hevc_b|O_original|9|22.5976|1.60609|20.9915|0.02076314|320|
|hevc_b|B_basicLoss_noP_500|0|61.2787|6.97511|54.3036|0.002624144|320|
|hevc_b|B_basicLoss_noP_500|5|31.894|2.94528|28.9487|0.00941856|320|
|hevc_b|B_basicLoss_noP_500|9|22.6203|1.61592|21.0044|0.02076314|320|
|hevc_b|F_fullLoss_noP_500|0|58.0397|6.25137|51.7884|0.002624144|320|
|hevc_b|F_fullLoss_noP_500|5|29.5655|2.48443|27.0811|0.00941856|320|
|hevc_b|F_fullLoss_noP_500|9|20.9927|1.3248|19.6679|0.02076314|320|

## Direct measured QP comparisons

The full per-video table is `results/same_qp_pairs.csv`; QP matches are not treated as rate matches. Bitstream equality is independently listed for each pair.

Bitstream hash equality among the 117 same-QP method pairs: 117/117.

Evaluation retries: 6 task(s), 7 failed attempt(s); original failures and tracebacks are preserved in task logs.

## Artifacts

- `results/all_rd_points.csv`: 117 measured/reused point records.
- `results/pooled_fid.csv`: 27 pooled FID points with D_mu/D_cov.
- `results/equal_rate_summary.csv`: per-video common-rate interpolation summary.
- `results/equal_rate_per_video.csv`, `results/equal_rate_bands.csv`, `results/global_common_rate_summary.csv`.
- `audits/reuse_pointwise_verification.json`: historical artifact hash checks.
- `final_integrity.json`: calculation status.

The screen uses one training seed and a fixed 500-update budget. It does not establish convergence.
