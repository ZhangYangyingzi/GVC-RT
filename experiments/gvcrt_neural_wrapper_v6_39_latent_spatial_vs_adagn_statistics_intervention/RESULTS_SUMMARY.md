# v6.39 latent spatial vs AdaGN statistics intervention

Execution: **PASS**. New training updates **0**; new real encodes **0**. RD **156/156**, pooled FID **36/36**, verified common RANS streams **39/39**. Endpoints bitwise exact: **True**.

All four methods use the same frozen trained generator G_F and q_recon. M00=(z0,z0), M10=(zF,z0), M01=(z0,zF), M11=(zF,zF), where the tuple gives the convolution input and all five AdaGN statistics inputs. Negative quality delta means improvement; it does not mean fewer bits.

## Fixed equal-rate contrasts

|Dataset|Effect|Metric|Method|Reference|Mean delta|Relative %|
|---|---|---|---|---|---:|---:|
|ulong|spatial_given_old_stats|LPIPS|M10|M00|-0.000715|-0.472|
|ulong|spatial_given_new_stats|LPIPS|M11|M01|-0.000371|-0.251|
|ulong|stats_given_old_spatial|LPIPS|M01|M00|-0.003539|-2.337|
|ulong|stats_given_new_spatial|LPIPS|M11|M10|-0.003196|-2.120|
|ulong|joint|LPIPS|M11|M00|-0.003911|-2.582|
|ulong|spatial_given_old_stats|DISTS|M10|M00|-0.000925|-0.891|
|ulong|spatial_given_new_stats|DISTS|M11|M01|-0.000486|-0.483|
|ulong|stats_given_old_spatial|DISTS|M01|M00|-0.003291|-3.173|
|ulong|stats_given_new_spatial|DISTS|M11|M10|-0.002852|-2.774|
|ulong|joint|DISTS|M11|M00|-0.003777|-3.641|
|ulong|spatial_given_old_stats|FloLPIPS|M10|M00|-0.000439|-0.280|
|ulong|spatial_given_new_stats|FloLPIPS|M11|M01|-0.000013|-0.008|
|ulong|stats_given_old_spatial|FloLPIPS|M01|M00|-0.004194|-2.669|
|ulong|stats_given_new_spatial|FloLPIPS|M11|M10|-0.003767|-2.405|
|ulong|joint|FloLPIPS|M11|M00|-0.004207|-2.678|
|ulong|spatial_given_old_stats|FID|M10|M00|-1.191699|-3.683|
|ulong|spatial_given_new_stats|FID|M11|M01|-1.043272|-3.353|
|ulong|stats_given_old_spatial|FID|M01|M00|-1.241533|-3.837|
|ulong|stats_given_new_spatial|FID|M11|M10|-1.093106|-3.507|
|ulong|joint|FID|M11|M00|-2.284805|-7.061|
|uvg_holdout|spatial_given_old_stats|LPIPS|M10|M00|0.000731|0.475|
|uvg_holdout|spatial_given_new_stats|LPIPS|M11|M01|0.001245|0.813|
|uvg_holdout|stats_given_old_spatial|LPIPS|M01|M00|-0.000489|-0.318|
|uvg_holdout|stats_given_new_spatial|LPIPS|M11|M10|0.000025|0.016|
|uvg_holdout|joint|LPIPS|M11|M00|0.000756|0.492|
|uvg_holdout|spatial_given_old_stats|DISTS|M10|M00|0.000087|0.094|
|uvg_holdout|spatial_given_new_stats|DISTS|M11|M01|0.000481|0.520|
|uvg_holdout|stats_given_old_spatial|DISTS|M01|M00|-0.000253|-0.273|
|uvg_holdout|stats_given_new_spatial|DISTS|M11|M10|0.000141|0.152|
|uvg_holdout|joint|DISTS|M11|M00|0.000228|0.246|
|uvg_holdout|spatial_given_old_stats|FloLPIPS|M10|M00|0.000901|0.571|
|uvg_holdout|spatial_given_new_stats|FloLPIPS|M11|M01|0.001581|1.009|
|uvg_holdout|stats_given_old_spatial|FloLPIPS|M01|M00|-0.000898|-0.570|
|uvg_holdout|stats_given_new_spatial|FloLPIPS|M11|M10|-0.000217|-0.137|
|uvg_holdout|joint|FloLPIPS|M11|M00|0.000683|0.433|
|uvg_holdout|spatial_given_old_stats|FID|M10|M00|-0.285255|-0.970|
|uvg_holdout|spatial_given_new_stats|FID|M11|M01|-0.319597|-1.096|
|uvg_holdout|stats_given_old_spatial|FID|M01|M00|-0.242099|-0.823|
|uvg_holdout|stats_given_new_spatial|FID|M11|M10|-0.276442|-0.949|
|uvg_holdout|joint|FID|M11|M00|-0.561696|-1.910|
|hevc_b|spatial_given_old_stats|LPIPS|M10|M00|-0.000427|-0.221|
|hevc_b|spatial_given_new_stats|LPIPS|M11|M01|0.000165|0.086|
|hevc_b|stats_given_old_spatial|LPIPS|M01|M00|0.000162|0.084|
|hevc_b|stats_given_new_spatial|LPIPS|M11|M10|0.000755|0.392|
|hevc_b|joint|LPIPS|M11|M00|0.000327|0.170|
|hevc_b|spatial_given_old_stats|DISTS|M10|M00|0.000545|0.612|
|hevc_b|spatial_given_new_stats|DISTS|M11|M01|0.000692|0.773|
|hevc_b|stats_given_old_spatial|DISTS|M01|M00|0.000347|0.389|
|hevc_b|stats_given_new_spatial|DISTS|M11|M10|0.000493|0.550|
|hevc_b|joint|DISTS|M11|M00|0.001039|1.165|
|hevc_b|spatial_given_old_stats|FloLPIPS|M10|M00|-0.003042|-1.599|
|hevc_b|spatial_given_new_stats|FloLPIPS|M11|M01|-0.002719|-1.442|
|hevc_b|stats_given_old_spatial|FloLPIPS|M01|M00|-0.001717|-0.902|
|hevc_b|stats_given_new_spatial|FloLPIPS|M11|M10|-0.001394|-0.745|
|hevc_b|joint|FloLPIPS|M11|M00|-0.004436|-2.331|
|hevc_b|spatial_given_old_stats|FID|M10|M00|-0.134154|-0.361|
|hevc_b|spatial_given_new_stats|FID|M11|M01|-0.133399|-0.361|
|hevc_b|stats_given_old_spatial|FID|M01|M00|-0.152387|-0.410|
|hevc_b|stats_given_new_spatial|FID|M11|M10|-0.151632|-0.410|
|hevc_b|joint|FID|M11|M00|-0.285785|-0.769|

## Interaction on the same fixed intervals

The raw-unit interaction is Q11−Q10−Q01+Q00. It is not a fraction of a contribution.

|Dataset|Metric|Interaction|Aggregation|
|---|---|---:|---|
|ulong|FID|0.148426|pooled FID component curve|
|ulong|D_mu|0.033389|pooled FID component curve|
|ulong|D_cov|0.115037|pooled FID component curve|
|uvg_holdout|FID|-0.034343|pooled FID component curve|
|uvg_holdout|D_mu|0.007305|pooled FID component curve|
|uvg_holdout|D_cov|-0.041648|pooled FID component curve|
|hevc_b|FID|0.000755|pooled FID component curve|
|hevc_b|D_mu|0.016865|pooled FID component curve|
|hevc_b|D_cov|-0.016110|pooled FID component curve|
|ulong|LPIPS|0.000343|video equal weight|
|ulong|DISTS|0.000439|video equal weight|
|ulong|FloLPIPS|0.000427|video equal weight|
|uvg_holdout|LPIPS|0.000515|video equal weight|
|uvg_holdout|DISTS|0.000394|video equal weight|
|uvg_holdout|FloLPIPS|0.000681|video equal weight|
|hevc_b|LPIPS|0.000593|video equal weight|
|hevc_b|DISTS|0.000147|video equal weight|
|hevc_b|FloLPIPS|0.000322|video equal weight|

## Direct same-stream anchors

`results/same_stream_pairs.csv` contains every video/QP comparison. Each four-method set has exactly the same persisted stream SHA and measured bpp. The equal-rate grid is diagnostic interpolation, not additional encoded samples.


