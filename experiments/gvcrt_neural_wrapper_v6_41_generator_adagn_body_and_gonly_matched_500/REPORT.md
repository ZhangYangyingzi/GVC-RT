# v6.41 generator parameter and G-only matched control

Execution **PASS**. A 156/156 RD, 36/36 pooled FID; B 117/117 RD, 27/27 pooled FID. Global unique 234/234 RD and 54/54 FID. Reused streams 39; new encodes 0. G-only committed updates 500/500.

A00 is original bridge+G0; A11 is original bridge+BG-trained generator. F_BG500 uses the BG-trained bridge and generator. All A inputs use the same z0. Negative quality delta is better at identical actual bitrate.

## Prespecified equal-rate contrasts

|Panel|Dataset|Contrast|Metric|Method|Reference|Absolute delta|Relative %|
|---|---|---|---|---|---|---:|---:|
|A|ulong|body_new_mod|LPIPS|A11|A01|-0.005636|-3.588|
|A|ulong|body_old_mod|LPIPS|A10|A00|-0.005817|-3.700|
|A|ulong|full_generator|LPIPS|A11|A00|-0.005745|-3.655|
|A|ulong|mod_new_body|LPIPS|A11|A10|0.000072|0.048|
|A|ulong|mod_old_body|LPIPS|A01|A00|-0.000109|-0.069|
|A|ulong|body_new_mod|DISTS|A11|A01|-0.003367|-3.144|
|A|ulong|body_old_mod|DISTS|A10|A00|-0.003491|-3.261|
|A|ulong|full_generator|DISTS|A11|A00|-0.003326|-3.106|
|A|ulong|mod_new_body|DISTS|A11|A10|0.000165|0.160|
|A|ulong|mod_old_body|DISTS|A01|A00|0.000041|0.039|
|A|ulong|body_new_mod|FloLPIPS|A11|A01|-0.005463|-3.360|
|A|ulong|body_old_mod|FloLPIPS|A10|A00|-0.005636|-3.463|
|A|ulong|full_generator|FloLPIPS|A11|A00|-0.005645|-3.468|
|A|ulong|mod_new_body|FloLPIPS|A11|A10|-0.000009|-0.006|
|A|ulong|mod_old_body|FloLPIPS|A01|A00|-0.000182|-0.112|
|A|ulong|mod_old_body|FID|A01|A00|0.022877|0.069|
|A|ulong|mod_new_body|FID|A11|A10|0.019857|0.061|
|A|ulong|body_old_mod|FID|A10|A00|-0.963820|-2.894|
|A|ulong|body_new_mod|FID|A11|A01|-0.966840|-2.901|
|A|ulong|full_generator|FID|A11|A00|-0.943963|-2.835|
|A|uvg_holdout|body_new_mod|LPIPS|A11|A01|-0.006911|-4.303|
|A|uvg_holdout|body_old_mod|LPIPS|A10|A00|-0.007125|-4.426|
|A|uvg_holdout|full_generator|LPIPS|A11|A00|-0.007258|-4.509|
|A|uvg_holdout|mod_new_body|LPIPS|A11|A10|-0.000133|-0.086|
|A|uvg_holdout|mod_old_body|LPIPS|A01|A00|-0.000346|-0.215|
|A|uvg_holdout|body_new_mod|DISTS|A11|A01|-0.009476|-9.271|
|A|uvg_holdout|body_old_mod|DISTS|A10|A00|-0.009558|-9.320|
|A|uvg_holdout|full_generator|DISTS|A11|A00|-0.009811|-9.567|
|A|uvg_holdout|mod_new_body|DISTS|A11|A10|-0.000253|-0.272|
|A|uvg_holdout|mod_old_body|DISTS|A01|A00|-0.000335|-0.326|
|A|uvg_holdout|body_new_mod|FloLPIPS|A11|A01|-0.006245|-3.810|
|A|uvg_holdout|body_old_mod|FloLPIPS|A10|A00|-0.006538|-3.979|
|A|uvg_holdout|full_generator|FloLPIPS|A11|A00|-0.006638|-4.040|
|A|uvg_holdout|mod_new_body|FloLPIPS|A11|A10|-0.000100|-0.063|
|A|uvg_holdout|mod_old_body|FloLPIPS|A01|A00|-0.000393|-0.239|
|A|uvg_holdout|mod_old_body|FID|A01|A00|-0.072413|-0.213|
|A|uvg_holdout|mod_new_body|FID|A11|A10|-0.055224|-0.187|
|A|uvg_holdout|body_old_mod|FID|A10|A00|-4.571998|-13.434|
|A|uvg_holdout|body_new_mod|FID|A11|A01|-4.554810|-13.412|
|A|uvg_holdout|full_generator|FID|A11|A00|-4.627222|-13.596|
|A|hevc_b|body_new_mod|LPIPS|A11|A01|-0.011480|-5.614|
|A|hevc_b|body_old_mod|LPIPS|A10|A00|-0.011650|-5.680|
|A|hevc_b|full_generator|LPIPS|A11|A00|-0.012104|-5.901|
|A|hevc_b|mod_new_body|LPIPS|A11|A10|-0.000454|-0.235|
|A|hevc_b|mod_old_body|LPIPS|A01|A00|-0.000624|-0.304|
|A|hevc_b|body_new_mod|DISTS|A11|A01|-0.007637|-7.889|
|A|hevc_b|body_old_mod|DISTS|A10|A00|-0.007636|-7.858|
|A|hevc_b|full_generator|DISTS|A11|A00|-0.008008|-8.241|
|A|hevc_b|mod_new_body|DISTS|A11|A10|-0.000371|-0.415|
|A|hevc_b|mod_old_body|DISTS|A01|A00|-0.000371|-0.381|
|A|hevc_b|body_new_mod|FloLPIPS|A11|A01|-0.014375|-7.024|
|A|hevc_b|body_old_mod|FloLPIPS|A10|A00|-0.014513|-7.072|
|A|hevc_b|full_generator|FloLPIPS|A11|A00|-0.014934|-7.277|
|A|hevc_b|mod_new_body|FloLPIPS|A11|A10|-0.000421|-0.221|
|A|hevc_b|mod_old_body|FloLPIPS|A01|A00|-0.000559|-0.272|
|A|hevc_b|mod_old_body|FID|A01|A00|-0.017732|-0.045|
|A|hevc_b|mod_new_body|FID|A11|A10|-0.032272|-0.087|
|A|hevc_b|body_old_mod|FID|A10|A00|-2.290515|-5.802|
|A|hevc_b|body_new_mod|FID|A11|A01|-2.305055|-5.842|
|A|hevc_b|full_generator|FID|A11|A00|-2.322787|-5.884|
|B|ulong|BG_vs_O|LPIPS|F_BG500|A00|-0.009656|-6.142|
|B|ulong|Gonly_vs_BG|LPIPS|G_fullLoss_Gonly_500|F_BG500|0.001582|1.072|
|B|ulong|Gonly_vs_O|LPIPS|G_fullLoss_Gonly_500|A00|-0.008074|-5.136|
|B|ulong|BG_vs_O|DISTS|F_BG500|A00|-0.007102|-6.634|
|B|ulong|Gonly_vs_BG|DISTS|G_fullLoss_Gonly_500|F_BG500|0.000129|0.129|
|B|ulong|Gonly_vs_O|DISTS|G_fullLoss_Gonly_500|A00|-0.006973|-6.514|
|B|ulong|BG_vs_O|FloLPIPS|F_BG500|A00|-0.009852|-6.053|
|B|ulong|Gonly_vs_BG|FloLPIPS|G_fullLoss_Gonly_500|F_BG500|0.001557|1.018|
|B|ulong|Gonly_vs_O|FloLPIPS|G_fullLoss_Gonly_500|A00|-0.008295|-5.097|
|B|ulong|Gonly_vs_O|FID|G_fullLoss_Gonly_500|A00|-2.064666|-6.200|
|B|ulong|BG_vs_O|FID|F_BG500|A00|-3.228768|-9.695|
|B|ulong|Gonly_vs_BG|FID|G_fullLoss_Gonly_500|F_BG500|1.164102|3.871|
|B|uvg_holdout|BG_vs_O|LPIPS|F_BG500|A00|-0.006502|-4.039|
|B|uvg_holdout|Gonly_vs_BG|LPIPS|G_fullLoss_Gonly_500|F_BG500|-0.000747|-0.484|
|B|uvg_holdout|Gonly_vs_O|LPIPS|G_fullLoss_Gonly_500|A00|-0.007249|-4.503|
|B|uvg_holdout|BG_vs_O|DISTS|F_BG500|A00|-0.009582|-9.344|
|B|uvg_holdout|Gonly_vs_BG|DISTS|G_fullLoss_Gonly_500|F_BG500|0.000840|0.904|
|B|uvg_holdout|Gonly_vs_O|DISTS|G_fullLoss_Gonly_500|A00|-0.008742|-8.525|
|B|uvg_holdout|BG_vs_O|FloLPIPS|F_BG500|A00|-0.005955|-3.624|
|B|uvg_holdout|Gonly_vs_BG|FloLPIPS|G_fullLoss_Gonly_500|F_BG500|-0.001075|-0.679|
|B|uvg_holdout|Gonly_vs_O|FloLPIPS|G_fullLoss_Gonly_500|A00|-0.007030|-4.279|
|B|uvg_holdout|Gonly_vs_O|FID|G_fullLoss_Gonly_500|A00|-5.083401|-14.937|
|B|uvg_holdout|BG_vs_O|FID|F_BG500|A00|-5.188918|-15.247|
|B|uvg_holdout|Gonly_vs_BG|FID|G_fullLoss_Gonly_500|F_BG500|0.105518|0.366|
|B|hevc_b|BG_vs_O|LPIPS|F_BG500|A00|-0.011777|-5.742|
|B|hevc_b|Gonly_vs_BG|LPIPS|G_fullLoss_Gonly_500|F_BG500|-0.000500|-0.259|
|B|hevc_b|Gonly_vs_O|LPIPS|G_fullLoss_Gonly_500|A00|-0.012277|-5.985|
|B|hevc_b|BG_vs_O|DISTS|F_BG500|A00|-0.006969|-7.172|
|B|hevc_b|Gonly_vs_BG|DISTS|G_fullLoss_Gonly_500|F_BG500|-0.001493|-1.655|
|B|hevc_b|Gonly_vs_O|DISTS|G_fullLoss_Gonly_500|A00|-0.008462|-8.708|
|B|hevc_b|BG_vs_O|FloLPIPS|F_BG500|A00|-0.019370|-9.439|
|B|hevc_b|Gonly_vs_BG|FloLPIPS|G_fullLoss_Gonly_500|F_BG500|0.002122|1.142|
|B|hevc_b|Gonly_vs_O|FloLPIPS|G_fullLoss_Gonly_500|A00|-0.017248|-8.405|
|B|hevc_b|Gonly_vs_O|FID|G_fullLoss_Gonly_500|A00|-2.735538|-6.929|
|B|hevc_b|BG_vs_O|FID|F_BG500|A00|-2.608573|-6.608|
|B|hevc_b|Gonly_vs_BG|FID|G_fullLoss_Gonly_500|F_BG500|-0.126965|-0.344|

See `results/all_rd_points.csv` for measured QP anchors, `results/pooled_fid.csv` for pooled Inception FID and mean/covariance terms, and `results/rate_coverage.csv` for common intervals. The 20 interpolation points are diagnostic, not independent encoded samples. A mixed-weight degradation can reflect incompatibility. G-only vs BG includes the difference in trainable set and optimizer trajectory under the same rule; this single seed, 13-video, three-QP comparison does not establish general behavior or convergence.
