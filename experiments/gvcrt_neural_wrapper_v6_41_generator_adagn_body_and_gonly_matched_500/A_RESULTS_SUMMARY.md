# v6.41 A generator body vs AdaGN parameter intervention

Execution: **PASS**. New training updates **0**; new real encodes **0**. RD **156/156**, pooled FID **36/36**, verified common RANS streams **39/39**. Endpoints bitwise exact: **True**.

All four methods use the same original bridge z0, q_recon and persisted stream. A00=(G0 body,G0 AdaGN), A01=(G0 body,GF AdaGN), A10=(GF body,G0 AdaGN), A11=(GF body,GF AdaGN). Negative quality delta means improvement; it does not mean fewer bits. A11 is the historical Y endpoint, not the full BG endpoint F.

## Fixed equal-rate contrasts

|Dataset|Effect|Metric|Method|Reference|Mean delta|Relative %|
|---|---|---|---|---|---:|---:|
|ulong|body_given_old_mod|LPIPS|A10|A00|-0.005817|-3.700|
|ulong|body_given_new_mod|LPIPS|A11|A01|-0.005636|-3.588|
|ulong|mod_given_old_body|LPIPS|A01|A00|-0.000109|-0.069|
|ulong|mod_given_new_body|LPIPS|A11|A10|0.000072|0.048|
|ulong|joint_generator|LPIPS|A11|A00|-0.005745|-3.655|
|ulong|body_given_old_mod|DISTS|A10|A00|-0.003491|-3.261|
|ulong|body_given_new_mod|DISTS|A11|A01|-0.003367|-3.144|
|ulong|mod_given_old_body|DISTS|A01|A00|0.000041|0.039|
|ulong|mod_given_new_body|DISTS|A11|A10|0.000165|0.160|
|ulong|joint_generator|DISTS|A11|A00|-0.003326|-3.106|
|ulong|body_given_old_mod|FloLPIPS|A10|A00|-0.005636|-3.463|
|ulong|body_given_new_mod|FloLPIPS|A11|A01|-0.005463|-3.360|
|ulong|mod_given_old_body|FloLPIPS|A01|A00|-0.000182|-0.112|
|ulong|mod_given_new_body|FloLPIPS|A11|A10|-0.000009|-0.006|
|ulong|joint_generator|FloLPIPS|A11|A00|-0.005645|-3.468|
|ulong|body_given_old_mod|FID|A10|A00|-0.963820|-2.894|
|ulong|body_given_new_mod|FID|A11|A01|-0.966840|-2.901|
|ulong|mod_given_old_body|FID|A01|A00|0.022877|0.069|
|ulong|mod_given_new_body|FID|A11|A10|0.019857|0.061|
|ulong|joint_generator|FID|A11|A00|-0.943963|-2.835|
|uvg_holdout|body_given_old_mod|LPIPS|A10|A00|-0.007125|-4.426|
|uvg_holdout|body_given_new_mod|LPIPS|A11|A01|-0.006911|-4.303|
|uvg_holdout|mod_given_old_body|LPIPS|A01|A00|-0.000346|-0.215|
|uvg_holdout|mod_given_new_body|LPIPS|A11|A10|-0.000133|-0.086|
|uvg_holdout|joint_generator|LPIPS|A11|A00|-0.007258|-4.509|
|uvg_holdout|body_given_old_mod|DISTS|A10|A00|-0.009558|-9.320|
|uvg_holdout|body_given_new_mod|DISTS|A11|A01|-0.009476|-9.271|
|uvg_holdout|mod_given_old_body|DISTS|A01|A00|-0.000335|-0.326|
|uvg_holdout|mod_given_new_body|DISTS|A11|A10|-0.000253|-0.272|
|uvg_holdout|joint_generator|DISTS|A11|A00|-0.009811|-9.567|
|uvg_holdout|body_given_old_mod|FloLPIPS|A10|A00|-0.006538|-3.979|
|uvg_holdout|body_given_new_mod|FloLPIPS|A11|A01|-0.006245|-3.810|
|uvg_holdout|mod_given_old_body|FloLPIPS|A01|A00|-0.000393|-0.239|
|uvg_holdout|mod_given_new_body|FloLPIPS|A11|A10|-0.000100|-0.063|
|uvg_holdout|joint_generator|FloLPIPS|A11|A00|-0.006638|-4.040|
|uvg_holdout|body_given_old_mod|FID|A10|A00|-4.571998|-13.434|
|uvg_holdout|body_given_new_mod|FID|A11|A01|-4.554810|-13.412|
|uvg_holdout|mod_given_old_body|FID|A01|A00|-0.072413|-0.213|
|uvg_holdout|mod_given_new_body|FID|A11|A10|-0.055224|-0.187|
|uvg_holdout|joint_generator|FID|A11|A00|-4.627222|-13.596|
|hevc_b|body_given_old_mod|LPIPS|A10|A00|-0.011650|-5.680|
|hevc_b|body_given_new_mod|LPIPS|A11|A01|-0.011480|-5.614|
|hevc_b|mod_given_old_body|LPIPS|A01|A00|-0.000624|-0.304|
|hevc_b|mod_given_new_body|LPIPS|A11|A10|-0.000454|-0.235|
|hevc_b|joint_generator|LPIPS|A11|A00|-0.012104|-5.901|
|hevc_b|body_given_old_mod|DISTS|A10|A00|-0.007636|-7.858|
|hevc_b|body_given_new_mod|DISTS|A11|A01|-0.007637|-7.889|
|hevc_b|mod_given_old_body|DISTS|A01|A00|-0.000371|-0.381|
|hevc_b|mod_given_new_body|DISTS|A11|A10|-0.000371|-0.415|
|hevc_b|joint_generator|DISTS|A11|A00|-0.008008|-8.241|
|hevc_b|body_given_old_mod|FloLPIPS|A10|A00|-0.014513|-7.072|
|hevc_b|body_given_new_mod|FloLPIPS|A11|A01|-0.014375|-7.024|
|hevc_b|mod_given_old_body|FloLPIPS|A01|A00|-0.000559|-0.272|
|hevc_b|mod_given_new_body|FloLPIPS|A11|A10|-0.000421|-0.221|
|hevc_b|joint_generator|FloLPIPS|A11|A00|-0.014934|-7.277|
|hevc_b|body_given_old_mod|FID|A10|A00|-2.290515|-5.802|
|hevc_b|body_given_new_mod|FID|A11|A01|-2.305055|-5.842|
|hevc_b|mod_given_old_body|FID|A01|A00|-0.017732|-0.045|
|hevc_b|mod_given_new_body|FID|A11|A10|-0.032272|-0.087|
|hevc_b|joint_generator|FID|A11|A00|-2.322787|-5.884|

## Interaction on the same fixed intervals

The raw-unit interaction is Q11−Q10−Q01+Q00. It is not a fraction of a contribution.

|Dataset|Metric|Interaction|Aggregation|
|---|---|---:|---|
|ulong|FID|-0.003020|pooled FID component curve|
|ulong|D_mu|-0.001938|pooled FID component curve|
|ulong|D_cov|-0.001082|pooled FID component curve|
|uvg_holdout|FID|0.017188|pooled FID component curve|
|uvg_holdout|D_mu|0.006344|pooled FID component curve|
|uvg_holdout|D_cov|0.010844|pooled FID component curve|
|hevc_b|FID|-0.014540|pooled FID component curve|
|hevc_b|D_mu|-0.002382|pooled FID component curve|
|hevc_b|D_cov|-0.012158|pooled FID component curve|
|ulong|LPIPS|0.000181|video equal weight|
|ulong|DISTS|0.000124|video equal weight|
|ulong|FloLPIPS|0.000174|video equal weight|
|uvg_holdout|LPIPS|0.000214|video equal weight|
|uvg_holdout|DISTS|0.000082|video equal weight|
|uvg_holdout|FloLPIPS|0.000292|video equal weight|
|hevc_b|LPIPS|0.000170|video equal weight|
|hevc_b|DISTS|-0.000001|video equal weight|
|hevc_b|FloLPIPS|0.000138|video equal weight|

## Direct same-stream anchors

`results/A_same_stream_pairs.csv` contains every video/QP comparison. Each four-method set has exactly the same persisted stream SHA and measured bpp. The equal-rate grid is diagnostic interpolation, not additional encoded samples.


