# v6.41 generator parameter and G-only matched control

Execution **PASS**. A 156/156 RD, 36/36 pooled FID; B 117/117 RD, 27/27 pooled FID. Global unique 234/234 RD and 54/54 FID. Reused streams 39; new encodes 0. G-only committed updates 500/500.

A00 is original bridge+G0; A11 is original bridge+BG-trained generator. F_BG500 uses the BG-trained bridge and generator. All A inputs use the same z0. Negative quality delta is better at identical actual bitrate.

## Prespecified equal-rate contrasts

|Panel|Dataset|Contrast|Metric|Method|Reference|Absolute delta|Relative %|
|---|---|---|---|---|---|---:|---:|
|A|ulong|full_generator|LPIPS|A11|A00|-0.005745|-3.655|
|A|ulong|full_generator|DISTS|A11|A00|-0.003326|-3.106|
|A|ulong|full_generator|FloLPIPS|A11|A00|-0.005645|-3.468|
|A|ulong|full_generator|FID|A11|A00|-0.943963|-2.835|
|A|uvg_holdout|full_generator|LPIPS|A11|A00|-0.007258|-4.509|
|A|uvg_holdout|full_generator|DISTS|A11|A00|-0.009811|-9.567|
|A|uvg_holdout|full_generator|FloLPIPS|A11|A00|-0.006638|-4.040|
|A|uvg_holdout|full_generator|FID|A11|A00|-4.627222|-13.596|
|A|hevc_b|full_generator|LPIPS|A11|A00|-0.012104|-5.901|
|A|hevc_b|full_generator|DISTS|A11|A00|-0.008008|-8.241|
|A|hevc_b|full_generator|FloLPIPS|A11|A00|-0.014934|-7.277|
|A|hevc_b|full_generator|FID|A11|A00|-2.322787|-5.884|
|B|ulong|Gonly_vs_BG|LPIPS|G_fullLoss_Gonly_500|F_BG500|0.001582|1.072|
|B|ulong|Gonly_vs_BG|DISTS|G_fullLoss_Gonly_500|F_BG500|0.000129|0.129|
|B|ulong|Gonly_vs_BG|FloLPIPS|G_fullLoss_Gonly_500|F_BG500|0.001557|1.018|
|B|ulong|Gonly_vs_BG|FID|G_fullLoss_Gonly_500|F_BG500|1.164102|3.871|
|B|uvg_holdout|Gonly_vs_BG|LPIPS|G_fullLoss_Gonly_500|F_BG500|-0.000747|-0.484|
|B|uvg_holdout|Gonly_vs_BG|DISTS|G_fullLoss_Gonly_500|F_BG500|0.000840|0.904|
|B|uvg_holdout|Gonly_vs_BG|FloLPIPS|G_fullLoss_Gonly_500|F_BG500|-0.001075|-0.679|
|B|uvg_holdout|Gonly_vs_BG|FID|G_fullLoss_Gonly_500|F_BG500|0.105518|0.366|
|B|hevc_b|Gonly_vs_BG|LPIPS|G_fullLoss_Gonly_500|F_BG500|-0.000500|-0.259|
|B|hevc_b|Gonly_vs_BG|DISTS|G_fullLoss_Gonly_500|F_BG500|-0.001493|-1.655|
|B|hevc_b|Gonly_vs_BG|FloLPIPS|G_fullLoss_Gonly_500|F_BG500|0.002122|1.142|
|B|hevc_b|Gonly_vs_BG|FID|G_fullLoss_Gonly_500|F_BG500|-0.126965|-0.344|

