# Independent FID audit

Execution:PASS. Features18/18; explicit covariance statistics18/18; full GT self-distance/symmetry and analytic Gaussian checks passed. This does not constitute a statistical-estimator reliability guarantee.

## Same-QP matched pooled differences

Deltas=B1−original on identical frame UIDs. These are diagnostic same-QP comparisons, not equal-rate gain claims.

|Dataset|QP|Delta FID|Delta D_mu|Delta D_cov|
|---|---:|---:|---:|---:|
|ulong|0|-0.28529468|-0.247566082|-0.0377285978|
|ulong|5|0.173348459|-0.10429682|0.27764528|
|ulong|9|1.43790963|0.064689771|1.37321986|
|hevc_b|0|5.64461266|1.74364665|3.90096601|
|hevc_b|5|3.98634559|0.52620904|3.46013655|
|hevc_b|9|4.30188004|0.506808325|3.79507171|
|mcljcv|0|0.818123976|-0.118959963|0.937083939|
|mcljcv|5|1.07289444|-0.0521058927|1.12500033|
|mcljcv|9|3.09437108|0.0281087898|3.06626229|

## Fixed frame-group differences

Only equal-identity, equal-count original/B1 pairs within a group are compared. Different group sample counts do not by themselves establish temporal drift.

|Dataset|QP|Group|Samples|Delta FID|Delta D_mu|Delta D_cov|
|---|---:|---|---:|---:|---:|---:|
|ulong|0|I0|8|0|0|0|
|ulong|0|P1_16|128|1.1045466|-0.16766329|1.27220989|
|ulong|0|P17_32|128|2.04589629|0.916141857|1.12975443|
|ulong|0|P33_48|128|1.02379621|-0.257112308|1.28090852|
|ulong|0|P49_63|120|-4.94996086|-1.33955599|-3.61040488|
|ulong|5|I0|8|0|0|0|
|ulong|5|P1_16|128|0.782221997|0.0596624194|0.722559577|
|ulong|5|P17_32|128|0.910370501|0.0424390438|0.867931458|
|ulong|5|P33_48|128|-1.45970937|-0.278220412|-1.18148896|
|ulong|5|P49_63|120|-0.33771254|-0.215598549|-0.122113991|
|ulong|9|I0|8|0|0|0|
|ulong|9|P1_16|128|0.90024654|-0.0019981714|0.902244711|
|ulong|9|P17_32|128|2.59285829|0.162135775|2.43072252|
|ulong|9|P33_48|128|1.70640645|0.155708229|1.55069822|
|ulong|9|P49_63|120|0.805265231|0.034921675|0.770343556|
|hevc_b|0|I0|5|0|0|0|
|hevc_b|0|P1_16|80|2.56916697|1.11043535|1.45873162|
|hevc_b|0|P17_32|80|6.08751863|1.54376775|4.54375089|
|hevc_b|0|P33_48|80|6.64972569|2.44961695|4.20010874|
|hevc_b|0|P49_63|75|8.821791|2.25728582|6.56450518|
|hevc_b|5|I0|5|0|0|0|
|hevc_b|5|P1_16|80|2.69041912|0.0392443644|2.65117476|
|hevc_b|5|P17_32|80|5.47634411|1.07003816|4.40630595|
|hevc_b|5|P33_48|80|5.24493515|1.08954673|4.15538842|
|hevc_b|5|P49_63|75|4.52586696|0.644110996|3.88175596|
|hevc_b|9|I0|5|0|0|0|
|hevc_b|9|P1_16|80|4.8116145|0.334728975|4.47688552|
|hevc_b|9|P17_32|80|3.17520694|0.542101352|2.63310558|
|hevc_b|9|P33_48|80|6.18033036|1.012334|5.16799636|
|hevc_b|9|P49_63|75|5.11257859|0.642940422|4.46963817|
|mcljcv|0|I0|30|0|0|0|
|mcljcv|0|P1_16|480|2.34441085|-0.120396523|2.46480737|
|mcljcv|0|P17_32|480|1.17094149|-0.33957861|1.5105201|
|mcljcv|0|P33_48|480|1.17706332|-0.112404814|1.28946814|
|mcljcv|0|P49_63|450|1.02966663|0.130907242|0.898759387|
|mcljcv|5|I0|30|0|0|0|
|mcljcv|5|P1_16|480|-0.178103362|-0.140826778|-0.037276584|
|mcljcv|5|P17_32|480|2.31630248|-0.0746275332|2.39093002|
|mcljcv|5|P33_48|480|2.08040914|-0.0266940621|2.1071032|
|mcljcv|5|P49_63|450|2.73384831|0.133741869|2.60010644|
|mcljcv|9|I0|30|0|0|0|
|mcljcv|9|P1_16|480|2.60041282|0.00493884312|2.59547397|
|mcljcv|9|P17_32|480|4.27382402|0.0124010925|4.26142293|
|mcljcv|9|P33_48|480|3.77338959|0.0497063428|3.72368325|
|mcljcv|9|P49_63|450|4.63231542|0.13642656|4.49588886|

## Leave-one-video-out sensitivity

Names below identify the extremes of the matched-QP pooled gap after one sequence is removed. This is sensitivity evidence, not an independent significance test or a gain comparison at equal rate.

|Dataset|QP|Smallest remaining gap|Excluded sequence|Largest remaining gap|Excluded sequence|
|---|---:|---:|---|---:|---|
|ulong|0|-2.55911928|f0f6a0ec-d531-4a1b-9d18-d7541bd0132d|1.02393857|3ed03b37-46cd-442c-a790-29fd5ed72122|
|ulong|5|-0.479661676|98881c99-ebda-4469-bbe4-0ee789ff217d|0.746049559|259bf3b4-4af4-4e1a-af3e-4b7df23dc061|
|ulong|9|1.01626841|98881c99-ebda-4469-bbe4-0ee789ff217d|1.69892097|259bf3b4-4af4-4e1a-af3e-4b7df23dc061|
|hevc_b|0|0.799284366|BQTerrace|8.13429798|Kimono1|
|hevc_b|5|1.55042899|BQTerrace|5.57974331|Kimono1|
|hevc_b|9|1.45279119|BQTerrace|5.47110615|ParkScene|
|mcljcv|0|-0.317706835|videoSRC06_1920x1080_25|1.33221787|videoSRC01_1920x1080_30|
|mcljcv|5|0.658722276|videoSRC12_1920x1080_30|1.52029388|videoSRC06_1920x1080_25|
|mcljcv|9|2.41759967|videoSRC06_1920x1080_25|3.23706097|videoSRC26_1920x1080_30|

## Structure and provenance

Spectra, GT-relative and direct original/B1 diagonal/off-diagonal differences, per-video mean shifts and within/between covariance closure are saved as separate numeric tables. Covariance additivity is not an additive attribution of covariance FID. The actual nine-checkpoint source chain is recorded, and18 available native-I intermediate points from V6.20/V6.21/V6.22 were matched. 首次出现阶段尚未定位：更早全部阶段的同输入、同 native-I 协议证据未建立，不能从版本号推断原因。

Both feature extractors received the identical floating RGB tensor, including directly decoded reconstruction (no reconstruction PNG quantization substitution). Same canonical GT sources and fixed feature weights were retained. No test data were used to select statistical layers, coefficients or budgets.
