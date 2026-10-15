# v6.33 pre-bottleneck coding mechanism audit

Complete: 3220/3220 evaluation points, 46 videos, 64 causal frames, QP0–9. Optimizer updates: 0.

1. Does P change actual bitstream rate? Fraction of alpha1/control video-QP pairs with changed total bytes/rate: 1.000000.
2. Strength/rate: alpha1 vs alpha0 relative bpp mean 0.0310402, median 0.0279703; video-cluster mean95% CI [0.0161706, 0.046885]. Fraction lower: 0.226087. Alpha trends are in trend_summary.csv.
3. Teacher distance: mean alpha1-minus-alpha0 cosine distance 0.009787454; relative to paper_compatible_teacher_proxy only. Positive is worse.
4. Perceptual quality: alpha sweep values below are local interventions without receiver readaptation. System-level B1_alpha100 versus Original common-rate results follow separately.
5. Teacher worse but rate lower fraction: 0.221739. This is paired by video/external QP, not an equal-rate quality comparison.
6. Quantization/entropy path: mean alpha1/control changed-symbol fraction 0.00298675. Real raw y, dequantized y_hat, actual centered integer symbols and actual RANS CDF entries were hooked; see tensor_semantics.json. CDF entry bits exclude escape bypass and hyperlatent z.
7. Registered mechanism decision: CASE_B_partial. This tests whether a representation with better prescribed latent fidelity is necessarily more efficient for the fixed finite-rate codec; it does not rank all possible representations.
8. Unavailable/unsupported claims: no exact official training teacher, no fair Strong No-P baseline, no optimal representation claim, no causal channel attribution, no claim that latent supervision is wrong. RGB prediction residual is unavailable because the architecture provides conditional context rather than an explicitly defined source prediction.

## Alpha1 versus alpha0 by dataset (same QP; not equal-rate quality)
```csv
dataset,relative_delta_bpp_vs_alpha0,delta_cos_vs_alpha0,delta_LPIPS_vs_alpha0,delta_DISTS_vs_alpha0,delta_FloLPIPS_vs_alpha0,fraction_quantized_symbols_changed
hevc_b,0.07631981,0.009837247,-0.04194128,-0.01278994,-0.05810522,0.003390869
mcljcv,0.03584101,0.009267093,-0.03144879,-0.01291007,-0.03028166,0.003096278
ulong,-0.01256878,0.01211135,-0.02138852,-0.006634658,-0.02167011,0.00216687
uvg_holdout,0.008496343,0.009422158,-0.01077627,-0.0003178414,-0.01426184,0.003433946
uvg_validation,0.05457521,0.007288716,-0.01588807,-0.003438818,-0.01071134,0.003344765

```

## Normal B1 versus Original matched-rate comparison
100 equally spaced log-bpp samples inside each video common measured interval; PCHIP without extrapolation; video-cluster confidence intervals.
```csv
dataset_scope,metric,expected_videos,available_videos,n_videos,n_pairs,mean,median,mean_ci_low,mean_ci_high,median_ci_low,median_ci_high,bootstrap,single_video_CI_degenerate
all,DISTS,46,46,46,46,-0.005405487,-0.003256042,-0.008555458,-0.002218694,-0.007656767,-0.001655804,2000 video-cluster resamples; mean equal video weights; median pooled pairs,False
all,FloLPIPS,46,46,46,46,-0.01248899,-0.009651597,-0.0167585,-0.007981155,-0.01216723,-0.006857541,2000 video-cluster resamples; mean equal video weights; median pooled pairs,False
all,LPIPS,46,46,46,46,-0.01123431,-0.007756724,-0.01504399,-0.007454688,-0.01163457,-0.005550494,2000 video-cluster resamples; mean equal video weights; median pooled pairs,False
hevc_b,DISTS,5,5,5,5,-0.006307995,-0.002955002,-0.01935166,0.003953586,-0.03231838,0.008017263,2000 video-cluster resamples; mean equal video weights; median pooled pairs,False
hevc_b,FloLPIPS,5,5,5,5,-0.02007318,-0.01612909,-0.03753702,-0.002609337,-0.04632788,0.01222094,2000 video-cluster resamples; mean equal video weights; median pooled pairs,False
hevc_b,LPIPS,5,5,5,5,-0.01253566,-0.0100568,-0.02923374,0.002032478,-0.04120887,0.01112943,2000 video-cluster resamples; mean equal video weights; median pooled pairs,False
mcljcv,DISTS,30,30,30,30,-0.00346155,-0.002439128,-0.007391119,0.0002225512,-0.005162192,0.001856929,2000 video-cluster resamples; mean equal video weights; median pooled pairs,False
mcljcv,FloLPIPS,30,30,30,30,-0.01053516,-0.00705016,-0.01658402,-0.005001564,-0.01191294,-0.005565414,2000 video-cluster resamples; mean equal video weights; median pooled pairs,False
mcljcv,LPIPS,30,30,30,30,-0.009753752,-0.006301066,-0.01435147,-0.005622224,-0.01239666,-0.003490408,2000 video-cluster resamples; mean equal video weights; median pooled pairs,False
ulong,DISTS,8,8,8,8,-0.01066308,-0.01240062,-0.01582126,-0.005381078,-0.01608455,-0.001272175,2000 video-cluster resamples; mean equal video weights; median pooled pairs,False
ulong,FloLPIPS,8,8,8,8,-0.01536355,-0.01270367,-0.0222512,-0.008573942,-0.02962911,-0.009398193,2000 video-cluster resamples; mean equal video weights; median pooled pairs,False
ulong,LPIPS,8,8,8,8,-0.01591592,-0.009963324,-0.02697989,-0.006641508,-0.03485585,-0.006912553,2000 video-cluster resamples; mean equal video weights; median pooled pairs,False
uvg_holdout,DISTS,2,2,2,2,-0.01106956,-0.01106956,-0.0148929,-0.007246228,-0.0148929,-0.007246228,2000 video-cluster resamples; mean equal video weights; median pooled pairs,False
uvg_holdout,FloLPIPS,2,2,2,2,-0.00929382,-0.00929382,-0.01215516,-0.006432483,-0.01215516,-0.006432483,2000 video-cluster resamples; mean equal video weights; median pooled pairs,False
uvg_holdout,LPIPS,2,2,2,2,-0.01126283,-0.01126283,-0.01636522,-0.006160438,-0.01636522,-0.006160438,2000 video-cluster resamples; mean equal video weights; median pooled pairs,False
uvg_validation,DISTS,1,1,1,1,-0.005822153,-0.005822153,-0.005822153,-0.005822153,-0.005822153,-0.005822153,2000 video-cluster resamples; mean equal video weights; median pooled pairs,True
uvg_validation,FloLPIPS,1,1,1,1,-0.01657677,-0.01657677,-0.01657677,-0.01657677,-0.01657677,-0.01657677,2000 video-cluster resamples; mean equal video weights; median pooled pairs,True
uvg_validation,LPIPS,1,1,1,1,-0.01163457,-0.01163457,-0.01163457,-0.01163457,-0.01163457,-0.01163457,2000 video-cluster resamples; mean equal video weights; median pooled pairs,True

```

## Full alpha sweep
P-frame quality/fidelity means; rate includes all 64 frames and all headers. Original is separate. QP summaries remain separate in alpha_summary.csv.
```csv
dataset,method,actual_bpp,P_frame_bpp,cosine,LPIPS,DISTS,FloLPIPS
ulong,Original,0.007381558,0.006866854,0.3704377,0.1209149,0.08771264,0.1144847
ulong,B1_alpha0,0.007381558,0.006866854,0.3757037,0.1264754,0.08385156,0.1203464
ulong,B1_alpha025,0.007254202,0.006737476,0.376401,0.1158929,0.07820433,0.1103952
ulong,B1_alpha050,0.007172694,0.006654675,0.3787446,0.1088447,0.07517505,0.1031361
ulong,B1_alpha075,0.007123025,0.006604217,0.3826554,0.1052428,0.07493878,0.09923709
ulong,B1_alpha100,0.00709576,0.00657652,0.3878151,0.1050868,0.0772169,0.09867627
ulong,B1_alpha125,0.007073263,0.006553666,0.3940099,0.1080579,0.08135956,0.1015365
uvg_holdout,Original,0.01387301,0.01336656,0.3729834,0.1540519,0.1006079,0.1572864
uvg_holdout,B1_alpha0,0.01387301,0.01336656,0.3778015,0.1526726,0.08947801,0.1612443
uvg_holdout,B1_alpha025,0.01382922,0.01332207,0.3787424,0.1456943,0.0863137,0.1532803
uvg_holdout,B1_alpha050,0.01380093,0.01329333,0.3806225,0.1416211,0.08565268,0.1481637
uvg_holdout,B1_alpha075,0.01377983,0.01327189,0.3835264,0.1404351,0.08672147,0.1460793
uvg_holdout,B1_alpha100,0.01376837,0.01326025,0.3872236,0.1418964,0.08916017,0.1469825
uvg_holdout,B1_alpha125,0.01375708,0.01324878,0.3917042,0.145923,0.09300926,0.1505707
uvg_validation,Original,0.009817968,0.009321778,0.3871798,0.1173535,0.07693194,0.1100266
uvg_validation,B1_alpha0,0.009817968,0.009321778,0.3920761,0.1191826,0.07362971,0.1017562
uvg_validation,B1_alpha025,0.009890492,0.009395454,0.3913779,0.1103403,0.0678972,0.09537916
uvg_validation,B1_alpha050,0.009995497,0.009502125,0.3924023,0.1045656,0.06550636,0.09133933
uvg_validation,B1_alpha075,0.01012032,0.009628932,0.3950578,0.1022781,0.06636525,0.09014781
uvg_validation,B1_alpha100,0.01026347,0.009774355,0.3993648,0.1032946,0.07019089,0.09104483
uvg_validation,B1_alpha125,0.01040787,0.009921045,0.4050409,0.1078838,0.07694307,0.09501736
hevc_b,Original,0.009759072,0.00920546,0.3730442,0.1977623,0.09534509,0.198227
hevc_b,B1_alpha0,0.009759072,0.00920546,0.3798543,0.224117,0.1004565,0.2348028
hevc_b,B1_alpha025,0.00984185,0.009289551,0.3797971,0.2058578,0.09358995,0.2121183
hevc_b,B1_alpha050,0.009952409,0.009401865,0.3814748,0.1929659,0.08922291,0.1956184
hevc_b,B1_alpha075,0.01009287,0.009544556,0.384777,0.1851267,0.08740224,0.1838245
hevc_b,B1_alpha100,0.01025041,0.009704601,0.3896915,0.1821757,0.08766659,0.1766975
hevc_b,B1_alpha125,0.01038022,0.009836471,0.3953493,0.183187,0.08968904,0.1739259
mcljcv,Original,0.0103036,0.009838135,0.4037071,0.160039,0.1011748,0.1585978
mcljcv,B1_alpha0,0.0103036,0.009838135,0.4095731,0.1799441,0.1099216,0.1766265
mcljcv,B1_alpha025,0.01031448,0.009849185,0.4092177,0.1664511,0.1034322,0.1643795
mcljcv,B1_alpha050,0.0103672,0.009902742,0.4105466,0.1566007,0.09917862,0.1551593
mcljcv,B1_alpha075,0.01044315,0.009979901,0.4137925,0.1506484,0.09705795,0.1492457
mcljcv,B1_alpha100,0.01053763,0.01007588,0.4188402,0.1484953,0.09701153,0.1463448
mcljcv,B1_alpha125,0.01062683,0.01016649,0.4256607,0.149919,0.09921267,0.1465605

```

B1_alpha0 retains the B1 bridge/generator. It is a strict source-side rate/coding intervention control, but its end-to-end quality can include receiver-interface mismatch. This experiment does not exclude insufficient baseline training.

Quantizer boundaries are half-integers only after the actual learned quantization scaling and actual conditional mean centering. The scale<=0.12 skip rule and saturation are recorded separately. y_hat restores conditional means and quantization scale; it is not itself the integer transmitted symbol array. All mechanisms are measured with full causal histories, so changes include propagated changes in codec context.

Teacher targets are computed from original GT and never from P_alpha(x). Native I0 bypasses P at every alpha. Actual total bytes include SPS/headers/I0; P-frame rate uses the 63 P-frame pixels as its denominator. 20fps changes only kbps display. Low/mid/high frequency energy follows the fixed v6.31a bands and is not causal attribution.

Figures are provided as PNG and PDF in figures/. All points have server-side receipts, also bundled in point_receipts.jsonl.gz. Binary streams, full teacher/codec tensors and large logs remain on the server.
