# v6.43 objective results

This is an offline GT-assisted oracle experiment on the fixed 64-frame ParkScene and Jockey clips. Only the v6.40 G_basic step100 Generator is used with the original frozen compression core and Bridge. New optimizer updates: **0**. Native I0, P1–P63, no extra reference reset; input 1920×1080, padded to 1920×1088; FP16 codec/receiver, FP32 metric inputs, force_zero_thres=0.12. ParkScene retains its native timeline; Jockey retains the canonical 30fps RGB and original mapping in data_manifest.json.

## Reproduction and integrity

Both newly encoded baseline streams reproduce v6.42 byte counts and SHA256 exactly. Independent G_basic reconstruction hashes, every frame metric, every FloLPIPS transition and aggregate metrics match exactly. Each intervention contains its own full RANS stream with embedded actual QPs; every independent decoder uses those fields. Prefix packet, reconstruction, receiver-input and reference-state hashes are identical before the first changed frame. Encoding and decoding latent/reference features match exactly on all 63 P frames. Weights, buffers and metric networks are unchanged. All frame packets include associated headers (initial SPS assigned to I0); packet sums equal file sizes.

| baselines_complete | impulses_complete | transfers_complete | total_streams | frame_records | transition_records |
| --- | --- | --- | --- | --- | --- |
| 2 | 24 | 16 | 42 | 2688 | 2646 |

## Whole-stream measured values

| sequence | variant | total_bits | actual_bpp | LPIPS | DISTS | L1 | PSNR | FloLPIPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ParkScene | baseline | 977264 | 0.0073638841 | 0.21537013 | 0.097027352 | 0.038031563 | 24.72731 | 0.19696641 |
| ParkScene | t08_qplus2 | 984200 | 0.0074161482 | 0.21525206 | 0.098287903 | 0.038093433 | 24.703493 | 0.19679809 |
| ParkScene | t08_qminus2 | 987824 | 0.0074434558 | 0.21382847 | 0.096597107 | 0.038012669 | 24.72825 | 0.19607606 |
| ParkScene | t16_qplus2 | 956344 | 0.0072062476 | 0.21631392 | 0.098633646 | 0.03811999 | 24.70094 | 0.19714637 |
| ParkScene | t16_qminus2 | 968344 | 0.00729667 | 0.21562643 | 0.097821043 | 0.038141102 | 24.694214 | 0.19672503 |
| ParkScene | t24_qplus2 | 956656 | 0.0072085986 | 0.21637455 | 0.098212114 | 0.038167456 | 24.692474 | 0.19792756 |
| ParkScene | t24_qminus2 | 962216 | 0.0072504943 | 0.21576356 | 0.098213066 | 0.038201811 | 24.684926 | 0.19749361 |
| ParkScene | t32_qplus2 | 961088 | 0.0072419946 | 0.21650031 | 0.098491866 | 0.038115188 | 24.701569 | 0.1977205 |
| ParkScene | t32_qminus2 | 961944 | 0.0072484447 | 0.21568775 | 0.098183937 | 0.038155549 | 24.698693 | 0.19729017 |
| ParkScene | t40_qplus2 | 958456 | 0.0072221619 | 0.21611321 | 0.097788238 | 0.038125217 | 24.704491 | 0.1971961 |
| ParkScene | t40_qminus2 | 963328 | 0.0072588735 | 0.21622907 | 0.097781097 | 0.038159387 | 24.698844 | 0.19721889 |
| ParkScene | t48_qplus2 | 961504 | 0.0072451292 | 0.21587095 | 0.097343257 | 0.038047919 | 24.71452 | 0.19705743 |
| ParkScene | t48_qminus2 | 960344 | 0.0072363884 | 0.2161271 | 0.097289112 | 0.038164094 | 24.6956 | 0.1975085 |
| Jockey | baseline | 3015576 | 0.022722982 | 0.16003031 | 0.084430344 | 0.034584384 | 26.788003 | 0.15826849 |
| Jockey | t08_qplus2 | 3041400 | 0.022917571 | 0.15967985 | 0.084009534 | 0.034489478 | 26.814587 | 0.15773529 |
| Jockey | t08_qminus2 | 2990728 | 0.022535747 | 0.16105514 | 0.084560046 | 0.034740812 | 26.743634 | 0.15852761 |
| Jockey | t16_qplus2 | 3040144 | 0.022908107 | 0.15970739 | 0.084208732 | 0.034496824 | 26.809541 | 0.15746412 |
| Jockey | t16_qminus2 | 2996448 | 0.022578848 | 0.16094356 | 0.084481967 | 0.034760837 | 26.735441 | 0.15928737 |
| Jockey | t24_qplus2 | 3036360 | 0.022879593 | 0.15966994 | 0.084141156 | 0.034492444 | 26.813778 | 0.15738558 |
| Jockey | t24_qminus2 | 2996992 | 0.022582948 | 0.16102483 | 0.084662085 | 0.034781989 | 26.73238 | 0.15974277 |
| Jockey | t32_qplus2 | 3033384 | 0.022857169 | 0.15974945 | 0.084204705 | 0.034515664 | 26.807628 | 0.15821453 |
| Jockey | t32_qminus2 | 2996504 | 0.02257927 | 0.16089637 | 0.084642763 | 0.034715026 | 26.752047 | 0.15815984 |
| Jockey | t40_qplus2 | 3035288 | 0.022871516 | 0.15937971 | 0.084104 | 0.034500087 | 26.812943 | 0.15692633 |
| Jockey | t40_qminus2 | 2996096 | 0.022576196 | 0.16102667 | 0.084788869 | 0.034700414 | 26.752118 | 0.15960268 |
| Jockey | t48_qplus2 | 3034432 | 0.022865066 | 0.15939625 | 0.084310832 | 0.034478239 | 26.81718 | 0.15626973 |
| Jockey | t48_qminus2 | 2996936 | 0.022582526 | 0.16109539 | 0.084793756 | 0.034713209 | 26.752771 | 0.15934945 |
| ParkScene | forward_r8_d32_a1_b1 | 959352 | 0.0072289135 | 0.21608386 | 0.098373454 | 0.038221998 | 24.677857 | 0.19720236 |
| ParkScene | forward_r8_d32_a1_b2 | 952448 | 0.0071768904 | 0.21657263 | 0.098566292 | 0.038305174 | 24.661269 | 0.19676215 |
| ParkScene | forward_r8_d32_a2_b1 | 978184 | 0.0073708165 | 0.21527326 | 0.098032662 | 0.038169315 | 24.689167 | 0.19673677 |
| ParkScene | forward_r8_d32_a2_b2 | 969808 | 0.0073077016 | 0.21593499 | 0.098993104 | 0.038270563 | 24.665489 | 0.19713766 |
| ParkScene | reverse_r8_d32_a1_b1 | 987544 | 0.007441346 | 0.21485231 | 0.097578492 | 0.037997003 | 24.73335 | 0.19663124 |
| ParkScene | reverse_r8_d32_a1_b2 | 970776 | 0.0073149957 | 0.21556059 | 0.097955622 | 0.038087116 | 24.709318 | 0.19711274 |
| ParkScene | reverse_r8_d32_a2_b1 | 988504 | 0.0074485798 | 0.21398682 | 0.09721297 | 0.037979532 | 24.734326 | 0.19667519 |
| ParkScene | reverse_r8_d32_a2_b2 | 969968 | 0.0073089072 | 0.2153738 | 0.097465564 | 0.038079179 | 24.711279 | 0.19722484 |
| Jockey | forward_r48_d8_a1_b1 | 3010208 | 0.022682533 | 0.16049313 | 0.084520696 | 0.034639449 | 26.777483 | 0.15755229 |
| Jockey | forward_r48_d8_a1_b2 | 2998080 | 0.022591146 | 0.16065832 | 0.084558968 | 0.034691781 | 26.755677 | 0.15916566 |
| Jockey | forward_r48_d8_a2_b1 | 3021336 | 0.022766385 | 0.1600572 | 0.084538952 | 0.034574386 | 26.792186 | 0.15724512 |
| Jockey | forward_r48_d8_a2_b2 | 3010088 | 0.022681629 | 0.16048029 | 0.084386886 | 0.034663034 | 26.760746 | 0.15774255 |
| Jockey | reverse_r48_d8_a1_b1 | 3017256 | 0.022735641 | 0.16036017 | 0.084104923 | 0.034614414 | 26.781798 | 0.15828183 |
| Jockey | reverse_r48_d8_a1_b2 | 3031696 | 0.022844449 | 0.16027109 | 0.084357294 | 0.034565784 | 26.7912 | 0.15887424 |
| Jockey | reverse_r48_d8_a2_b1 | 3007568 | 0.02266264 | 0.16100297 | 0.08455437 | 0.034647537 | 26.772265 | 0.15912178 |
| Jockey | reverse_r48_d8_a2_b2 | 3022616 | 0.02277603 | 0.1608276 | 0.084605396 | 0.034623088 | 26.773792 | 0.15920668 |

PSNR above is in dB from aggregate RGB SSE, not the mean of frame dB values. Lower LPIPS, DISTS, L1 and FloLPIPS is better. The unaggregated frame and transition measurements are in all_frames.csv and all_transitions.csv.

## Fixed impulse windows and rankings

impulse_summary.csv contains current, future15, full 16-frame, t…63 and complete-stream gains/costs, and cumulative cutoffs t, t+4, t+8, t+15. Gains are baseline minus variant; positive denotes lower distortion. Costs are variant minus baseline in actual bits. FloLPIPS gains are reported separately by transition in transition_differences.csv.

| sequence | target | label | G_now | G_future15 | G_16 | delta_bits_now | delta_bits_future15 | delta_bits_16 | U_now | U_16 | cost_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ParkScene | 8 | qplus2 | 0.0098263323 | 0.0092636496 | 0.019089982 | 21168 | -14016 | 7152 | 0.0013739279 | 0.0026691809 | positive_full_and_window |
| ParkScene | 8 | qminus2 | -0.0050085783 | 0.045731455 | 0.040722877 | -7464 | 19272 | 11808 | -0.00042416822 | 0.0034487531 | positive_full_and_window |
| ParkScene | 16 | qplus2 | 0.01100345 | -0.044177622 | -0.033174172 | 24880 | -42680 | -17800 | None | None | nonpositive_window |
| ParkScene | 16 | qminus2 | -0.011479393 | -0.027214766 | -0.038694158 | -15728 | 6432 | -9296 | None | None | nonpositive_window |
| ParkScene | 24 | qplus2 | 0.0094164312 | -0.045036033 | -0.035619602 | 22176 | -44200 | -22024 | None | None | nonpositive_window |
| ParkScene | 24 | qminus2 | -0.0087908655 | -0.030386075 | -0.039176941 | -18392 | 3032 | -15360 | None | None | nonpositive_window |
| ParkScene | 32 | qplus2 | 0.0081122369 | -0.055323437 | -0.0472112 | 21816 | -42688 | -20872 | None | None | nonpositive_window |
| ParkScene | 32 | qminus2 | -0.0079910606 | -0.026086733 | -0.034077793 | -18400 | 304 | -18096 | None | None | nonpositive_window |
| ParkScene | 40 | qplus2 | 0.0091838986 | -0.035755232 | -0.026571333 | 22600 | -43128 | -20528 | None | None | nonpositive_window |
| ParkScene | 40 | qminus2 | -0.0089398026 | -0.040092364 | -0.049032167 | -18784 | 2624 | -16160 | None | None | nonpositive_window |
| ParkScene | 48 | qplus2 | 0.0074646473 | -0.039517075 | -0.032052428 | 23376 | -39136 | -15760 | None | None | nonpositive_window |
| ParkScene | 48 | qminus2 | -0.0094195157 | -0.039026648 | -0.048446164 | -19288 | 2368 | -16920 | None | None | nonpositive_window |
| Jockey | 8 | qplus2 | 0.022739843 | 0.0045758337 | 0.027315676 | 27184 | 712 | 27896 | 0.00081516499 | 0.00097919688 | positive_full_and_window |
| Jockey | 8 | qminus2 | -0.034542918 | -0.026860744 | -0.061403662 | -15600 | -9552 | -25152 | None | None | nonpositive_window |
| Jockey | 16 | qplus2 | 0.027025908 | 0.00084911287 | 0.027875021 | 24520 | 192 | 24712 | 0.001093635 | 0.0011279954 | positive_full_and_window |
| Jockey | 16 | qminus2 | -0.03414844 | -0.020288289 | -0.054436728 | -13776 | -7312 | -21088 | None | None | nonpositive_window |
| Jockey | 24 | qplus2 | 0.027204752 | -0.0030028373 | 0.024201915 | 24216 | -3736 | 20480 | 0.001328357 | 0.0011817341 | positive_full_and_window |
| Jockey | 24 | qminus2 | -0.040725619 | -0.022540346 | -0.063265964 | -14056 | -6416 | -20472 | None | None | nonpositive_window |
| Jockey | 32 | qplus2 | 0.023895368 | -0.00043106079 | 0.023464307 | 22264 | -2920 | 19344 | 0.0012352858 | 0.0012130018 | positive_full_and_window |
| Jockey | 32 | qminus2 | -0.038224757 | -0.021607727 | -0.059832484 | -12968 | -6456 | -19424 | None | None | nonpositive_window |
| Jockey | 40 | qplus2 | 0.029982701 | 0.013362899 | 0.0433456 | 22520 | -2016 | 20504 | 0.0014622855 | 0.002114007 | positive_full_and_window |
| Jockey | 40 | qminus2 | -0.048733115 | -0.012288198 | -0.061021313 | -14912 | -4456 | -19368 | None | None | nonpositive_window |
| Jockey | 48 | qplus2 | 0.036275014 | 0.0043046921 | 0.040579706 | 22792 | -3936 | 18856 | 0.0019237916 | 0.0021520846 | positive_full_and_window |
| Jockey | 48 | qminus2 | -0.053375244 | -0.014789969 | -0.068165213 | -14912 | -3728 | -18640 | None | None | nonpositive_window |

U_now and U_16 share delta_bits_16 as denominator, in cumulative LPIPS decrease/kbit. Positive-cost ranks require positive total added bits and positive window bits. Nonpositive-cost operations retain raw gains and costs, with no fabricated efficiency. Supplemental DISTS/L1 results use the same rules.

| sequence | metric | n | now | future | top_two_set_same | top_two_order_same | pairwise_reversals | Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ParkScene | LPIPS | 2 | t08_qplus2, t08_qminus2 | t08_qminus2, t08_qplus2 | True | False | 1 | None |
| ParkScene | DISTS | 2 | t08_qplus2, t08_qminus2 | t08_qminus2, t08_qplus2 | True | False | 1 | None |
| ParkScene | L1 | 2 | t08_qplus2, t08_qminus2 | t08_qplus2, t08_qminus2 | True | True | 0 | None |
| Jockey | LPIPS | 6 | t48_qplus2, t40_qplus2, t24_qplus2, t32_qplus2, t16_qplus2, t08_qplus2 | t48_qplus2, t40_qplus2, t32_qplus2, t24_qplus2, t16_qplus2, t08_qplus2 | True | True | 1 | 0.94285714 |
| Jockey | DISTS | 6 | t48_qplus2, t32_qplus2, t24_qplus2, t08_qplus2, t40_qplus2, t16_qplus2 | t32_qplus2, t24_qplus2, t40_qplus2, t16_qplus2, t08_qplus2, t48_qplus2 | False | False | 7 | -0.028571429 |
| Jockey | L1 | 6 | t48_qplus2, t32_qplus2, t40_qplus2, t16_qplus2, t24_qplus2, t08_qplus2 | t48_qplus2, t40_qplus2, t16_qplus2, t32_qplus2, t24_qplus2, t08_qplus2 | False | False | 2 | 0.82857143 |

Spearman values are descriptive only. Full pairwise reversal lists are in ranking_comparison.json.

## Fixed budget transfer

Selection uses qplus2 receivers and qminus2 donors with the specified actual-cost signs and |r−d|≥16; maximize A(r)−C(d), then lexicographic (r,d). All pairs and raw scores are in transfer_selection.json. The selected pairs and finite grids are used even if the maximum score is negative.

| sequence | status | r | d | A | C | score | reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ParkScene | SELECTED | 8 | 32 | 2.6691809e-06 | 1.8831672e-06 | 7.8601373e-07 |  |
| Jockey | SELECTED | 48 | 8 | 2.1520846e-06 | 2.4413034e-06 | -2.892188e-07 |  |

| sequence | direction | a | b | total_bits | budget_ratio | strict_feasible | near_budget | saved_percent | LPIPS | delta_LPIPS | delta_DISTS | delta_L1 | delta_PSNR_dB | delta_FloLPIPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ParkScene | forward | 1 | 1 | 959352 | 0.98167128 | True | False | 1.8328722 | 0.21608386 | 0.00071372534 | 0.001346102 | 0.00019043515 | -0.049453813 | 0.00023595681 |
| ParkScene | forward | 1 | 2 | 952448 | 0.97460666 | True | False | 2.5393343 | 0.21657263 | 0.0012024962 | 0.0015389398 | 0.00027361099 | -0.066040952 | -0.00020426086 |
| ParkScene | forward | 2 | 1 | 978184 | 1.0009414 | False | False | -0.094140376 | 0.21527326 | -9.6871983e-05 | 0.0010053096 | 0.00013775245 | -0.038143404 | -0.00022963706 |
| ParkScene | forward | 2 | 2 | 969808 | 0.99237054 | True | True | 0.76294635 | 0.21593499 | 0.00056486065 | 0.0019657519 | 0.00023900002 | -0.061821686 | 0.00017124864 |
| ParkScene | reverse | 1 | 1 | 987544 | 1.0105192 | False | False | -1.0519164 | 0.21485231 | -0.00051781954 | 0.00055113994 | -3.45601e-05 | 0.0060390975 | -0.00033516756 |
| ParkScene | reverse | 1 | 2 | 970776 | 0.99336106 | True | True | 0.6638943 | 0.21556059 | 0.0001904564 | 0.0009282697 | 5.5553333e-05 | -0.017992074 | 0.00014632988 |
| ParkScene | reverse | 2 | 1 | 988504 | 1.0115015 | False | False | -1.1501498 | 0.21398682 | -0.0013833146 | 0.00018561818 | -5.2030722e-05 | 0.0070155583 | -0.00029122096 |
| ParkScene | reverse | 2 | 2 | 969968 | 0.99253426 | True | True | 0.74657411 | 0.2153738 | 3.6628917e-06 | 0.00043821242 | 4.761637e-05 | -0.01603179 | 0.00025842966 |
| Jockey | forward | 1 | 1 | 3010208 | 0.99821991 | True | True | 0.17800911 | 0.16049313 | 0.00046281959 | 9.0351328e-05 | 5.5064622e-05 | -0.010520062 | -0.00071619841 |
| Jockey | forward | 1 | 2 | 2998080 | 0.99419812 | True | True | 0.58018767 | 0.16065832 | 0.0006280106 | 0.00012862403 | 0.00010739721 | -0.03232572 | 0.00089716923 |
| Jockey | forward | 2 | 1 | 3021336 | 1.0019101 | False | False | -0.19100828 | 0.1600572 | 2.688379e-05 | 0.00010860711 | -9.9980971e-06 | 0.0041826841 | -0.0010233657 |
| Jockey | forward | 2 | 2 | 3010088 | 0.99818012 | True | True | 0.18198845 | 0.16048029 | 0.00044997735 | -4.3458305e-05 | 7.8650512e-05 | -0.027256736 | -0.00052594252 |
| Jockey | reverse | 1 | 1 | 3017256 | 1.0005571 | False | False | -0.05571075 | 0.16036017 | 0.00032986049 | -0.0003254218 | 3.0030205e-05 | -0.0062053847 | 1.3336894e-05 |
| Jockey | reverse | 1 | 2 | 3031696 | 1.0053456 | False | False | -0.53455791 | 0.16027109 | 0.00024077762 | -7.3050149e-05 | -1.8600287e-05 | 0.0031971445 | 0.00060574888 |
| Jockey | reverse | 2 | 1 | 3007568 | 0.99734445 | True | True | 0.26555457 | 0.16100297 | 0.00097265723 | 0.00012402609 | 6.315278e-05 | -0.015738187 | 0.00085329383 |
| Jockey | reverse | 2 | 2 | 3022616 | 1.0023345 | False | False | -0.23345457 | 0.1608276 | 0.00079728663 | 0.00017505139 | 3.8704049e-05 | -0.014210923 | 0.00093819473 |

Within each direction, display selections use minimum whole LPIPS among candidates in [0.99 B0,B0], with ties broken by lower bits then (a,b).

| sequence | direction | status | a | b | total_bits | budget_ratio | delta_LPIPS | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ParkScene | forward | SELECTED | 2 | 2 | 969808 | 0.99237054 | 0.00056486065 |  |
| ParkScene | reverse | SELECTED | 2 | 2 | 969968 | 0.99253426 | 3.6628917e-06 |  |
| Jockey | forward | SELECTED | 2 | 2 | 3010088 | 0.99818012 | 0.00044997735 |  |
| Jockey | reverse | SELECTED | 2 | 1 | 3007568 | 0.99734445 | 0.00097265723 |  |

Candidates below 0.99 B0 remain recorded with their actual savings; they are not exact matched-rate comparisons. Candidates above B0 cannot establish fixed-budget gains. No padding bytes or stream truncation is used.

## Figures and raw materials

- [Jockey_gain_cost.png](figures/Jockey_gain_cost.png)
- [Jockey_t08_impulse.png](figures/Jockey_t08_impulse.png)
- [Jockey_t16_impulse.png](figures/Jockey_t16_impulse.png)
- [Jockey_t24_impulse.png](figures/Jockey_t24_impulse.png)
- [Jockey_t32_impulse.png](figures/Jockey_t32_impulse.png)
- [Jockey_t40_impulse.png](figures/Jockey_t40_impulse.png)
- [Jockey_t48_impulse.png](figures/Jockey_t48_impulse.png)
- [Jockey_transfer.png](figures/Jockey_transfer.png)
- [ParkScene_gain_cost.png](figures/ParkScene_gain_cost.png)
- [ParkScene_t08_impulse.png](figures/ParkScene_t08_impulse.png)
- [ParkScene_t16_impulse.png](figures/ParkScene_t16_impulse.png)
- [ParkScene_t24_impulse.png](figures/ParkScene_t24_impulse.png)
- [ParkScene_t32_impulse.png](figures/ParkScene_t32_impulse.png)
- [ParkScene_t40_impulse.png](figures/ParkScene_t40_impulse.png)
- [ParkScene_t48_impulse.png](figures/ParkScene_t48_impulse.png)
- [ParkScene_transfer.png](figures/ParkScene_transfer.png)

All bitstreams are small and included in bitstreams/. stream_manifest.json provides paths, exact sizes and SHA256. Each task directory contains frames.csv, transitions.csv, proofs.json, audits.json and a signed completion receipt.

## Limits

Same QP adjustment does not imply the same added bit count. Single-frame intervention gain is not a fixed-total-rate gain. QP changes encoding features and Generator conditions together; these are overall QP interventions, not isolated latent effects. Multiple intervention positions within these two videos are not independent video samples. Selection uses the same video GT and is oracle-only; it is not a deployable allocation algorithm or a generalization/statistical-significance result. No mechanism, innovation or subjective image-quality conclusion is drawn. No model training, FID, BD-rate or speed test was performed.
