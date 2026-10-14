# v6.32 execution record

Status: BLOCKED. Actual committed updates: {'A_C1_lr1': 1000, 'B_C1_lr2': 1000, 'C_C1_lr2_FD': 0, 'D_C1_lr2_mean_CORAL': 0}.
Physical optimizer calls and replay/discard accounting: {'A_C1_lr1': {'physical_optimizer_calls': 1000, 'committed_updates': 1000, 'replayed_or_discarded_calls': 0}, 'B_C1_lr2': {'physical_optimizer_calls': 1000, 'committed_updates': 1000, 'replayed_or_discarded_calls': 0}, 'C_C1_lr2_FD': {'physical_optimizer_calls': 0, 'committed_updates': 0, 'replayed_or_discarded_calls': 0}, 'D_C1_lr2_mean_CORAL': {'physical_optimizer_calls': 0, 'committed_updates': 0, 'replayed_or_discarded_calls': 0}}.
Logical evaluation: 2116/3312; new encodes: 1196/2392; baseline aliases: 920/920.
B1 full-frame prefill: 996/1000. Calibration: 0/40 (blocked).

The four fixed-plan source windows listed in audits/full_resolution_conflicts.json have native heights 960, 1038, 810, 816 instead of the required 1080. No resize, padding substitution, frame or plan replacement was performed. A/B crop C1 optimization continues independently; C/D, complete seeded EMA and their comparisons remain unavailable.

Source: {'path': '/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_22_long_reference_reconstruction_loss_control/branches/B1_reconstruction/checkpoints/adaptation_step_1000.pt', 'sha256': 'dc76f0e221dc7aaa5e5f45f648722584ce226e65828ce24151f9961985f3f6f7', 'module_hashes': {'wrapper': '08cfe75a5292c9ad874c6ab593a418fbc425c5ac9182ba4edaf4f779a441d0a9', 'bridge': '3af207756b6289c56a5c9ac4a4744b519e5469d467968cd5834b03dec22f6c1e', 'generator': 'cc93348fce3e49d9adf5f0afcce38f9c9d571a4ecb4b7fbb171ed7b8cbfacd37'}, 'compression_hash': '0f0c0e2358ceec024230154da4777ed5c444a8382af1ec9d7165f3d7886141cf'}.
Objective: fused LPIPS + .5 DISTS + beta(q)*rate + .01 proxy; separately restored 4096 reconstruction/stat gradients, fixed weights .1383589248954108 and .038636700918173825; supervised P49–63 mean /15.
A B/G LR: 3e-6,5e-7. B: 6e-6,1e-6. Fresh AdamW (.9,.999), eps1e-8, decay1e-4, global clip1. P/core/I/quality frozen. D updates zero.

250-step 3-QP diagnostic tables and 1000-step 10-QP main tables are separate. Common six-method intervals and unavailable rows are preserved; no missing branch is dropped from a claimed full average. FID uses each fixed dataset pool, float64 ddof1; MCL29 is a subset of MCL30 with no new encoding.

A/B full-frame auxiliary pre-update features are preserved when native dimensions match, with independent global RNG and restored training buffers. No incomplete reference is substituted for a complete EMA seed. Crop updates do not depend on auxiliary statistics. Correlated frames and repeated UIDs are not independent samples. FP32 training and FP16 deployment hashes remain separate.

See evaluation_points.csv, pooled_fid.csv, equal_rate_main.csv, equal_rate_early_diagnostic.csv, loss_update_summary.csv, feature_statistics_history.csv and final_integrity.json for mechanical numeric records and exact coverage. No branch/checkpoint selection was performed.

## Fixed QP5 raw diagnostics

Video metrics use equal video weights. FID and its components use the fixed pooled frame set. These are same-QP observations; differing measured bpp prevents claiming equal-rate gains. The required six-method equal-rate comparisons remain unavailable because C/D are blocked.

| Phase | Dataset | Method | bpp | LPIPS | DISTS | FloLPIPS | pooled FID | D_mu | D_cov |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 250 diagnostic | ulong | original | 0.0070356882 | 0.10983905 | 0.081725936 | 0.10156582 | 18.822769 | 1.1624631 | 17.660306 |
| 250 diagnostic | ulong | B1_initial | 0.0067881342 | 0.095051622 | 0.071164872 | 0.086989063 | 18.996117 | 1.0581663 | 17.937951 |
| 250 diagnostic | ulong | A_C1_lr1_250 | 0.0067881342 | 0.095524971 | 0.072371053 | 0.087351467 | 19.216023 | 1.0670877 | 18.148935 |
| 250 diagnostic | ulong | B_C1_lr2_250 | 0.0067881342 | 0.096293876 | 0.073460008 | 0.087786772 | 19.44321 | 1.0751449 | 18.368065 |
| 250 diagnostic | ulong | C_C1_lr2_FD_250 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 250 diagnostic | ulong | D_C1_lr2_mean_CORAL_250 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 250 diagnostic | uvg_holdout | original | 0.01374569 | 0.13624891 | 0.0922976 | 0.13804114 | 22.269883 | 4.040573 | 18.22931 |
| 250 diagnostic | uvg_holdout | B1_initial | 0.01367296 | 0.12472406 | 0.080677802 | 0.13033398 | 19.974176 | 2.8114046 | 17.162771 |
| 250 diagnostic | uvg_holdout | A_C1_lr1_250 | 0.01367296 | 0.12203144 | 0.078237126 | 0.12668956 | 20.72344 | 3.0084482 | 17.714991 |
| 250 diagnostic | uvg_holdout | B_C1_lr2_250 | 0.01367296 | 0.12397507 | 0.079470717 | 0.12845883 | 21.177824 | 3.1843939 | 17.99343 |
| 250 diagnostic | uvg_holdout | C_C1_lr2_FD_250 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 250 diagnostic | uvg_holdout | D_C1_lr2_mean_CORAL_250 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 250 diagnostic | uvg_validation | original | 0.0092170018 | 0.10362482 | 0.071467855 | 0.09439292 | 16.602287 | 5.1204197 | 11.481868 |
| 250 diagnostic | uvg_validation | B1_initial | 0.0097102865 | 0.093011212 | 0.065961048 | 0.079929621 | 16.105218 | 4.0184642 | 12.086754 |
| 250 diagnostic | uvg_validation | A_C1_lr1_250 | 0.0097102865 | 0.093283147 | 0.065506232 | 0.079998499 | 16.330022 | 4.323809 | 12.006213 |
| 250 diagnostic | uvg_validation | B_C1_lr2_250 | 0.0097102865 | 0.094948553 | 0.067000159 | 0.081539666 | 16.52771 | 4.5074236 | 12.020287 |
| 250 diagnostic | uvg_validation | C_C1_lr2_FD_250 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 250 diagnostic | uvg_validation | D_C1_lr2_mean_CORAL_250 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 250 diagnostic | hevc_b | original | 0.0094185595 | 0.18300462 | 0.090402665 | 0.18673873 | 31.471174 | 2.7999098 | 28.671264 |
| 250 diagnostic | hevc_b | B1_initial | 0.0099283854 | 0.16724761 | 0.082740162 | 0.16214052 | 35.45752 | 3.3261188 | 32.131401 |
| 250 diagnostic | hevc_b | A_C1_lr1_250 | 0.0099283854 | 0.1653848 | 0.079916594 | 0.16070426 | 36.454013 | 3.521614 | 32.932399 |
| 250 diagnostic | hevc_b | B_C1_lr2_250 | 0.0099283854 | 0.1670292 | 0.08139794 | 0.16268968 | 37.035036 | 3.6694038 | 33.365632 |
| 250 diagnostic | hevc_b | C_C1_lr2_FD_250 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 250 diagnostic | hevc_b | D_C1_lr2_mean_CORAL_250 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 250 diagnostic | mcljcv | original | 0.0099531089 | 0.14474197 | 0.094715583 | 0.14327512 | 26.33645 | 0.98186265 | 25.354587 |
| 250 diagnostic | mcljcv | B1_initial | 0.010271466 | 0.13438456 | 0.090714473 | 0.13194161 | 27.409344 | 0.92975675 | 26.479587 |
| 250 diagnostic | mcljcv | A_C1_lr1_250 | 0.010271466 | 0.13393129 | 0.090052942 | 0.13178066 | 27.826349 | 0.96194956 | 26.8644 |
| 250 diagnostic | mcljcv | B_C1_lr2_250 | 0.010271466 | 0.13516381 | 0.090798424 | 0.13319724 | 27.948674 | 0.97973246 | 26.968941 |
| 250 diagnostic | mcljcv | C_C1_lr2_FD_250 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 250 diagnostic | mcljcv | D_C1_lr2_mean_CORAL_250 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 250 diagnostic | mcljcv29 | original | 0.009763563 | 0.14342296 | 0.094190484 | 0.14214794 | 26.933228 | 1.0087628 | 25.924465 |
| 250 diagnostic | mcljcv29 | B1_initial | 0.010091632 | 0.13414855 | 0.091169099 | 0.13180262 | 28.070736 | 0.96769843 | 27.103038 |
| 250 diagnostic | mcljcv29 | A_C1_lr1_250 | 0.010091632 | 0.13371147 | 0.09049844 | 0.13165832 | 28.483211 | 1.0011244 | 27.482086 |
| 250 diagnostic | mcljcv29 | B_C1_lr2_250 | 0.010091632 | 0.13498558 | 0.091222643 | 0.13313478 | 28.589751 | 1.0171306 | 27.572621 |
| 250 diagnostic | mcljcv29 | C_C1_lr2_FD_250 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 250 diagnostic | mcljcv29 | D_C1_lr2_mean_CORAL_250 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1000 main | ulong | original | 0.0070356882 | 0.10983905 | 0.081725936 | 0.10156582 | 18.822769 | 1.1624631 | 17.660306 |
| 1000 main | ulong | B1_initial | 0.0067881342 | 0.095051622 | 0.071164872 | 0.086989063 | 18.996117 | 1.0581663 | 17.937951 |
| 1000 main | ulong | A_C1_lr1_1000 | 0.0067881342 | 0.094421877 | 0.070187464 | 0.086526211 | 18.983839 | 1.038768 | 17.945071 |
| 1000 main | ulong | B_C1_lr2_1000 | 0.0067881342 | 0.09485506 | 0.070564241 | 0.086921284 | 18.877804 | 1.0226727 | 17.855131 |
| 1000 main | ulong | C_C1_lr2_FD_1000 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1000 main | ulong | D_C1_lr2_mean_CORAL_1000 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1000 main | uvg_holdout | original | 0.01374569 | 0.13624891 | 0.0922976 | 0.13804114 | 22.269883 | 4.040573 | 18.22931 |
| 1000 main | uvg_holdout | B1_initial | 0.01367296 | 0.12472406 | 0.080677802 | 0.13033398 | 19.974176 | 2.8114046 | 17.162771 |
| 1000 main | uvg_holdout | A_C1_lr1_1000 | 0.01367296 | 0.12304531 | 0.078370628 | 0.12748423 | 20.179099 | 2.8943008 | 17.284798 |
| 1000 main | uvg_holdout | B_C1_lr2_1000 | 0.01367296 | 0.1255601 | 0.079644481 | 0.13046543 | 19.998737 | 2.8906725 | 17.108065 |
| 1000 main | uvg_holdout | C_C1_lr2_FD_1000 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1000 main | uvg_holdout | D_C1_lr2_mean_CORAL_1000 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1000 main | uvg_validation | original | 0.0092170018 | 0.10362482 | 0.071467855 | 0.09439292 | 16.602287 | 5.1204197 | 11.481868 |
| 1000 main | uvg_validation | B1_initial | 0.0097102865 | 0.093011212 | 0.065961048 | 0.079929621 | 16.105218 | 4.0184642 | 12.086754 |
| 1000 main | uvg_validation | A_C1_lr1_1000 | 0.0097102865 | 0.092876511 | 0.064693975 | 0.079868746 | 15.554867 | 3.8907974 | 11.66407 |
| 1000 main | uvg_validation | B_C1_lr2_1000 | 0.0097102865 | 0.09457681 | 0.06592339 | 0.081205071 | 15.545741 | 3.9228293 | 11.622911 |
| 1000 main | uvg_validation | C_C1_lr2_FD_1000 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1000 main | uvg_validation | D_C1_lr2_mean_CORAL_1000 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1000 main | hevc_b | original | 0.0094185595 | 0.18300462 | 0.090402665 | 0.18673873 | 31.471174 | 2.7999098 | 28.671264 |
| 1000 main | hevc_b | B1_initial | 0.0099283854 | 0.16724761 | 0.082740162 | 0.16214052 | 35.45752 | 3.3261188 | 32.131401 |
| 1000 main | hevc_b | A_C1_lr1_1000 | 0.0099283854 | 0.1660681 | 0.081750316 | 0.15884272 | 34.869664 | 3.2769987 | 31.592666 |
| 1000 main | hevc_b | B_C1_lr2_1000 | 0.0099283854 | 0.16715657 | 0.082948366 | 0.15870547 | 34.467836 | 3.21416 | 31.253676 |
| 1000 main | hevc_b | C_C1_lr2_FD_1000 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1000 main | hevc_b | D_C1_lr2_mean_CORAL_1000 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1000 main | mcljcv | original | 0.0099531089 | 0.14474197 | 0.094715583 | 0.14327512 | 26.33645 | 0.98186265 | 25.354587 |
| 1000 main | mcljcv | B1_initial | 0.010271466 | 0.13438456 | 0.090714473 | 0.13194161 | 27.409344 | 0.92975675 | 26.479587 |
| 1000 main | mcljcv | A_C1_lr1_1000 | 0.010271466 | 0.13417651 | 0.089950324 | 0.13213243 | 27.090169 | 0.89804116 | 26.192128 |
| 1000 main | mcljcv | B_C1_lr2_1000 | 0.010271466 | 0.13539053 | 0.090555131 | 0.13327763 | 26.813555 | 0.87656274 | 25.936992 |
| 1000 main | mcljcv | C_C1_lr2_FD_1000 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1000 main | mcljcv | D_C1_lr2_mean_CORAL_1000 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1000 main | mcljcv29 | original | 0.009763563 | 0.14342296 | 0.094190484 | 0.14214794 | 26.933228 | 1.0087628 | 25.924465 |
| 1000 main | mcljcv29 | B1_initial | 0.010091632 | 0.13414855 | 0.091169099 | 0.13180262 | 28.070736 | 0.96769843 | 27.103038 |
| 1000 main | mcljcv29 | A_C1_lr1_1000 | 0.010091632 | 0.13396679 | 0.090419958 | 0.13203494 | 27.756779 | 0.9353162 | 26.821463 |
| 1000 main | mcljcv29 | B_C1_lr2_1000 | 0.010091632 | 0.13519631 | 0.091021565 | 0.13318901 | 27.490188 | 0.913625 | 26.576563 |
| 1000 main | mcljcv29 | C_C1_lr2_FD_1000 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1000 main | mcljcv29 | D_C1_lr2_mean_CORAL_1000 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

## Verification and incomplete dependencies

Common C1 checks: PASS; FP64 distribution mathematics/toy VJP: PASS; source/frozen artifact verification: PASS.
Full-chain C/D replay, calibration coefficients, EMA commits and optimizer/EMA restoration are unexecuted and BLOCKED. Their numeric coefficients and gradients are unavailable, rather than zero or passing. A/B EMA commit count is zero because the required shared seed is incomplete.
The source heights above conflict with the full-frame requirement. Eligible prefill has 996 plan occurrences and 14940 target feature occurrences; 16/20 strata have complete planned coverage. Unique UID/window/video counts and low-rank bounds are in prefill_integrity.json. Repeated or correlated frames are not independent samples.
Fixed display: HEVC and MCL first video, QP5, frames16/32/63; GT/original/B1/A/B are shown with measured bpp, and missing C/D are labeled unavailable. See previews/hevc_b/fixed_comparison.png and previews/mcljcv/fixed_comparison.png.
