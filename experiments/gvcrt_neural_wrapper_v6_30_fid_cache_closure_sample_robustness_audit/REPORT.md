# V6.30 FID cache closure and sample robustness audit

Execution: PASS. All model/optimizer updates: 0. Calculation completion, cache closure, numerical gain retention, and Git publication are separate fields.

## 1. Fresh decode features versus historical caches

Closure: PASS; video points 420/420; pooled comparisons 36/36. Maximum relative L2=0.0; maximum absolute feature error=0.0; failed feature rows=0. UID-level errors and UNKNOWN reasons are preserved in `cache_closure_per_frame.csv`, `cache_closure_per_point.csv` and `audits/cache_closure_integrity.json`. Tolerances are fixed, no row reordering or historical cache overwrite.

## 2. Original to B1 gap after fresh extraction

3-QP pooled common-actual-bpp diagnostics:

- hevc_b_old64: B1_initial − original, FID, delta=5.39539707696381, relative%=14.432535566862473, status=available
- mcl29_old64: B1_initial − original, FID, delta=1.9249627448666722, relative%=6.018934690305685, status=available
- mcljcv_old64: B1_initial − original, FID, delta=1.7487473691257058, relative%=5.555714847302475, status=available

## 3. B minus A on fixed new clips and combined pools

- hevc_b_combined_two_clips: B_covstats_1000 − A_continue_1000, FID, delta=-0.1961378777338254, relative%=-0.5452030072757206, status=available
- hevc_b_matched_old64: B_covstats_1000 − A_continue_1000, FID, delta=-0.254232944143304, relative%=-0.6014636788844888, status=available
- hevc_b_new64: B_covstats_1000 − A_continue_1000, FID, delta=-0.15592866650273862, relative%=-0.3814277715404993, status=available
- hevc_b_old64: B_covstats_1000 − A_continue_1000, FID, delta=-0.254232944143304, relative%=-0.6014636788844888, status=available
- mcl29_old64: B_covstats_1000 − A_continue_1000, FID, delta=-0.15358981248737535, relative%=-0.4553277064941308, status=available
- mcljcv_old64: B_covstats_1000 − A_continue_1000, FID, delta=-0.1547845256447502, relative%=-0.46846933566216187, status=available

Comparisons use matching-video old/new/combined pools. Combined is two independently I-initialized 64-frame streams, not a continuous 128-frame GOP. Same-QP values are raw diagnostics unless actual bpp matches. Different pools' absolute FID values are not quality improvements. Skipped clips: see frozen new manifest; no alternate clips were selected.

## 4. Direction under video subset sampling

Subset jobs 26/26. Fixed PCG64(20261030), 10 HEVC three-video combinations; MCL eight distinct ten-video and eight distinct twenty-video subsets. See `subset_manifest.json`, `sample_sensitivity.csv` and `sample_sensitivity_summary.csv` for ranges, sign counts, native actual bpp and all members.

- {"dataset": "hevc_b", "QP": 0, "subsets": 10, "delta_min": -0.3992176258364424, "delta_max": -0.29418281404460345, "negative": 10, "positive": 0, "zero": 0, "not_confidence_interval": true}
- {"dataset": "hevc_b", "QP": 5, "subsets": 10, "delta_min": -0.28593757855674795, "delta_max": -0.1733636414361115, "negative": 10, "positive": 0, "zero": 0, "not_confidence_interval": true}
- {"dataset": "hevc_b", "QP": 9, "subsets": 10, "delta_min": -0.19255422261656463, "delta_max": -0.07467369554004577, "negative": 10, "positive": 0, "zero": 0, "not_confidence_interval": true}
- {"dataset": "mcljcv", "QP": 0, "subsets": 16, "delta_min": -0.38007163642907926, "delta_max": -0.10255652389062675, "negative": 16, "positive": 0, "zero": 0, "not_confidence_interval": true}
- {"dataset": "mcljcv", "QP": 5, "subsets": 16, "delta_min": -0.23001124460084554, "delta_max": -0.08991858298271893, "negative": 16, "positive": 0, "zero": 0, "not_confidence_interval": true}
- {"dataset": "mcljcv", "QP": 9, "subsets": 16, "delta_min": -0.10812222964607443, "delta_max": 0.0004462346183657928, "negative": 15, "positive": 1, "zero": 0, "not_confidence_interval": true}

## 5. Supported evidence and limits

A PASS supports only the specified feature-cache/decode/pooled-statistics consistency checks. Recorded sign counts describe these frozen temporal clips and video subsets. They are not repeated training runs, confidence intervals, significance proofs, or evidence that sample-size bias has been removed. Video correlation, finite samples and a single training realization remain limitations. New temporal clips are from the same test videos, not new scenes. MCL videoSRC17 matches HEVC Kimono1 across test datasets; this is not by itself training-leakage evidence. No FID-infinity fitting or checkpoint selection was performed.

## 6. Evaluation-chain consistency status

The fixed chain passed the specified closed-loop checks. This is not a method-success or significant-gain claim.

Publication: independently recorded in `publication_status.json`; a computation PASS does not claim remote publication. No next-round training is launched.
