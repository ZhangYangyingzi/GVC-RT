# v6.31a paper-faithful proxy audit

1. **Is final perceptual quality better at the same rate?** The table below gives method-minus-Original P-frame means over each method's primary <=1% matched pairs. Negative means improvement; LPIPS, DISTS and FloLPIPS are assessed separately. Coverage can differ between methods.
2. **Is actual generator input closer to the LFQ target?** The cosine delta is relative to the same fixed continuous pre-LFQ proxy teacher. Negative means closer. This is not a verified original GVC-RT training teacher.
3. **Do real matched-rate worse/same-latent, better-output counterexamples exist?** Counts below require worse/same primary proxy cosine fidelity and improvement in at least two principal quality metrics; all individual metric tests are also saved.
4. **How broad is the phenomenon?** Dataset/video/QP-pair counts below use primary B1 pairs only. B_covstats is secondary; MCL29 is a sensitivity subset, not an independent dataset.
5. **Decision:** Strong, conditional on this proxy and the measured coverage.
6. **What does this say about P?** The observed counterexamples show that the P+B system can improve output without improving this proxy fidelity. P and receiver adaptation are evaluated jointly; their individual causal contributions are not isolated by this comparison.

| Method | Primary pairs | LPIPS delta | DISTS delta | FloLPIPS delta | Proxy cosine delta |
|---|---:|---:|---:|---:|---:|
| B1 | 23 | -0.0110116 | -0.00848641 | -0.0110512 | 0.0192805 |
| B_covstats | 23 | -0.0111808 | -0.00897027 | -0.0113474 | 0.0195852 |

Primary B1 reliable counterexamples: 19; strict worse-latent subset: 19; datasets: 4; videos: 19; distinct QP pairs: 9.

**Observation:** Relative to the fixed paper-compatible proxy teacher, improved tokenizer-latent fidelity is not a necessary condition for improved generative reconstruction at matched actual rate.

**Evidence:** persisted RANS bytes and independent decode, exact decoder-input hooks, fixed source sampling, pre-registered matching independent of quality, 10 points decoded twice for numerical tolerance, and video-cluster intervals. See `matched_quality_summary.csv`, `inversion_summary.csv`, `bootstrap_summary.csv`, `numerical_noise_floor.json` and figures. Quantized-target distances are separate diagnostics; no actual transmitted LFQ token-match claim is made.

**Interpretation and limits:** All conclusions are conditional on the fixed paper-compatible proxy. Neither exact original teacher identity nor the full unpublished `0.2 L_cos + 0.02 L_margin` implementation is reproduced. Cosine is an explicitly defined proxy distance, not an official primary metric. Correlation and residual correlation are associations, not causal effects. Unmatched videos contribute no primary pair. The 2% neighborhood is secondary and cannot substantiate an exact same-rate claim. P-frame fidelity is primary; I0 and all-frame results are separately retained. FID is pooled distribution evidence only. MCL29 excludes historical overlap source17 and is not independently counted.

**P diagnostics:** `wrapper_diagnostics.csv` records source-to-P RGB residual and fixed frequency-band energy changes; `latent_fidelity_per_frame.csv` records actual frame bits and payload bytes remain in per-point quality parts. The coder exposes combined entropy streams rather than uniquely attributable channel/spatial bit budgets, so no invented allocation breakdown or differentiable rate estimate is reported. No new entropy estimator was introduced.

Optional equal-norm perturbation probe was not executed; natural coding evidence remains the basis of this audit. Counterexample visualizations use fixed frame32 and automatically selected smallest-rate-gap pairs, then inversion magnitude. No counterexample is fabricated when the selection is empty.
