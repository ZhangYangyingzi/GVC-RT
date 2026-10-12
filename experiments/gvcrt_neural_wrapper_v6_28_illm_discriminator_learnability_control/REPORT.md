# V6.28 fixed-B1 discriminator learnability

Only D and its condition adapter were updated. GVC-RT/B1 and the labeler remained frozen. No new codec, MCL-JCV, FID or RD evaluation was performed.

## Fixed budgets

Four branches each completed TINY 2000 updates and MAIN 5000 updates. TINY and MAIN each started from the same full C step0 D state with a fresh optimizer.

## Final validation

| Branch | Stage | Panel | AUROC | fake−real p(fake) | Content top1 | Training majority | Content-only CE | Frequency CE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A_raw_current | TINY | tiny_fit | 0.53125 | 1.1418e-05 | 1 | 0.0272217 | 0.0842278 | 6.38326 |
| A_raw_current | TINY | unseen | 0.5001 | 1.0256e-06 | 0.00115885 | 0.00189779 | 14.1925 | 6.99558 |
| A_raw_current | MAIN | seen | 0.499875 | -5.66383e-06 | 0.0040804 | 0.0141162 | 6.86895 | 6.77204 |
| A_raw_current | MAIN | unseen | 0.499811 | -8.92291e-06 | 0.00247396 | 0.00405599 | 6.89367 | 6.84222 |
| B_rms_current | TINY | tiny_fit | 0.53125 | 0.000120021 | 1 | 0.0272217 | 0.0484206 | 6.38326 |
| B_rms_current | TINY | unseen | 0.500039 | -3.74267e-06 | 0.00103841 | 0.00189779 | 18.2751 | 6.99558 |
| B_rms_current | MAIN | seen | 0.500089 | 2.31246e-06 | 0.004375 | 0.0141162 | 6.86834 | 6.77204 |
| B_rms_current | MAIN | unseen | 0.500872 | 2.37048e-06 | 0.00249349 | 0.00405599 | 6.89214 | 6.84222 |
| C_zero_current | TINY | tiny_fit | 1 | 0.906378 | 1 | 0.0272217 | 0.0191272 | 6.38326 |
| C_zero_current | TINY | unseen | 0.479911 | -0.0175892 | 0.00148438 | 0.00189779 | 28.2758 | 6.99558 |
| C_zero_current | MAIN | seen | 0.505329 | 1.07478e-05 | 0.00495443 | 0.0141162 | 6.86524 | 6.77204 |
| C_zero_current | MAIN | unseen | 0.507278 | 9.01401e-06 | 0.00257161 | 0.00405599 | 6.89454 | 6.84222 |
| D_raw_official_optimizer | TINY | tiny_fit | 0.484375 | -4.39212e-06 | 0.032959 | 0.0272217 | 6.28794 | 6.38326 |
| D_raw_official_optimizer | TINY | unseen | 0.499233 | -6.11375e-06 | 0.00200521 | 0.00189779 | 7.29086 | 6.99558 |
| D_raw_official_optimizer | MAIN | seen | 0.499856 | -2.30372e-07 | 0.00407552 | 0.0141162 | 6.87449 | 6.77204 |
| D_raw_official_optimizer | MAIN | unseen | 0.499022 | -4.04666e-06 | 0.00247396 | 0.00405599 | 6.89491 | 6.84222 |

## Evidence and limits

An initial local preflight rejected mixed integer/float optimizer betas. Both values were cast to float without changing the optimizer recipe; the complete preflight then passed. No historical results were modified.

Normal matched-condition paired responses, AUROC and content-only classification must be considered together. Frequency baselines use training labels only with Laplace alpha=1, stratified by domain/external QP (global fallback for untrained TINY QPs); global references are retained separately. Validation label frequencies are descriptive. Unseen denotes disjoint frame intervals from this round’s MAIN plan, not absent historical exposure. Fake is one aggregate class, while real content is split across 1024 classes; fake top1 share or low full-class real accuracy alone is not used to diagnose collapse.

- A_raw_current: TINY fitting AUROC 0.531250; MAIN unseen AUROC 0.499811 (step0 change +0.004861); content top1 change +0.002074; content-only CE below training-frequency reference: False.
- B_rms_current: TINY fitting AUROC 0.531250; MAIN unseen AUROC 0.500872 (step0 change +0.004417); content top1 change +0.002096; content-only CE below training-frequency reference: False.
- C_zero_current: TINY fitting AUROC 1.000000; MAIN unseen AUROC 0.507278 (step0 change +0.017106); content top1 change +0.002161; content-only CE below training-frequency reference: False.
- D_raw_official_optimizer: TINY fitting AUROC 0.484375; MAIN unseen AUROC 0.499022 (step0 change +0.004072); content top1 change +0.002074; content-only CE below training-frequency reference: False.

## Recipe contrasts on unseen MAIN

- B_rms_current vs A: AUROC +0.001061; content-only CE -0.001528; content top1 +0.000020; paired probability separation +0.000011.
- C_zero_current vs A: AUROC +0.007467; content-only CE +0.000875; content top1 +0.000098; paired probability separation +0.000018.
- D_raw_official_optimizer vs A: AUROC -0.000789; content-only CE +0.001246; content top1 +0.000000; paired probability separation +0.000005.

The RMS and training-zero contrasts change condition handling; any corresponding learning difference is evidence about this connection under the fixed budget, not proof of a universal cause. The official-optimizer contrast changes the full AdamW recipe, not learning rate alone. Successful TINY fitting is not generalization; discriminator learning does not establish FID benefit.

## Evidence-based answers

- **A with fixed G:** MAIN unseen AUROC is 0.499811 and content top1 is 0.247396%, below the training-frequency majority reference 0.405599%. It does not establish useful unseen supervision. TINY content fits exactly, while real/fake ranking remains near chance.
- **B global RMS:** input RMS is exactly one by construction. MAIN unseen AUROC is 0.500872 and content top1 0.249349%. This does not provide convincing unseen learning improvement over A.
- **C training-zero:** TINY real/fake AUROC is 1.0, paired probability separation is 0.906378 and content top1 is 100%. The architecture/loss can fit this tiny set through the image path. MAIN unseen AUROC is 0.507278, paired separation is only 0.000009014 and content top1 0.257161%, below the frequency reference. Tiny fitting does not transfer to this unseen panel.
- **D official optimizer:** MAIN unseen AUROC is 0.499022 and content top1 0.247396%. TINY content top1 is only 3.2959% and AUROC 0.484375. This full optimizer recipe did not resolve the measured problem; learning rate alone is not isolated.
- **Frequency/fake priors:** all four MAIN unseen content-only CEs are about 6.89, above the stratified training-frequency CE 6.842224. Their near-chance real/fake ranking and tiny paired differences do not establish more useful unseen supervision than the reference predictors. Fake share alone is not used as a collapse criterion.
- **Condition and activation scales:** fixed unseen raw-condition RMS averages 1.863365. Transformed RMS is 1.863365 / 1 / 0 / 1.863365 for A/B/C/D. Final adapter-output RMS is 6.006433 / 2.822715 / 0.012140 / 2.607796; first U-Net convolution RMS is 0.498232 / 0.087455 / 0.040847 / 0.296890. C's nonzero adapter output includes constant learned biases. These scale differences and TINY contrasts concern the condition connection under this setting; they do not establish a universal cause.
- **Useful supervision:** image gradients are finite and the allowed D parameters update, but reliable unseen local-content supervision has not been established. This conclusion is limited to the fixed model, data panels, architecture and budgets. Confidence intervals were not computed and frames are correlated. No G/FID/RD improvement was tested or claimed.

## One unexecuted joint-training contrast

Compare B1-initialized decoder joint training with **raw condition versus training-zero condition**, with identical initialization, data/order, D architecture, optimizer recipe, budget and whole-generator adversarial gradient control rho=0.05. Only conditional information differs. The TINY image-path fitting contrast motivates this comparison; the MAIN unseen results do not satisfy a reliability guarantee. It was not executed and does not automatically unfreeze G.

All original validation frames, category histograms, scales, training gradients and parameter changes are indexed for review.
