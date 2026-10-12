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

