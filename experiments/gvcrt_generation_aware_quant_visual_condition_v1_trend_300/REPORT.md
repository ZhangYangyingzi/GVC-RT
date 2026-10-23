# Results

## Training and coding checks

Four branches completed 300 verified optimizer updates. Preflight identity, real-stream and gradient checks passed. Every listed evaluation decoded a persisted RANS file with freshly instantiated decoder models.
46/74 evaluation tasks also compared all 64 encoder and independent-decoder causal reference states by hash. Earlier tasks ran before that extra full-sequence assertion was added; their independent decode and two-frame preflight checks passed.

## Fixed validation, step 300

The following means combine two validation sources and the two predeclared external QPs as descriptive summaries; each QP has its own actual rate. Supplemental QPs are excluded from this mean.

| Group | Mean actual bpp | LPIPS | DISTS | FloLPIPS | PSNR dB |
|---|---:|---:|---:|---:|---:|
| A | 0.009394 | 0.09170 | 0.06085 | 0.08869 | 26.827 |
| B | 0.022291 | 0.07835 | 0.05521 | 0.06897 | 27.160 |
| C | 0.009394 | 0.09292 | 0.06674 | 0.08988 | 26.643 |
| D | 0.022262 | 0.07907 | 0.06166 | 0.06943 | 26.812 |

Matched-rate overlap exists in 20/20 method pairs across the four sequences. `matched_rate_comparisons.csv` evaluates only the overlapping bpp interval in each individual sequence with descriptive linear interpolation. Same-QP means above do not demonstrate equal-rate advantage. Two points are not a formal BD-rate estimate.

### Matched-rate LPIPS differences at overlap midpoint

Positive values mean the second method has higher LPIPS at the displayed actual bpp.

| Sequence | Comparison | Bpp | ΔLPIPS second − first |
|---|---|---:|---:|
| UVal0 | A vs B | 0.011551 | +0.03069 |
| UVal0 | A vs C | 0.008177 | +0.00288 |
| UVal0 | A vs D | 0.011533 | +0.03101 |
| UVal1 | A vs B | 0.024808 | +0.02495 |
| UVal1 | A vs C | 0.016803 | +0.00057 |
| UVal1 | A vs D | 0.024783 | +0.02401 |
| ParkScene | A vs B | 0.016231 | +0.03904 |
| ParkScene | A vs C | 0.010368 | +0.00191 |
| ParkScene | A vs D | 0.016218 | +0.04011 |
| Jockey | A vs B | 0.040510 | +0.04027 |
| Jockey | A vs C | 0.028511 | +0.00384 |
| Jockey | A vs D | 0.040469 | +0.04439 |

## Descriptive cross-dataset checks, step 300

| Sequence | Group | QP | Actual bpp | LPIPS | DISTS | FloLPIPS | PSNR dB |
|---|---|---:|---:|---:|---:|---:|---:|
| ParkScene | A | 2 | 0.003284 | 0.26514 | 0.12250 | 0.24262 | 23.630 |
| ParkScene | A | 7 | 0.011640 | 0.20638 | 0.09451 | 0.19058 | 25.281 |
| ParkScene | B | 2 | 0.015010 | 0.24351 | 0.11402 | 0.23123 | 24.525 |
| ParkScene | B | 7 | 0.025322 | 0.19982 | 0.09214 | 0.18584 | 25.638 |
| ParkScene | C | 2 | 0.003284 | 0.26619 | 0.12548 | 0.23912 | 23.602 |
| ParkScene | C | 7 | 0.011640 | 0.20845 | 0.09811 | 0.19073 | 25.231 |
| ParkScene | D | 2 | 0.014984 | 0.24437 | 0.11645 | 0.22914 | 24.400 |
| ParkScene | D | 7 | 0.025336 | 0.20285 | 0.09793 | 0.18784 | 25.485 |
| Jockey | A | 2 | 0.011396 | 0.23566 | 0.11629 | 0.23848 | 24.920 |
| Jockey | A | 7 | 0.033264 | 0.15757 | 0.08468 | 0.15661 | 27.726 |
| Jockey | B | 2 | 0.035396 | 0.20471 | 0.10901 | 0.20899 | 25.249 |
| Jockey | B | 7 | 0.054660 | 0.14900 | 0.08197 | 0.14754 | 27.440 |
| Jockey | C | 2 | 0.011396 | 0.24129 | 0.11659 | 0.24573 | 24.813 |
| Jockey | C | 7 | 0.033264 | 0.16091 | 0.08419 | 0.15961 | 27.517 |
| Jockey | D | 2 | 0.035312 | 0.20907 | 0.10838 | 0.21075 | 25.078 |
| Jockey | D | 7 | 0.054627 | 0.15292 | 0.08184 | 0.14937 | 27.226 |

## Data and measurement limits

- This is a 300-update trend run with two validation source identities; no convergence or statistical significance claim is made.
- ParkScene and Jockey are fixed descriptive checks only and did not choose checkpoints or training settings.
- The inherited G_basic_100 checkpoint has mixed U-Long/UVG historical exposure. Clean historical nonexposure of evaluation datasets cannot be established.
- Real total bytes, payload bytes and headers are exact. The RANS payload jointly encodes y and z, so exact separate y/z byte counts are unavailable; the table marks analytical P-only component rates as estimates.
- FID is not computed.
