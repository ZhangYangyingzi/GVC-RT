# v6.44 measured result

Fixed offline oracle candidate: ParkScene frame 8 and 32 actual QP 5→3. One new full 64-frame stream, zero optimizer updates.

| Metric | v6.43 baseline | two-frame measured | variant − baseline |
|---|---:|---:|---:|
| Complete stream bits | 977264 | 971984 | -5280 |
| actual_bpp | 0.0073638841 | 0.0073240982 | -0.0000397859 |
| LPIPS | 0.2153701321 | 0.2147590932 | -0.0006110389 |
| DISTS | 0.0970273521 | 0.0973497657 | +0.0003224136 |
| L1 | 0.0380315627 | 0.0381301373 | +0.0000985746 |
| PSNR | 24.7273104223 | 24.7030832304 | -0.0242271920 |
| FloLPIPS | 0.1969664073 | 0.1972537246 | +0.0002873173 |

Non-interaction additive prediction from the two v6.43 single-frame runs: -4760 bit, -0.0012240477 LPIPS. Measured minus prediction: -520 bit, +0.0006130089 LPIPS.

**Prespecified criterion: PASS.** The measured stream has no more than 977264 bit and mean LPIPS is below 0.2153701321221888.

The variant was encoded from native I0 with its own evolving references; all 64 frames were independently decoded from its persisted stream. Frames 0–7 match the baseline packets, reconstructions and reference states exactly. All packets, QP fields, bytes consumed, frozen models and 63 transitions passed checks. PSNR uses summed whole-segment RGB SSE. The source was chosen by an offline oracle; this single sequence does not establish a deployable rule or generalization.
