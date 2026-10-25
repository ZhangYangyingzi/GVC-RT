# v3 rate and spatial multiplier ablation

Fixed step 300 inference only. All reported rates include the 16-byte v3 experiment header. No training or weight update was performed. Lower LPIPS, DISTS and FloLPIPS is better; higher PSNR is better.

## Execution and accounting

- Preflight: PASS; delivery audit: PASS.
- Protocol result records: 57; 16 native-stream anchors and 5 existing QP9 supplement points were rewrapped from v2, 36 full protocol evaluations were newly run.
- Rate-only candidates: 0; new selected-alpha full evaluations: 0.
- Encoder/decoder consistency for newly encoded tasks checks every frame reference and every P-frame latent and qmap. All 21 reused streams also pass fresh v2/v3 decoder comparisons with exact RGB, reference, latent and qmap hashes on all 64 frames. A and IDENTITY match native streams and all codec states at all four stage-2 source/QP combinations.
- Two additional alpha=0.8 rate-only encodes were performed for the preflight synchronization checks, not for candidate selection. They are separately audited in `delivery_audit.json`.
- Preflight also contains three full metric runs (A/MAP/IDENTITY on UVal0 QP2). The whole round therefore used 39 full metric runs including preflight, within the cap of 44. The final receiver audits only decode existing streams and do not recompute quality metrics.
- `evaluation.csv` gives total bytes, codec payload, native header, experiment header, P payload, P-only estimated y/z bits, runtime and peak memory. Interleaved RANS does not expose exact y/z payload byte partitions.

## A. A versus MAP: measured rates and bracketed interpolation

### Same external QP: measured, with rate differences

These rows compare the same external QP, not equal rate. Each delta is MAP minus A.

| video | QP | A bpp | MAP bpp | rate gap bpp | delta LPIPS | delta DISTS | delta FloLPIPS | delta PSNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UVal0 | 2 | 0.003374 | 0.006872 | 0.003498 | -0.02422 | -0.01200 | -0.03375 | 1.35144 |
| UVal0 | 4 | 0.005218 | 0.008374 | 0.003156 | -0.01124 | -0.00624 | -0.01566 | 0.81449 |
| UVal0 | 6 | 0.007735 | 0.009543 | 0.001808 | -0.00421 | -0.00279 | -0.00475 | 0.37706 |
| UVal0 | 7 | 0.009307 | 0.010079 | 0.000772 | -0.00192 | -0.00126 | -0.00089 | 0.16755 |
| UVal0 | 9 | 0.012983 | 0.011502 | -0.001481 | 0.00175 | 0.00113 | 0.00367 | -0.10886 |
| UVal1 | 2 | 0.005636 | 0.011474 | 0.005838 | -0.02609 | -0.01160 | -0.02969 | 0.76931 |
| UVal1 | 4 | 0.009722 | 0.014914 | 0.005192 | -0.01376 | -0.00665 | -0.01958 | 0.56851 |
| UVal1 | 6 | 0.015575 | 0.017493 | 0.001918 | -0.00330 | -0.00244 | -0.00590 | 0.22717 |
| UVal1 | 7 | 0.019263 | 0.018741 | -0.000522 | 0.00063 | 0.00063 | -0.00005 | 0.04995 |
| ParkScene | 2 | 0.003285 | 0.008275 | 0.004990 | -0.02932 | -0.01362 | -0.02203 | 0.94096 |
| ParkScene | 4 | 0.005751 | 0.011058 | 0.005308 | -0.01670 | -0.00982 | -0.01421 | 0.66421 |
| ParkScene | 6 | 0.009299 | 0.013962 | 0.004663 | -0.00975 | -0.00529 | -0.00903 | 0.40524 |
| ParkScene | 7 | 0.011641 | 0.015292 | 0.003651 | -0.00731 | -0.00438 | -0.00729 | 0.27612 |
| Jockey | 2 | 0.011397 | 0.021127 | 0.009729 | -0.04325 | -0.01291 | -0.03971 | 1.39700 |
| Jockey | 4 | 0.018306 | 0.024779 | 0.006473 | -0.02134 | -0.00589 | -0.02819 | 0.81785 |
| Jockey | 6 | 0.027668 | 0.027294 | -0.000373 | -0.00348 | -0.00063 | -0.00634 | 0.14990 |
| Jockey | 7 | 0.033265 | 0.028898 | -0.004367 | 0.00300 | 0.00162 | 0.00128 | -0.11697 |

### Nearest measured A point and bracketed estimates

Each target below is the fixed measured MAP QP2 or QP7 rate. Nearest-point rows are measured but retain a rate gap. Interpolation uses only measured A points bracketing that fixed target; missing brackets are left empty. The interpolated A quality is an estimate, not a measurement.

### LPIPS

| video | MAP QP | MAP bpp | MAP measured | nearest A QP / bpp | rate gap bpp | nearest A measured | MAP−A nearest | MAP−A linear | MAP−A log | same sign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UVal0 | 2 | 0.006872 | 0.07034 | 5 / 0.006398 | -0.000474 | 0.06288 | 0.00746 | 0.00932 | 0.00944 | True |
| UVal0 | 7 | 0.010079 | 0.05130 | 7 / 0.009307 | -0.000772 | 0.05322 | -0.00192 | -0.00071 | -0.00054 | True |
| UVal1 | 2 | 0.011474 | 0.10861 | 5 / 0.012399 | 0.000925 | 0.09774 | 0.01087 | 0.00750 | 0.00777 | True |
| UVal1 | 7 | 0.018741 | 0.08493 | 7 / 0.019263 | 0.000522 | 0.08430 | 0.00063 | -0.00019 | -0.00011 | True |
| ParkScene | 2 | 0.008275 | 0.23581 | 5 / 0.007365 | -0.000911 | 0.22269 | 0.01313 | 0.01783 | 0.01812 | True |
| ParkScene | 7 | 0.015292 | 0.19907 | 9 / 0.017453 | 0.002161 | 0.19741 | 0.00166 | -0.00167 | -0.00126 | True |
| Jockey | 2 | 0.021127 | 0.19241 | 5 / 0.022724 | 0.001597 | 0.17852 | 0.01390 | 0.00835 | 0.00872 | True |
| Jockey | 7 | 0.028898 | 0.16057 | 6 / 0.027668 | -0.001230 | 0.16703 | -0.00646 | -0.00438 | -0.00423 | True |

### DISTS

| video | MAP QP | MAP bpp | MAP measured | nearest A QP / bpp | rate gap bpp | nearest A measured | MAP−A nearest | MAP−A linear | MAP−A log | same sign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UVal0 | 2 | 0.006872 | 0.03891 | 5 / 0.006398 | -0.000474 | 0.03449 | 0.00443 | 0.00559 | 0.00566 | True |
| UVal0 | 7 | 0.010079 | 0.02683 | 7 / 0.009307 | -0.000772 | 0.02810 | -0.00126 | -0.00066 | -0.00058 | True |
| UVal1 | 2 | 0.011474 | 0.08726 | 5 / 0.012399 | 0.000925 | 0.07548 | 0.01178 | 0.00984 | 0.00999 | True |
| UVal1 | 7 | 0.018741 | 0.06616 | 7 / 0.019263 | 0.000522 | 0.06553 | 0.00063 | 0.00002 | 0.00007 | True |
| ParkScene | 2 | 0.008275 | 0.10889 | 5 / 0.007365 | -0.000911 | 0.10249 | 0.00639 | 0.00854 | 0.00867 | True |
| ParkScene | 7 | 0.015292 | 0.09013 | 9 / 0.017453 | 0.002161 | 0.08946 | 0.00067 | -0.00121 | -0.00098 | True |
| Jockey | 2 | 0.021127 | 0.10338 | 5 / 0.022724 | 0.001597 | 0.09385 | 0.00952 | 0.00733 | 0.00748 | True |
| Jockey | 7 | 0.028898 | 0.08630 | 6 / 0.027668 | -0.001230 | 0.08862 | -0.00232 | -0.00146 | -0.00139 | True |

### FloLPIPS

| video | MAP QP | MAP bpp | MAP measured | nearest A QP / bpp | rate gap bpp | nearest A measured | MAP−A nearest | MAP−A linear | MAP−A log | same sign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UVal0 | 2 | 0.006872 | 0.08272 | 5 / 0.006398 | -0.000474 | 0.07563 | 0.00709 | 0.00984 | 0.01001 | True |
| UVal0 | 7 | 0.010079 | 0.06149 | 7 / 0.009307 | -0.000772 | 0.06238 | -0.00089 | 0.00063 | 0.00084 | True |
| UVal1 | 2 | 0.011474 | 0.08682 | 5 / 0.012399 | 0.000925 | 0.07491 | 0.01191 | 0.00734 | 0.00769 | True |
| UVal1 | 7 | 0.018741 | 0.05933 | 7 / 0.019263 | 0.000522 | 0.05938 | -0.00005 | -0.00117 | -0.00107 | True |
| ParkScene | 2 | 0.008275 | 0.22059 | 5 / 0.007365 | -0.000911 | 0.20586 | 0.01473 | 0.01886 | 0.01911 | True |
| ParkScene | 7 | 0.015292 | 0.18329 | 9 / 0.017453 | 0.002161 | 0.18034 | 0.00296 | -0.00085 | -0.00039 | True |
| Jockey | 2 | 0.021127 | 0.19877 | 5 / 0.022724 | 0.001597 | 0.18106 | 0.01772 | 0.01119 | 0.01163 | True |
| Jockey | 7 | 0.028898 | 0.15789 | 6 / 0.027668 | -0.001230 | 0.16778 | -0.00989 | -0.00743 | -0.00725 | True |

### PSNR

| video | MAP QP | MAP bpp | MAP measured | nearest A QP / bpp | rate gap bpp | nearest A measured | MAP−A nearest | MAP−A linear | MAP−A log | same sign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UVal0 | 2 | 0.006872 | 30.06566 | 5 / 0.006398 | -0.000474 | 30.46942 | -0.40376 | -0.54199 | -0.55060 | True |
| UVal0 | 7 | 0.010079 | 31.38075 | 7 / 0.009307 | -0.000772 | 31.21320 | 0.16755 | 0.07100 | 0.05749 | True |
| UVal1 | 2 | 0.011474 | 23.63110 | 5 / 0.012399 | 0.000925 | 23.94705 | -0.31595 | -0.20441 | -0.21305 | True |
| UVal1 | 7 | 0.018741 | 24.56801 | 7 / 0.019263 | 0.000522 | 24.51806 | 0.04995 | 0.08797 | 0.08468 | True |
| ParkScene | 2 | 0.008275 | 24.57047 | 5 / 0.007365 | -0.000911 | 24.75177 | -0.18130 | -0.31408 | -0.32229 | True |
| ParkScene | 7 | 0.015292 | 25.55685 | 9 / 0.017453 | 0.002161 | 25.67388 | -0.11703 | 0.02915 | 0.01129 | True |
| Jockey | 2 | 0.021127 | 26.31662 | 5 / 0.022724 | 0.001597 | 26.80036 | -0.48374 | -0.27900 | -0.29283 | True |
| Jockey | 7 | 0.028898 | 27.60886 | 6 / 0.027668 | -0.001230 | 27.29702 | 0.31184 | 0.21759 | 0.21059 | True |

Interpolation sensitivity is tabulated in `interpolation_sensitivity.csv`; there is no extrapolation or BD-rate estimate. Sparse estimates use only the existing A QP2/7/9 measurements. Dense estimates add A QP3/4/5/6; all comparisons keep exactly the same MAP QP2/7 target rates.

Densifying A changes the sign of 8/32 linear-bpp contrasts and 1/32 log-bpp contrasts relative to the sparse measurements. After densification, linear-bpp and log-bpp signs agree in all 32 metric/target contrasts. At the four MAP QP2 target rates, MAP has worse interpolated comparisons on all four metrics. At the four MAP QP7 target rates, MAP has lower LPIPS in 4/4, lower DISTS in 3/4, lower FloLPIPS in 3/4, and higher PSNR in 4/4. These are local interpolated comparisons; the nearest measured points and their rate gaps remain the direct measurement evidence.

| video | MAP QP | metric | sparse linear delta | dense linear delta | sparse log delta | dense log delta | linear sign unchanged | log sign unchanged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UVal0 | 2 | LPIPS | 0.00015 | 0.00932 | 0.00476 | 0.00944 | True | True |
| UVal0 | 2 | DISTS | 0.00145 | 0.00559 | 0.00400 | 0.00566 | True | True |
| UVal0 | 2 | FloLPIPS | -0.00186 | 0.00984 | 0.00417 | 0.01001 | False | True |
| UVal0 | 2 | PSNR | -0.12191 | -0.54199 | -0.40053 | -0.55060 | True | True |
| UVal0 | 7 | LPIPS | -0.00071 | -0.00071 | -0.00054 | -0.00054 | True | True |
| UVal0 | 7 | DISTS | -0.00066 | -0.00066 | -0.00058 | -0.00058 | True | True |
| UVal0 | 7 | FloLPIPS | 0.00063 | 0.00063 | 0.00084 | 0.00084 | True | True |
| UVal0 | 7 | PSNR | 0.07100 | 0.07100 | 0.05749 | 0.05749 | True | True |
| UVal1 | 2 | LPIPS | -0.00450 | 0.00750 | 0.00307 | 0.00777 | False | True |
| UVal1 | 2 | DISTS | 0.00268 | 0.00984 | 0.00768 | 0.00999 | True | True |
| UVal1 | 2 | FloLPIPS | -0.00521 | 0.00734 | 0.00336 | 0.00769 | False | True |
| UVal1 | 2 | PSNR | 0.05969 | -0.20441 | -0.18879 | -0.21305 | False | True |
| UVal1 | 7 | LPIPS | -0.00130 | -0.00019 | -0.00050 | -0.00011 | True | True |
| UVal1 | 7 | DISTS | -0.00064 | 0.00002 | -0.00011 | 0.00007 | False | False |
| UVal1 | 7 | FloLPIPS | -0.00224 | -0.00117 | -0.00132 | -0.00107 | True | True |
| UVal1 | 7 | PSNR | 0.11336 | 0.08797 | 0.08695 | 0.08468 | True | True |
| ParkScene | 2 | LPIPS | 0.00577 | 0.01783 | 0.01358 | 0.01812 | True | True |
| ParkScene | 2 | DISTS | 0.00310 | 0.00854 | 0.00682 | 0.00867 | True | True |
| ParkScene | 2 | FloLPIPS | 0.00905 | 0.01886 | 0.01597 | 0.01911 | True | True |
| ParkScene | 2 | PSNR | -0.04516 | -0.31408 | -0.26485 | -0.32229 | True | True |
| ParkScene | 7 | LPIPS | -0.00167 | -0.00167 | -0.00126 | -0.00126 | True | True |
| ParkScene | 7 | DISTS | -0.00121 | -0.00121 | -0.00098 | -0.00098 | True | True |
| ParkScene | 7 | FloLPIPS | -0.00085 | -0.00085 | -0.00039 | -0.00039 | True | True |
| ParkScene | 7 | PSNR | 0.02915 | 0.02915 | 0.01129 | 0.01129 | True | True |
| Jockey | 2 | LPIPS | -0.00850 | 0.00835 | 0.00175 | 0.00872 | False | True |
| Jockey | 2 | DISTS | 0.00115 | 0.00733 | 0.00530 | 0.00748 | True | True |
| Jockey | 2 | FloLPIPS | -0.00328 | 0.01119 | 0.00746 | 0.01163 | False | True |
| Jockey | 2 | PSNR | 0.14845 | -0.27900 | -0.21987 | -0.29283 | False | True |
| Jockey | 7 | LPIPS | -0.01259 | -0.00438 | -0.00726 | -0.00423 | True | True |
| Jockey | 7 | DISTS | -0.00469 | -0.00146 | -0.00253 | -0.00139 | True | True |
| Jockey | 7 | FloLPIPS | -0.01507 | -0.00743 | -0.00947 | -0.00725 | True | True |
| Jockey | 7 | PSNR | 0.44343 | 0.21759 | 0.25173 | 0.21059 | True | True |

## B. A versus IDENTITY: same native codec stream, different generator

IDENTITY uses H11 generator weights with H multiplier set to one. The native stream SHA is compared with A at each source/QP.

| video | QP | metric | A bpp | IDENTITY bpp | IDENTITY−A measured | native SHA equal |
| --- | --- | --- | --- | --- | --- | --- |
| UVal1 | 2 | LPIPS | 0.005636 | 0.005636 | 0.00093 | True |
| UVal1 | 2 | DISTS | 0.005636 | 0.005636 | 0.00005 | True |
| UVal1 | 2 | FloLPIPS | 0.005636 | 0.005636 | 0.00115 | True |
| UVal1 | 2 | PSNR | 0.005636 | 0.005636 | 0.08001 | True |
| UVal1 | 7 | LPIPS | 0.019263 | 0.019263 | 0.00011 | True |
| UVal1 | 7 | DISTS | 0.019263 | 0.019263 | -0.00020 | True |
| UVal1 | 7 | FloLPIPS | 0.019263 | 0.019263 | 0.00002 | True |
| UVal1 | 7 | PSNR | 0.019263 | 0.019263 | 0.07342 | True |
| Jockey | 2 | LPIPS | 0.011397 | 0.011397 | -0.00116 | True |
| Jockey | 2 | DISTS | 0.011397 | 0.011397 | 0.00045 | True |
| Jockey | 2 | FloLPIPS | 0.011397 | 0.011397 | -0.00072 | True |
| Jockey | 2 | PSNR | 0.011397 | 0.011397 | 0.09751 | True |
| Jockey | 7 | LPIPS | 0.033265 | 0.033265 | -0.00265 | True |
| Jockey | 7 | DISTS | 0.033265 | 0.033265 | -0.00049 | True |
| Jockey | 7 | FloLPIPS | 0.033265 | 0.033265 | -0.00317 | True |
| Jockey | 7 | PSNR | 0.033265 | 0.033265 | 0.07497 | True |

With identical native streams and exact codec states, IDENTITY has higher PSNR in 4/4 cases. LPIPS, DISTS and FloLPIPS are each lower in 2/4 cases. Generator weight changes therefore have metric-dependent effects even when compression is held fixed.

## C. MAP versus MEAN

The same-QP table uses alpha 1; the near-rate table uses the candidate selected only by total bytes. Every near-rate row retains its actual rate gap and match status. All four alpha=1 cases already fall within the fixed 2% tolerance, so the near-rate table reuses those same full evaluations and no further alpha candidates were encoded.

### same_QP_spatial_mean

| video | QP | metric | alpha | MAP bpp | variant bpp | rate gap % | variant−MAP measured | rate status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UVal1 | 2 | LPIPS | 1.0 | 0.011474 | 0.011511 | 0.32 | 0.00020 | same QP |
| UVal1 | 2 | DISTS | 1.0 | 0.011474 | 0.011511 | 0.32 | 0.00090 | same QP |
| UVal1 | 2 | FloLPIPS | 1.0 | 0.011474 | 0.011511 | 0.32 | -0.00144 | same QP |
| UVal1 | 2 | PSNR | 1.0 | 0.011474 | 0.011511 | 0.32 | -0.00473 | same QP |
| UVal1 | 7 | LPIPS | 1.0 | 0.018741 | 0.018857 | 0.62 | -0.00076 | same QP |
| UVal1 | 7 | DISTS | 1.0 | 0.018741 | 0.018857 | 0.62 | -0.00092 | same QP |
| UVal1 | 7 | FloLPIPS | 1.0 | 0.018741 | 0.018857 | 0.62 | -0.00122 | same QP |
| UVal1 | 7 | PSNR | 1.0 | 0.018741 | 0.018857 | 0.62 | 0.05275 | same QP |
| Jockey | 2 | LPIPS | 1.0 | 0.021127 | 0.020962 | -0.78 | 0.00083 | same QP |
| Jockey | 2 | DISTS | 1.0 | 0.021127 | 0.020962 | -0.78 | 0.00039 | same QP |
| Jockey | 2 | FloLPIPS | 1.0 | 0.021127 | 0.020962 | -0.78 | -0.00019 | same QP |
| Jockey | 2 | PSNR | 1.0 | 0.021127 | 0.020962 | -0.78 | -0.02353 | same QP |
| Jockey | 7 | LPIPS | 1.0 | 0.028898 | 0.028863 | -0.12 | -0.00036 | same QP |
| Jockey | 7 | DISTS | 1.0 | 0.028898 | 0.028863 | -0.12 | -0.00021 | same QP |
| Jockey | 7 | FloLPIPS | 1.0 | 0.028898 | 0.028863 | -0.12 | -0.00030 | same QP |
| Jockey | 7 | PSNR | 1.0 | 0.028898 | 0.028863 | -0.12 | 0.02769 | same QP |

### near_rate_MEAN

| video | QP | metric | alpha | MAP bpp | variant bpp | rate gap % | variant−MAP measured | rate status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UVal1 | 2 | LPIPS | 1.0 | 0.011474 | 0.011511 | 0.32 | 0.00020 | CLOSE_RATE_MATCH |
| UVal1 | 2 | DISTS | 1.0 | 0.011474 | 0.011511 | 0.32 | 0.00090 | CLOSE_RATE_MATCH |
| UVal1 | 2 | FloLPIPS | 1.0 | 0.011474 | 0.011511 | 0.32 | -0.00144 | CLOSE_RATE_MATCH |
| UVal1 | 2 | PSNR | 1.0 | 0.011474 | 0.011511 | 0.32 | -0.00473 | CLOSE_RATE_MATCH |
| UVal1 | 7 | LPIPS | 1.0 | 0.018741 | 0.018857 | 0.62 | -0.00076 | CLOSE_RATE_MATCH |
| UVal1 | 7 | DISTS | 1.0 | 0.018741 | 0.018857 | 0.62 | -0.00092 | CLOSE_RATE_MATCH |
| UVal1 | 7 | FloLPIPS | 1.0 | 0.018741 | 0.018857 | 0.62 | -0.00122 | CLOSE_RATE_MATCH |
| UVal1 | 7 | PSNR | 1.0 | 0.018741 | 0.018857 | 0.62 | 0.05275 | CLOSE_RATE_MATCH |
| Jockey | 2 | LPIPS | 1.0 | 0.021127 | 0.020962 | -0.78 | 0.00083 | CLOSE_RATE_MATCH |
| Jockey | 2 | DISTS | 1.0 | 0.021127 | 0.020962 | -0.78 | 0.00039 | CLOSE_RATE_MATCH |
| Jockey | 2 | FloLPIPS | 1.0 | 0.021127 | 0.020962 | -0.78 | -0.00019 | CLOSE_RATE_MATCH |
| Jockey | 2 | PSNR | 1.0 | 0.021127 | 0.020962 | -0.78 | -0.02353 | CLOSE_RATE_MATCH |
| Jockey | 7 | LPIPS | 1.0 | 0.028898 | 0.028863 | -0.12 | -0.00036 | CLOSE_RATE_MATCH |
| Jockey | 7 | DISTS | 1.0 | 0.028898 | 0.028863 | -0.12 | -0.00021 | CLOSE_RATE_MATCH |
| Jockey | 7 | FloLPIPS | 1.0 | 0.028898 | 0.028863 | -0.12 | -0.00030 | CLOSE_RATE_MATCH |
| Jockey | 7 | PSNR | 1.0 | 0.028898 | 0.028863 | -0.12 | 0.02769 | CLOSE_RATE_MATCH |

MEAN has lower LPIPS in 2/4, lower DISTS in 2/4, lower FloLPIPS in 4/4, and higher PSNR in 2/4 cases. The fixed-weight replacement does not show a consistent advantage for MAP across these cases and metrics. MEAN averages only H's added multiplier; q_base retains its own spatial/channel variation.

## D. MAP versus ROLL

The same-QP table uses alpha 1; the near-rate table uses the candidate selected only by total bytes. Every near-rate row retains its actual rate gap and match status. All four alpha=1 cases already fall within the fixed 2% tolerance, so the near-rate table reuses those same full evaluations and no further alpha candidates were encoded.

### same_QP_spatial_roll

| video | QP | metric | alpha | MAP bpp | variant bpp | rate gap % | variant−MAP measured | rate status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UVal1 | 2 | LPIPS | 1.0 | 0.011474 | 0.011504 | 0.26 | 0.00034 | same QP |
| UVal1 | 2 | DISTS | 1.0 | 0.011474 | 0.011504 | 0.26 | 0.00092 | same QP |
| UVal1 | 2 | FloLPIPS | 1.0 | 0.011474 | 0.011504 | 0.26 | -0.00099 | same QP |
| UVal1 | 2 | PSNR | 1.0 | 0.011474 | 0.011504 | 0.26 | -0.01355 | same QP |
| UVal1 | 7 | LPIPS | 1.0 | 0.018741 | 0.019108 | 1.96 | -0.00061 | same QP |
| UVal1 | 7 | DISTS | 1.0 | 0.018741 | 0.019108 | 1.96 | -0.00045 | same QP |
| UVal1 | 7 | FloLPIPS | 1.0 | 0.018741 | 0.019108 | 1.96 | -0.00040 | same QP |
| UVal1 | 7 | PSNR | 1.0 | 0.018741 | 0.019108 | 1.96 | 0.06457 | same QP |
| Jockey | 2 | LPIPS | 1.0 | 0.021127 | 0.020971 | -0.74 | 0.00108 | same QP |
| Jockey | 2 | DISTS | 1.0 | 0.021127 | 0.020971 | -0.74 | 0.00046 | same QP |
| Jockey | 2 | FloLPIPS | 1.0 | 0.021127 | 0.020971 | -0.74 | 0.00066 | same QP |
| Jockey | 2 | PSNR | 1.0 | 0.021127 | 0.020971 | -0.74 | -0.06631 | same QP |
| Jockey | 7 | LPIPS | 1.0 | 0.028898 | 0.028990 | 0.32 | -0.00057 | same QP |
| Jockey | 7 | DISTS | 1.0 | 0.028898 | 0.028990 | 0.32 | -0.00019 | same QP |
| Jockey | 7 | FloLPIPS | 1.0 | 0.028898 | 0.028990 | 0.32 | 0.00129 | same QP |
| Jockey | 7 | PSNR | 1.0 | 0.028898 | 0.028990 | 0.32 | 0.02469 | same QP |

### near_rate_ROLL

| video | QP | metric | alpha | MAP bpp | variant bpp | rate gap % | variant−MAP measured | rate status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UVal1 | 2 | LPIPS | 1.0 | 0.011474 | 0.011504 | 0.26 | 0.00034 | CLOSE_RATE_MATCH |
| UVal1 | 2 | DISTS | 1.0 | 0.011474 | 0.011504 | 0.26 | 0.00092 | CLOSE_RATE_MATCH |
| UVal1 | 2 | FloLPIPS | 1.0 | 0.011474 | 0.011504 | 0.26 | -0.00099 | CLOSE_RATE_MATCH |
| UVal1 | 2 | PSNR | 1.0 | 0.011474 | 0.011504 | 0.26 | -0.01355 | CLOSE_RATE_MATCH |
| UVal1 | 7 | LPIPS | 1.0 | 0.018741 | 0.019108 | 1.96 | -0.00061 | CLOSE_RATE_MATCH |
| UVal1 | 7 | DISTS | 1.0 | 0.018741 | 0.019108 | 1.96 | -0.00045 | CLOSE_RATE_MATCH |
| UVal1 | 7 | FloLPIPS | 1.0 | 0.018741 | 0.019108 | 1.96 | -0.00040 | CLOSE_RATE_MATCH |
| UVal1 | 7 | PSNR | 1.0 | 0.018741 | 0.019108 | 1.96 | 0.06457 | CLOSE_RATE_MATCH |
| Jockey | 2 | LPIPS | 1.0 | 0.021127 | 0.020971 | -0.74 | 0.00108 | CLOSE_RATE_MATCH |
| Jockey | 2 | DISTS | 1.0 | 0.021127 | 0.020971 | -0.74 | 0.00046 | CLOSE_RATE_MATCH |
| Jockey | 2 | FloLPIPS | 1.0 | 0.021127 | 0.020971 | -0.74 | 0.00066 | CLOSE_RATE_MATCH |
| Jockey | 2 | PSNR | 1.0 | 0.021127 | 0.020971 | -0.74 | -0.06631 | CLOSE_RATE_MATCH |
| Jockey | 7 | LPIPS | 1.0 | 0.028898 | 0.028990 | 0.32 | -0.00057 | CLOSE_RATE_MATCH |
| Jockey | 7 | DISTS | 1.0 | 0.028898 | 0.028990 | 0.32 | -0.00019 | CLOSE_RATE_MATCH |
| Jockey | 7 | FloLPIPS | 1.0 | 0.028898 | 0.028990 | 0.32 | 0.00129 | CLOSE_RATE_MATCH |
| Jockey | 7 | PSNR | 1.0 | 0.028898 | 0.028990 | 0.32 | 0.02469 | CLOSE_RATE_MATCH |

ROLL has lower LPIPS, lower DISTS, lower FloLPIPS and higher PSNR in 2/4 cases each. Its multiplier value multiset is exactly preserved on every P frame, but coupling to q_base and the subsequent decoded state can change. These mixed observations do not establish a final contribution of spatial correspondence.

## Receiver diagnostics

Values below average each P frame's spatial statistics. Skip fractions refer to the symbols in each coded pass; clamp fractions count H-adjusted q entries below the v2 lower bound before clamping. At QP2, about 88% of MAP q entries trigger that lower bound, so effective rho can differ substantially from H's multiplier. The full per-frame quantiles are in `quant_diagnostics.csv`.

| video | QP | mode | m mean | m spatial std | rho mean | rho spatial/channel std | q floor % | pass0 skip % | pass1 skip % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UVal1 | 2 | MAP | 0.65576 | 0.05538 | 0.95483 | 0.11727 | 87.72 | 91.41 | 97.63 |
| UVal1 | 7 | MAP | 1.00717 | 0.06773 | 1.05089 | 0.05125 | 43.20 | 88.76 | 97.02 |
| Jockey | 2 | MAP | 0.67508 | 0.05492 | 0.95702 | 0.11137 | 87.58 | 88.99 | 96.85 |
| Jockey | 7 | MAP | 1.10370 | 0.05663 | 1.11798 | 0.04770 | 25.79 | 87.13 | 96.26 |
| UVal1 | 2 | MEAN | 0.65612 | 0.00000 | 0.95478 | 0.11562 | 87.60 | 91.43 | 97.62 |
| UVal1 | 2 | ROLL | 0.65612 | 0.05579 | 0.95493 | 0.11704 | 87.72 | 91.43 | 97.62 |
| UVal1 | 2 | IDENTITY | 1.00000 | 0.00000 | 1.00000 | 0.00000 | 0.00 | 93.77 | 98.18 |
| UVal1 | 7 | MEAN | 1.00705 | 0.00000 | 1.04447 | 0.01474 | 48.16 | 88.81 | 97.04 |
| UVal1 | 7 | ROLL | 1.00717 | 0.06955 | 1.05121 | 0.05201 | 43.24 | 88.84 | 97.04 |
| UVal1 | 7 | IDENTITY | 1.00000 | 0.00000 | 1.00000 | 0.00000 | 0.00 | 88.82 | 97.03 |
| Jockey | 2 | MEAN | 0.67485 | 0.00000 | 0.95692 | 0.10969 | 87.40 | 88.98 | 96.85 |
| Jockey | 2 | ROLL | 0.67484 | 0.05527 | 0.95708 | 0.11126 | 87.61 | 88.98 | 96.85 |
| Jockey | 2 | IDENTITY | 1.00000 | 0.00000 | 1.00000 | 0.00000 | 0.00 | 90.73 | 97.38 |
| Jockey | 7 | MEAN | 1.10349 | 0.00000 | 1.11337 | 0.00404 | 34.23 | 87.14 | 96.26 |
| Jockey | 7 | ROLL | 1.10352 | 0.05650 | 1.11785 | 0.04744 | 25.82 | 87.14 | 96.26 |
| Jockey | 7 | IDENTITY | 1.00000 | 0.00000 | 1.00000 | 0.00000 | 0.00 | 87.13 | 96.25 |

## Fixed outputs and limitations

- `rate_match_search.csv` records every encoded candidate, target bytes, rate error, and selected flag. Selection used no quality metric.
- `per_frame.csv`, `per_transition.csv`, and `frame_segment_summaries.csv` separate individual frames and the first 15 versus remaining 49 frames.
- `figures/RD_*.png` shows measured A/MAP rate–quality points separately for all four metrics. `figures/fixed_previews/` fixes frames 15 and 47 and crop rows 200:456, columns 400:656. `figures/quant_maps/` shows full latent fields using fixed 0.5–2.0 scales for multiplier/rho and 0.5–16 for mean-channel q across all methods.
- `task_manifest.csv` and `configs/` bind each task to source, weight, runtime and configuration SHAs. Original storage names are retained as aliases for resume compatibility. `quant_diagnostics.csv` records every P-frame receiver distribution and both skip fractions; `quant_summary.csv` aggregates their frame statistics. `resource_summary.csv` separates original evaluation cost from the additional receiver audits.
- MEAN removes only the spatial variation of the added H multiplier; base codec q may still vary over space and channels.
- Inference substitution can shift generator input distributions. These observations do not establish equal-training-budget superiority, convergence, statistical significance, or a proven causal mechanism.
- This experiment covers four fixed 64-frame sources and two anchor QPs. Existing checkpoint training exposure follows v1/v2; no strict unseen-data claim is made. FID was not computed.
