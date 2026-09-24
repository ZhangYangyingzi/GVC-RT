# GVC-RT V1.6 Attribution and Quality Guard

## Conclusion

V1.6 completed the frozen-candidate, zero-training validation on Train2 (2 clips) and Val6 (6 clips). It performed 0 network updates and 0 code/latent optimization updates. The principal guarded point is `M4_O_FRAME_GUARD, alpha=0.5`.

On Val6 IP, that point covers 6/6 videos, uses 47.5926% enhancement/base by per-video macro average (47.7045% pooled), and changes ZERO by PSNR +0.278584 dB, MS-SSIM +0.001521, LPIPS -0.007861, and `d=MSE+0.001*LPIPS` -0.000120284. Thus the guard preserves an average objective and perceptual improvement, unlike unrestricted M4's LPIPS change of +0.000782.

## Q1: Where Does The Gain Come From?

At the same `alpha=0.5` budget upper bound, macro-averaged per-video IP differences are:

| Attribution | A minus B |
|---|---|
| Frame allocation (`M2-M1`) | PSNR +0.138899 dB, MS-SSIM +0.001010, LPIPS -0.009099, d -0.000067995, enhancement bytes +4215.3 |
| Frame allocation with optimized pool (`M4-M3`) | PSNR +0.477728 dB, MS-SSIM +0.000663, LPIPS +0.001098, d -0.000270099, enhancement bytes +4686.5 |
| Optimized candidates beyond E (`M4-M2`) | PSNR +0.343969 dB, MS-SSIM -0.000269, LPIPS +0.009881, d -0.000206574, enhancement bytes +568.3 |

These are same-budget-upper-bound comparisons, not equal-actual-rate comparisons. Both methods' actual bytes are retained per video in `Val6_method_attribution.csv`.

## Q2: Does The Joint Quality Guard Work?

Yes for the stated finite candidate pool and per-frame constraint. The principal point improves every selected nonzero frame against that frame's ZERO in both MSE and LPIPS, while averaging PSNR +0.278584 dB and LPIPS -0.007861 against ZERO. The Val6 non-Z filter counts are {'ALLOWED': 162, 'MSE_ONLY_DEGRADES': 24, 'LPIPS_ONLY_DEGRADES': 84, 'MSE_AND_LPIPS_DEGRADE': 0}. Compared with unrestricted M4, the guard changes PSNR -0.204284 dB, MS-SSIM +0.000781, LPIPS -0.008643, d +0.000154284, enhancement bytes -104.3.

The guard does not prove that no other sequence-level jointly improving solution exists. A failure would only characterize this frozen finite pool. MS-SSIM was not constrained and is not required to be monotone.

## Q3: How Does It Compare At The Same Total Rate?

All 6 principal Val6 points lie inside each video's measured `ORIGINAL_FP32` QP0-QP3 overlap. Per-video piecewise-linear interpolation versus `log(total_bpp)` gives final-minus-original macro means: PSNR +2.300004 dB, MS-SSIM -0.007178, LPIPS +0.090062. Before enhancement, ZERO-minus-original-QP0 macro means are PSNR +2.733381 dB, MS-SSIM +0.019042, LPIPS +0.070056. No point was extrapolated.

The guarded enhancement therefore does not close the full gap to original GVC-RT at matched total rate on these clips, despite improving its own fixed ZERO path.

## Q4: Is The Signal Source-Specific And Temporally Safe?

The fixed cyclic WRONG_SOURCE diagnostic covers 38 nonzero principal selections. Replacing target symbols by another video's same-label symbols changes target quality by PSNR -1.181033 dB, LPIPS +0.016468, and d +0.000530187; this supports source-specific information rather than a generic injection effect.

Mean I-to-P `T_error` changes from 0.036126132 (ZERO) to 0.035032602 (guard). Mean P-to-P changes from 0.022423873 to 0.024848853. `T_error=mean(abs(e_t-e_(t-1)))` is reported without motion alignment; it is a short-GOP diagnostic, not proof of long-term stability. Per-transition values and switch neighborhoods are in `Val6_temporal_timeline.csv` and `Val6_switch_neighborhood_temporal.csv`.

## Integrity And Artifacts

- Guard tolerance: exactly 0.0 after 120/120 Train2 repeated candidate measurements had max MSE and LPIPS absolute error 0.0.
- Real streams: 11 Train2 and 33 Val6 logical nonzero streams; 36 unique stream contents independently received.
- Independent receiver: every reconstructed RGB frame hash matched; source access was actively blocked; base DPB state remained unchanged.
- Negative tests: truncated, trailing-byte, CRC-corrupt, and wrong-base-bound streams were all rejected.
- Videos: 6 synchronized 16-frame principal reconstructions in `results_v1/videos/`.
- Final verification: `passed=true` in `results_v1/final_verification.json`.
- Input provenance: `results_v1/input_hash_manifest.json`.

No next-stage training was started.
