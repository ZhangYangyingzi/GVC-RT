# Interim development results

Snapshot UTC: 2026-11-04T08:38:50.990207+00:00
Snapshot Asia/Shanghai: 2026-11-04T16:38:50.990207+08:00
Status: INTERIM_DEVELOPMENT_SNAPSHOT_NOT_FINAL_TEST. Training/evaluation/publication of the full experiment are not complete.
Reference: common parent step5000.32 development videos,64 native frames,QP0/5/9,96 paired points per model. All compared streams and actual bpp match pointwise.
Overall means equally weight96 sequence/QP points; these represent32 videos, not96 independent videos. A/B equal-budget comparison is additional500. A additional1000 is a trajectory state.
Directions: PSNR and MS-SSIM higher; LPIPS,DISTS,FloLPIPS and warping errors lower. No visual-quality, convergence, generalization or statistical-significance conclusion is included.

| State | bpp | kbps | PSNR | MS_SSIM | LPIPS | DISTS | FloLPIPS | WarpingError | WarpingError_ReconFlow_PWC | WarpingError_GTFlow_PWC | Normalized FloLPIPS M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parent5000 | 0.008291843 | 501.794404597 | 25.587885187 | 0.887053172 | 0.134200877 | 0.081985270 | 0.130787771 | 0.040335727 | 0.040335727 | 0.040285324 | 1.000000000 |
| A_clean_old_pool_add500 | 0.008291843 | 501.794404597 | 25.540138395 | 0.886557086 | 0.133989723 | 0.082319106 | 0.130599144 | 0.040571268 | 0.040571268 | 0.040521366 | 0.998302853 |
| B_expanded_ulong_add500 | 0.008291843 | 501.794404597 | 25.524211745 | 0.886570696 | 0.134605046 | 0.082490312 | 0.131070895 | 0.040710999 | 0.040710999 | 0.040667043 | 1.002399472 |
| A_clean_old_pool_add1000 | 0.008291843 | 501.794404597 | 25.540114412 | 0.886678909 | 0.134393050 | 0.082341420 | 0.130810872 | 0.040548710 | 0.040548710 | 0.040506621 | 0.999550309 |

Full point metrics: metrics.csv. QP breakdown: summary_by_QP.csv. Paired point deltas vs parent5000: paired_deltas.csv. Sources and hashes: summary.json.
These are development results. Final held-out tests and fixed viewing samples remain pending.
The main REPORT.md remains the live supervisor snapshot; this separate snapshot does not change training, stopping rules, plans, RNG or selected checkpoints.
