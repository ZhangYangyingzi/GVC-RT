# Measured rate-quality review — 2026-11-05

Superseded for dataset reporting by [separate dataset and individual-video curves](../rd_by_dataset_20261105/README.md). The historical_reference figure below pools six ULong, five HEVC-B, and two UVG videos; it must not be presented as a curve for any one dataset. See the new [tensor-level baseline audit](../rd_by_dataset_20261105/BASELINE_FINDINGS.md) for the identity and precision of parent0.

These figures summarize the existing final measurements in ../../metrics_v3.csv. No training, decoding, new QP measurement, model selection, or source-metric changes were performed.

- New held-out ULong: 32 videos, shown in new_holdout_test_rate_quality.png.
- Historical reference: 13 videos, shown separately in historical_reference_rate_quality.png.
- rd_points.csv contains the plotted absolute values; same_rate_changes.csv contains the paired mean changes.
- build_rd_review.py regenerates the figures and CSVs. audit.json records the source and output hashes. PDF exports are local-only.

The horizontal axis is measured bits per pixel (bpp), displayed on a logarithmic scale. Each dataset point is the equal-video arithmetic mean at one external QP (0, 5, or 9); the underlying video rates differ. The five models have identical stream hashes, real bytes, bpp, and kbps within each video/QP pair. This is an equal-rate comparison between models on matched content, not a claim that every video was encoded at the dataset mean rate.

LPIPS, DISTS, FloLPIPS, and GT-flow warping error are lower-is-better; PSNR is higher-is-better. Each line has only three measured points. Connecting lines are visual guides, not additional measurements or fitted performance claims. No BD-rate or bitrate-saving estimate is made. The GT-flow metric uses PWC estimates from reference frames and includes reconstruction/warping error; it is not an isolated measure of flicker.

Model labels: parent0 is the experiment's zero-residual TCG baseline; parent5000 is the trained TCG parent. A_common and B_common each contain 1500 additional optimizer updates. A_best_dev contains 500 additional updates. B_common is also B_best_dev (absolute step6500), whereas B's stopped checkpoint is absolute step8000 and is not the selected model. The independently selected A/B best models use different update budgets.

On the new test set, B+1500 versus parent0 at QP0/5/9 changes FloLPIPS by -9.710%/-13.519%/-13.336%. Versus parent5000, the incremental changes are -0.609%/-0.762%/-0.769%. Improvements are metric-specific: PSNR and GT-flow warping error deteriorate relative to parent0. Historical results remain separate and must not be inferred from the new ULong curve.

Percentages are 100*(mean_target/mean_reference-1), computed within a fixed split/QP. They are not averages of per-point percentages and are not percentages of bitrate saved.

Publication allowlist: this README, the plotting script, two PNG figures, two CSV tables, and audit.json. The existing training/evaluation protocol and original report remain unchanged.
