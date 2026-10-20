# Fixed objectives

All new training is no-P receiver-only: D_basicDISTS_noP_500, 500 fresh AdamW updates from the verified original step0. B and F are read-only historical endpoints; O and all diagnostic combinations have zero new updates.

- B: LPIPS + 0.1383589248954108 reconstruction_L1 + beta(q) rate_bpp + 0.01 proxy_L1.
- D: B + 0.5 DISTS.
- F: D + 0.038636700918173825 legacy_DISTS_spatial_stat + 0.0044560228041899182 native_frame_FD.

D changes only the LPIPS fused branch of v6.36 basic_crop to LPIPS+0.5DISTS. DISTS explicitly uses require_grad=True. Reconstruction backward keeps the historical4096 scale/unscale and weight/order. After averaging P49–P63, one global norm clip1 and one optimizer.step occur. P1–P48 warmup, native I0, detached references and all actual QPs retain their source implementation. The identity proxy has zero proxy_L1; receiver-only rate can have no BG gradient and is not altered here.

The basic objective is not the unpublished full official GVC-RT training objective. F−D is the conditional effect of the remaining statistical-supervision scheme, including native-resolution FD and reference statistics. It is neither an isolated FD-term comparison nor an equal-compute/equal-supervision-exposure experiment. No new training FD, legacy_stat, Inception, EMA, reference-prefill or full-resolution VJP is executed for D. Read-only step0 auditing evaluates the inherited F crop function solely to verify its common scalar subterms; this is separate from formal D training and produces no updates.

D−B isolates the added DISTS term under this shared500-update setup. All other comparisons follow the task's fixed list, without treating metric reductions as bitrate savings or summing differences into contribution percentages.
