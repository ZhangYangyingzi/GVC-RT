# 基础损失与加入 DISTS 的对照

This experiment is **not a reproduction of the unpublished complete official GVC-RT training objective**. The repository README has not marked official training-code release complete; v6.37 explicitly distinguishes this basic objective from the official full objective. Results cannot establish official-model undertraining.

- O_original: original Bridge and Generator; zero updates.
- G_basic: LPIPS + 0.1383589248954108 reconstruction_L1 + beta_q(schedule_s1p0) rate_bpp + 0.01 proxy_L1.
- G_dists: G_basic + 0.5 DISTS (require_grad=True).

Identity wrapper, frozen I/P compression and entropy networks, frozen Bridge and quality networks. Only Generator receives FP32 gradients and fresh constant-lr AdamW updates. The inherited rate estimate is logged and its Generator gradient is verified zero; this is not rate optimization. Identity proxy_L1 is exactly zero. DISTS may be logged as an observation for G_basic but is absent from its fused training objective.

Native I0, P1–48 no-grad warmup, P49–63 supervised and averaged over 15 frames; detached references and no BPTT. The LPIPS/DISTS fused VJP and reconstruction VJP retain the inherited separate 4096 scale/unscale, FP32 merge, one global clip and one optimizer.step. Common step0 scalar terms, objective difference=0.5 DISTS, nonzero finite Generator DISTS VJP and zero rate/proxy VJP are checked without optimizer updates.

Fixed comparisons: G_basic−O_original, G_dists−O_original, G_dists−G_basic. Total objectives differ and are not directly comparable as measures of training quality. Perceptual relative changes are not bitrate savings; PSNR deltas are dB.

Single seed, 100 updates, two fixed training windows, two test videos: descriptive trend experiment only. No mechanism attribution or research-innovation judgment. Historical v6.38 G-only uses a different 99:1 plan and sampling; any reference to it is noncausal and cannot isolate the effect of changing to 50:50.
