# Fixed intervention protocol

`G_F` is the deployed generator from the fixed v6.34b N_noP_BG_FD step-500
checkpoint. `z0` is the original bridge output and `zF` is the trained bridge
output, both extracted by the historical v6.37 verified replay from the same
decoded codec feature. All receivers are frozen and in eval mode.

| Label | Full method | `conv_in` | AdaGN statistics input |
|---|---|---|---|
| M00 | Y_originalBridge_fullG | z0 | z0 |
| M10 | C_zF_spatial_z0_stats | zF | z0 |
| M01 | S_z0_spatial_zF_stats | z0 | zF |
| M11 | F_fullLoss_noP_500 | zF | zF |

The worker calls the unchanged decoder forward with `z_spatial` and the actual
`q_recon`. Five short-lived `forward_pre_hook` callbacks replace only the
second AdaGN argument with `z_stats`. The original `var(dim=-1)` default
correction, `sqrt(var + eps)`, mean, gamma/beta linear layers, GroupNorm,
spatial padding, clamp, pixel shuffle, and output conversion remain in the
original module. Each forward verifies all five hook visits, then removes the
hooks in `finally`.

Fixed evaluation: v6.37 manifest; 6 U-Long, 2 UVG holdout and 5 HEVC-B
videos; 64 source frames each; external QP 0/5/9; original native I0 and
63 P frames; 1080p valid area; original RGB conversion and actual frame QP.
One of 39 previously verified complete RANS streams is read per video/QP.
Actual bpp includes every persisted byte and is divided by 64 × 1920 × 1080.
The 20 fps kbps field is a display conversion only.

Both historical endpoint outputs are recomputed on the intervention path and
checked against v6.37 raw reconstruction and Inception feature hashes, spatial
metrics and pooled FID. Bitwise equality is preferred. If only numerically
equal, the prespecified limits are RGB maxabs 1e-5, relative L2 1e-6,
LPIPS/DISTS/FloLPIPS absolute 1e-5, pooled FID absolute 1e-3, and all
four methods must then use recomputed outputs. A larger mismatch blocks
mechanism conclusions.

Each video/QP uses the same stream for all four methods. The five fixed
quality contrasts are Q10−Q00, Q11−Q01, Q01−Q00, Q11−Q10 and Q11−Q00.
The interaction is Q11−Q10−Q01+Q00, always in raw quality units. Negative
quality difference means lower distance. No quality difference is reported
as saved bits, BD-rate or an independent interpolation replicate.
