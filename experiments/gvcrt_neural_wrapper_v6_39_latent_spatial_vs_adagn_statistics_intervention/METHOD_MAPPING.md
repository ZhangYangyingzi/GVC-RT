# Method mapping

| Alias | Full name | Spatial latent | Statistics latent | Source |
|---|---|---|---|---|
| M00 | Y_originalBridge_fullG | original z0 | original z0 | v6.37 endpoint |
| M10 | C_zF_spatial_z0_stats | trained zF | original z0 | v6.39 inference |
| M01 | S_z0_spatial_zF_stats | original z0 | trained zF | v6.39 inference |
| M11 | F_fullLoss_noP_500 | trained zF | trained zF | v6.37 endpoint |

All four rows use the identical frozen `G_F` generator. `M10` and `M01`
exchange the AdaGN statistics source without altering the original generator
implementation. They are inference interventions rather than newly trained
models or new bitstreams.
