# Read-only reuse sources

- v6.37 replay receipts: `experiments/gvcrt_neural_wrapper_v6_37_loss_bridge_generator_mechanism_500/tasks`. Each of 39 tasks verifies the persisted RANS stream, source RGB, decoded feature, q_recon, and endpoint output/feature hashes. Source code SHA256: `0252f152cb605ad2de3372f6d92bcf4af62d13c5dc705951126236732ad2ecad`.
- v6.39 endpoint audit: `experiments/gvcrt_neural_wrapper_v6_39_latent_spatial_vs_adagn_statistics_intervention/audits/endpoint_equivalence.json` SHA256 `8c8cb1974ecd7548041a595d2e682a72f2baca076aa9e9fabd620faf193e917e`. This is a read-only cross-check only; v6.41 does not depend on its numerical results.
- Original GVC-RT bridge and generator: `checkpoints/GVC-RT_P.pt`, loaded by v6.37 original runtime and checked against A00.
- BG-trained bridge and generator: v6.34b step500 inference checkpoint SHA256 `ca2c66aa1d68877779c8646b66d02e4ba06e82fc4f3b3c92c11271fc7ffc19dc`.
- G-only full-loss500 was not found as a fully matching preexisting run. v6.38 and v6.40 have 100-step schedules and different objectives, so neither is used as a substitute.
- Reuse is validated per task, not by similar method names. Original O, Y and F are separately sourced; 39 unique streams are reused and no encode occurs.
