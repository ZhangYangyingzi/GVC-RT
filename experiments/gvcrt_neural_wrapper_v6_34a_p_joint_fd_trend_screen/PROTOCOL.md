# v6.34a fixed trend screen protocol

Execution mode: **A (reuse)**. The v6.34 N/P branches have intact step 0, 100, and 250 full training checkpoints with matching index hashes. This run adds no optimizer updates. The former 5000-step run was no longer active when inspected; its last driver snapshot showed N=459 and P=355 committed updates, but the budget override does not label that run complete.

The fixed endpoint is step 250 for both branches. Step 100 is included only as an early trend point. The six evaluation videos are the first six unique source identities, in order, in the old v6.34 reserved U-Long diagnostic manifest. The same 64 canonical source frames are used for all methods, checkpoints, and external QP 0, 5, and 9. This is a training-pool holdout screen, not a claim of historical unseen data or cross-domain generalization.

Methods are `original`, `N_noP_BG_FD` (identity P; receiver weights from the matching N checkpoint), and `P_PBG_FD` (zero-residual identity-initialized NeuralWrapper plus jointly trained receiver weights). For each branch, step 0 is a logical alias of `original`; it is not encoded twice. Physical points are 18 original + 36 N100/N250 + 36 P100/P250 = 90. No full 46-video evaluation is run.

Real RANS encoding and independent decode use the v6.34/v6.32 validated runtime, native I0, IP=-1, 64 frames, inherited QP phase and deployment precision. Rate is complete bitstream bytes including headers/SPS/I divided by 64×1920×1080. The three-QP curves are descriptive: per-video LPIPS/DISTS/FloLPIPS use linear interpolation on ln(actual bpp), 20 points, no extrapolation, only on each comparison's common interval. Pooled FID uses all 384 reconstruction/reference frame features and is shown against mean actual bpp; it is not a mean of per-video FID.

The inherited model, crop/FD loss, optimizer, seed, and training plan are unchanged. This is a single-seed short trend screen. No checkpoint selection or budget extension is performed.
