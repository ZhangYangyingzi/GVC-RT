# Loss and pipeline difference audit

## Reused complete objective

The actual v6.34b implementation computes a 15-target-frame mean over the existing crop objective and adds native-resolution FD. Its logged crop terms are LPIPS, `0.5 × DISTS`, `beta(q) × rate/65536`, `0.01 × proxy L1`, and `REC × reconstruction L1` with `REC=0.1383589248954108`, plus `STAT × legacy DISTS spatial-statistic loss` with `STAT=0.038636700918173825`. The FD weight is `0.0044560228041899182`. Source code and configuration hashes are recorded in `source_hashes.json`.

In the no-P branch the wrapper is `Identity`: proxy L1 is identically zero. Whether the rate term has receiver-side gradients is determined by its actual graph; the receiver-only adaptation did not modify the entropy model. The P branch uses the existing zero-residual-initialized NeuralWrapper and updates P, bridge, and generator. The no-P branch updates bridge and generator. Both preserve the inherited 64-frame causal path, native I bypass, P1–48 warmup, P49–63 supervision, and detached per-frame references.

## Published objective evidence and unknowns

The paper describes Stage II/III terms including rate, reconstruction, latent distribution matching, and a Stage III adversarial term. The paper gives a cosine/margin distribution description and some scalar settings, but a complete receiver-appropriate reproducible training objective is not available in the checked official repository snapshot: the repository README leaves training-code release unchecked. Missing items include exact per-QP lambda/normalization, exact margin construction and teacher interface, verifiable original teacher checkpoint/preprocessing, and Stage III discriminator initialization/update implementation. See `reference_objective_feasibility.json`.

Accordingly, `A_refLoss_noP_500` is blocked before optimizer construction/update. It is not replaced with v6.22 B1, v6.33 loss, v6.26 proxy teacher, or a loss with terms set to zero. This prevents attributing N–original or P–original differences exclusively to the objective.

## Pipeline changes represented in the fixed comparison

- `original` is the public I/P model and original receiver.
- `N_oursLoss_noP_500` uses identity P and the adapted bridge/generator at cumulative 500.
- `P_oursLoss_withP_500` uses the existing P plus adapted bridge/generator at cumulative 500.
- Both trained models came from the same original initialization and common training trajectory definition documented by v6.34b. The P/N global clipping interaction is inherited as-is; no compensating learning rate or clip change was made.
- The fixed evaluation is on 13 listed videos and QP 0/5/9 only. It is a screening comparison, not a claim of broad unseen-domain coverage.
