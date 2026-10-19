# Objective comparison

The new B branch uses the explicitly nested basic objective: LPIPS + 0.1383589248954108 × reconstruction L1 + beta(q) × rate_bpp + 0.01 × proxy L1. With Identity P, proxy L1 is exactly zero.

The reused F endpoint adds 0.5 × DISTS, 0.038636700918173825 × the inherited DISTS spatial statistic, and 0.0044560228041899182 × the inherited native-frame FD term. These three terms form one combined extra-supervision package here; this design does not separate their individual effects.

This experiment does not call the basic target an official objective or claim to reproduce the unpublished original full training loss. F is a read-only existing endpoint; only B receives 500 new updates.
