# Fixed v3 entropy-adaptation trend protocol

Four logical controls: N0 (reused v1 A/v2 evaluation, G only), N1 (new G+Phi, no H), Q0 (reused v2 H11, G+H), Q1 (new G+Phi+H). No spatial visual condition V. Only fixed comparisons N1−N0, Q1−Q0, Q1−N1. No H00/H10/H01 repair ablation is repeated.

Common Generator/Bridge origin is G_basic_100 with SHA256 89c1a5baa195c8ffc9695af495d43389e00021b926eae13a0337d3b2eb117fd4. N1 and Q1 do not start from H11 step300. Original I/P weights come from the verified v2 weight manifest; Q1 H comes from the indexed H11 step0. initialization_manifest.json and checkpoint_index.json give actual server paths, not guessed worktree-relative paths.

The fixed v2 data_manifest.json and train_plan.json are byte copies. New training uses only the eight selected U-Long sources, seed20261023, batch1,256×256 crop,16-frame windows, original native time axes. Keep original window/QP coupling and report it as a limitation. Historical inherited G_basic exposure, including UVG Beauty, is disclosed by data_manifest.json; do not claim unseen UVG exposure.

I0 initialization; P1–11 no-grad warmup; P12–15 supervised and averaged1/4; cross-frame reference detach. Actual INDEX_MAP=[0,1,0,2,0,2,0,2], native qp_shift=[0,2,1], effective offsets=[0,2,0,1,0,1,0,1]. The inherited four-item description is not a mod4 execution rule.

Objective LPIPS + 0.1383589248954108 reconstruction_L1 + beta_q(schedule_s1p0,external_QP)*estimated_bpp, with live y+z CDF probabilities. No DISTS/FD/CORAL/GAN training loss or loss search. AdamW G=1e-6,Phi=1e-6,H=1e-4(Q1 only),betas0.9/0.999,eps1e-8,weight_decay1e-4,global clip1.0. Each new group has exactly300 effective optimizer updates; snapshots0/100/300. Preflight uses and saves the first two plan updates; after required checks pass they count toward300, and continuation restores step2 optimizer/RNG/cursor/Phi. No extra effective smoke updates or training extension.

Both new groups use corrected skipped-symbol STE; Q1 uses exact H11 coordinates. q_base is the lower-bounded original output; q_new includes H, lower bound and actual dtype conversion; rho=q_new/q_base. Pass1 means/scales divide rho; reconstructed pass1 u0 multiplies rho before the original-domain spatial prior; pass2 means/scales divide rho; latent reconstructs with q_new. All common parameters come from the current Phi. H-less N1 still allows the original quant_step to adapt through Phi.

Every new measured point persists a real native RANS file, frees the encoder model (weak-reference check), builds a new decoder, reloads complete Phi/G/H, rebuilds CDFs, then checks all64 causal references, P integer symbols, latent and qmap. CDF hashes and forward/master Phi hashes are recorded on both sides. No new per-video side information or RANS algorithm is introduced.

Evaluation:64 frames,native I,IP=-1,unchanged v2 source indices/color/size. Step0/100/300: UVal0/UVal1, externalQP2/7. Step300: ParkScene/Jockey,QP2/7. Fixed preview frame15,rows200:456,columns400:656,QP2,step300. This is not selected by output quality.

Actual total bytes, headers,I/P payload and actual total bpp are separated; kbps=total_bits*20/64/1000. Estimated y/z bits are analytic, actual RANS payload is combined and is not falsely split. P-rate comparison uses only actual P1–63 payload and corresponding estimated P y+z bits. Inherited evaluation z estimate uses native get_z_bits (p+1e-5); inherited training uses clamp_min1e-9. Both always read current Phi; neither is an integer-table gradient surrogate.

For each sequence, list observed-point dominance (per metric and joint) before interpolation within actual measured overlap. No extrapolation or formal BD-rate from two points. Supplements depend only on actual rate bracketing, externalQP0–9, at most3 extra QPs per method/sequence (including imported extra endpoints). No quality-based checkpoint selection. FID=not_computed.

Preflight behavior multipliers .95/1/1.05 are global preset diagnostics, not learned branches or video-specific side information. They are not a finite-difference STE derivative test; complete causal rate need not be monotone. Their files are separate from the declared R-D tasks.

Default physicalGPU4=N1,5=Q1,6=checks/auxiliary,7=real evaluation. After both branches complete300, released GPUs4/5 may share the fixed remaining evaluation queue with GPU7; actual assignments are recorded in resource_events.jsonl; occupancy polling queues behind existing jobs and never terminates them. Branch/GPU/runner locks protect recovery. Code/results are written only under this new experiment in its own worktree/branch. Histories remain read-only.
