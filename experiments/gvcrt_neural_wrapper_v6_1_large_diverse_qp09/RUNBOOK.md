# V6.1 large/diverse U-Long QP0–9

Only this directory may be written. Existing experiments, sources and checkpoints are read-only; their dependency hashes are audited. GPU 0–3 are not selected.

1. Preflight verifies the selected V4.1 step20000 model, optimizer, losses, crop/clip, STE/real-coding QP rules and threshold 0.12.
2. Profile all archived U-Long candidates with four uniformly spaced adjacent-frame pairs, small source-statistics images, no optical-flow/pretrained models; retain raw per-video statistics and errors.
3. Reserve a 32-video U-Long validation cohort first, retain all 256 training files, fill underrepresented 4x4 temporal/texture strata to 1024 unique training SHA256s.
4. Restore weights and optimizer, perform all 40,000 additional updates on GPU4 using the original single-GPU optimizer semantics. Evaluate saved checkpoints on GPU5–7 using real RANS.
5. Select solely from validation, then evaluate held-out U-Long, UVG and fixed VIRAT720 QP0–9. Verify any baseline reuse machine-readably; VIRAT baseline must be recoded with threshold .12.
6. Write numerical reports, plots, diagnostics and final integrity; no interpretation or discussion.

Training remains clip4, crop256, batch1, unchanged P/B/G, beta, lambda_proxy, AdamW, learning rates and gradient clipping. Training uses the pre-existing direct external QP rule; real coding retains shift_qp offsets. Do not turn this into DDP or alter update semantics.

Runtime packaging fix: shared Python3.12 has setuptools60.2.0, which raises pkgutil.ImpImporter AttributeError during torchvision/triton import. An existing compatible setuptools75.1.0 is copied only into runtime_dependencies and prepended by v61_io. Torch2.6.0+cu124 and torchvision0.21.0+cu124 are unchanged. runtime_dependency_audit.json records file hashes. The original error is retained in logs/initialization_smoke_attempt1_error.log. QP0/QP9 initialization/backward checks take zero optimizer updates.
