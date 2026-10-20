# v6.38 Bridge/Generator factorial trend

O: original receiver, zero updates. B: Bridge only. G: Generator only. BG: joint. Each trained condition starts from the same v6.37-verified original step0 with fresh AdamW and receives exactly100 updates including the first2 smoke updates. Checkpoints0/50/100; only100 evaluated. No P, FD, legacy stat or new teacher.

The fixed historical plan is filtered to its first two encountered training identities, retaining the first100 matching rows (1 U-Long,99 Beauty). This deterministic two-video screen preserves existing source rows, crops, frame indices and QPs. The unfiltered historical first100 rows contain49 identities and cannot satisfy the two-video restriction.

Evaluation: BasketballDrive and BQTerrace,64 native frames, external QP0/5/9, native I then causal P, IP=-1. All methods freshly decode the same6 immutable audited real RANS streams. LPIPS/DISTS/FloLPIPS and L1 are frame/transition means; PSNR uses aggregate RGB MSE, with mean-frame PSNR retained separately. No FID requested.

Run computation with the GVC-RT conda Python: `python -B -u runner.py` in this folder. Then run `python -B -u finish.py` to validate, plot and publish; it may also run in a separate terminal and wait for the computation. Completed published runs exit without more training or evaluation. Uses physical GPU4–7 only when at least10GiB free, one native task per GPU. Failed tasks get at most2 retries. Durable locks, per-update committed checkpoints, receipts and task exit events allow restart. No v6.37 task is rerun.

Artifacts: `branches/*/receipts/`, `logs/`, `tasks/`, `results/`, `audits/`. Large checkpoints remain server-side and are excluded from Git. `protocol.json`, `source_hashes.json`, `manifests/streams.json`, `plans/first100.json` fix dependencies and input identity.

Single seed,100 updates,two test videos: descriptive mechanism trend only, without significance or generalization claims.
