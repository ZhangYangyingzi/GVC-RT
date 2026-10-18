# Commands

Environment: `/Huang_group/zyyz/home_dir/.conda/envs/glc/bin/python`.

Run/resume: `python -B launch.py` from this experiment directory.

State: `python -B monitor_progress.py`.

Checkpoints and optimizer/RNG states remain under `checkpoints/resume/<branch>/latest.pt`. Completed receipts are validated before reuse. Publication uses a separate checkout and normal pushes to personal main.

After all four branches finish, `launch.py` verifies final checkpoint hashes, bound
training signatures and all 4000 receipts, then resumes at training publication
without repeating training or rewriting frozen calibration artifacts.

Optional CPU-only baseline FID precomputation (same hash-keyed cache as report):
`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -u precompute_baseline_fid.py`.
Monitor: `tail -n 5 logs/baseline_fid_precompute.log`.
