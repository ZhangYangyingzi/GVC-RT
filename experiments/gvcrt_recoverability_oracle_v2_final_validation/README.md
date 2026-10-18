# GVC-RT oracle v2 final validation

Frozen codec and weights; no training. Four datasets, calibration QP2/7, and preselected independent video identities. See audit/v2_design.md and configs/experiment.json for the protocol registered before new GPU work.

After `src/prepare.py` has completed, run `/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B src/launch.py` from this directory. The launcher serializes scheduler ownership with a file lock, preserves exit codes, and runs finalization and publication after the resumable driver exits. Only GPUs 4-7 may run workers, one worker per GPU with at least 7000 MiB free. Per-attempt stdout/stderr, GPU snapshots and actual exit codes are saved under logs/. The 0.5% one-sided rate band includes every control byte; no empty strategy is a valid candidate. No rate- or quality-based tolerance relaxation is performed.

Reused calibration records remain at their immutable v1 server paths with SHA256 indexes. New streams, symbols, features and representative lossless reconstruction videos are under artifacts/ (server-only; indexed for reproduction). Raw per-video, per-strategy metrics and all three random seeds are exported under metrics/ and oracle/. See manifests/run_status.json for actual state. Dataset and model files are excluded from Git. No research discussion is included.

After finalization, run `/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B src/verify_final.py` for independent checks of persisted stream sizes, control-byte accounting, dataset separation, the shared calibration candidate pool, every random seed, policy freeze timing, rate-only test QP selection, and worker exits. Its receipt is `manifests/independent_postflight_verification.json`.

Final bounded-run status: PASS. New successful GPU jobs: 39. Exact independent experimental pairs: 0. Approximate under-budget pairs: 0. Eligibility, skips and all raw metrics are saved in oracle/rate_allocation/strategy_results.csv and rate_matching_status.json. Driver and worker exit codes are in logs/GPU_task_exit_status.json.
