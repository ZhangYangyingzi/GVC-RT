# V6.18 long-horizon error propagation audit

Repository: `/Huang_group/zyyz/Projects/GVC-RT`.
Experiment: `experiments/gvcrt_neural_wrapper_v6_18_long_horizon_error_propagation_audit`.
No model training or optimizer updates are performed.

## Prepare

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_18_long_horizon_error_propagation_audit/prepare.py
```

Preparation verifies full step1000 checkpoints and their inference exports against historical indexes/configuration, all 16 canonical 64-frame sequences, native fps/rate semantics and input SHA256s. It takes a SHA256 snapshot of historical V6.15/V6.16/V6.17 files and all referenced dependencies/assets.

## Execute or resume

Run with access to NVIDIA devices. GPU4/5/6/7 are the only allowed devices.

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_18_long_horizon_error_propagation_audit/run_pipeline.py
```

The pipeline checks live GPU memory, audits the actual frozen codec and metric modules, then verifies historical full causal real-RANS artifacts. Reuse requires exact checkpoint hash, frame pool, QP mapping, settings, metrics, real stream accounting, independent decode and synchronization integrity. Missing or mismatched points use the same historical evaluator in this experiment directory. Existing verified points are skipped on resume. OOM reduces concurrency, selects another allowed GPU where possible and retries. An unfinished point without valid JSON/integrity is never counted as complete. Concurrent controller invocation is prevented by a lock. No unrelated process is terminated.

The diagnostic uses external QP0/4/9 and 64 causal frames for each of five methods over 8 U-Long, 2 UVG holdout, 1 UVG validation and 5 HEVC-B sequences: 240 points and 15360 frame rows. All frame indexes in exported tables are one-based. FloLPIPS is assigned to the target frame of each historical transition; frame1 is NaN. Its early window therefore has 7 valid samples. Population std uses ddof=0. Drift uses late49–64 minus early1–8; excess drift subtracts the original drift at the same external QP. Raw and delta OLS slopes use frames2–64 with an intercept. Dataset summaries and plotted curves weight sequences equally. This is a diagnostic same-external-QP comparison.

## Status and outputs

```bash
cat experiments/gvcrt_neural_wrapper_v6_18_long_horizon_error_propagation_audit/pipeline_status.json
tail -n 20 experiments/gvcrt_neural_wrapper_v6_18_long_horizon_error_propagation_audit/logs/pipeline.log
cat experiments/gvcrt_neural_wrapper_v6_18_long_horizon_error_propagation_audit/final_integrity.json
```

Raw points and reuse manifests are in `evaluation/`. Numeric CSV/JSON tables are in `results/`. Raw and delta frame curves are in `plots/`. `final_integrity.json` requires all points, complete metrics, permitted NaN placement, checkpoint/core hashes, recurrent state and before/after historical SHA equality. `gpu_schedule.json` records GPU snapshots, worker allocations, OOM/retry history and status.

## Publish after integrity PASS

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_18_long_horizon_error_propagation_audit/publish.py
```

Publication stages only this experiment directory. A path-limited commit preserves unrelated staged/working changes. Weights, optimizer states, bitstreams, feature caches, raw video and temporary files are excluded. The current branch's configured GitHub remote/branch is used with an ordinary push. A push failure is saved with the local commit retained in `publication_status.json`.

Commit message: `experiment: add v6.18 long-horizon error propagation audit`.

## Remote main advanced during publication

The original experiment-only commit is retained in the source worktree. A detached publication worktree inside `.publication_worktree/` applies only that commit on the latest origin/main, then stages only this experiment directory and pushes HEAD:main without force. The publication checkout itself is ignored and never uploaded.

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_18_long_horizon_error_propagation_audit/audits/republish_only_experiment.py
```

The helper refuses to overwrite an existing publication checkout. Publication status, both original/published commits and remote visibility checks are saved in `publication_status.json`.
