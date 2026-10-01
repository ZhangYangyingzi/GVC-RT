# V6.2-A full-QP objective calibration

Execution-only protocol for request 676e6fe7-eef6-441b-aa4c-8e8923e590db.

- Authoritative configuration: `fullqp_config.json` (the older `config.json` belongs to the superseded fixed-beta sweep).
- Four schedules plus the fixed-beta aligned-QP control; 3000 paired updates each.
- Optional new beta0 omitted to prioritize required branches. Legacy beta0 outputs remain untouched and excluded.
- `superseded_fixed_sweep_setup/` retains earlier setup files; `revision_audit.json` lists invalidated audit claims.
- Normal/Hard cohorts: 32 videos each, first 32 original-resolution frames, native source FPS for actual-byte rate accounting.
- No UVG, VIRAT, or final eight U-Long evaluation, selection, or training. No final model selection.
- Lambda endpoints come from the paper; the linear interpolation formula is explicitly the user's calibration specification, not an assertion of unpublished official code.
- All six checkpoints of all five required branches receive independent full real-RANS validation. Step0 is not silently skipped or relabeled.
- Queue only uses free GPUs among physical 4, 5, 6, 7; other jobs are never preempted.
- Artifacts may be partial while status is RUNNING. Only `final_integrity.json: PASS` means the full request completed.

Run from repository root:

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_2_objective_calibration/run_fullqp.py --launch
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_2_objective_calibration/run_fullqp.py --status
tail -f experiments/gvcrt_neural_wrapper_v6_2_objective_calibration/logs/fullqp_pipeline.log
```

Per-branch progress is in `branches/<name>/training_status.json`; all per-update losses and sample plans are in `training_log.jsonl`.
Real RANS and frame metrics are under `bitstreams/fullqp/`, `features/fullqp/`, `parts/fullqp/`. Numerical rate-distortion CSVs are under `results/fullqp/<branch>/step_<update>/<normal|hard>/`.
Failures are recorded explicitly in `fullqp_failures.json` or `fullqp_pipeline_failure.json`; never infer completion from an idle GPU.
