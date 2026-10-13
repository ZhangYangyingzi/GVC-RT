# V6.29

Run from `/Huang_group/zyyz/Projects/GVC-RT` with the existing `gvc-rt` environment.

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_29_fid_audit_feature_covariance_control/setup.py
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_29_fid_audit_feature_covariance_control/launch_pipeline.py
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_29_fid_audit_feature_covariance_control/run_pipeline.py --status
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_29_fid_audit_feature_covariance_control/launch_cpu_diagnostics.py --launch
```

Restart the same scheduler command after an exited scheduler. Its exclusive lock prevents duplicates. It adopts live workers by PID and process-start identity and reserves model-loading memory immediately. Training restores optimizer, model, all RNG states and the plan cursor. Validated completed points are skipped. Other GPU processes are never stopped. Only GPUs 4–7 are allowed.

The source is v6.22 B1 step1000. The v6.26 plan is byte-exact. No discriminator is used. Calibration is forty training-only windows; lambda is frozen before formal training. FID auditing and training are independent, but final FID claims require all audits to pass.

Large artifacts remain on the server. Text shards restore with `restore_text_shards.py`; hashes and indices are published from an isolated latest-main worktree without force push.

The optional CPU launcher overlaps completed-dataset diagnostics with the GPU queue, at most four workers with two matrix threads each. Check `cpu_diagnostics_status.json` and `logs/early_cpu_diagnostics.log`. Finalization reuses only hash-validated diagnostic outputs; per-job locks prevent duplicate calculations.

Fixed lambda_stat:0.038636700918173825, calibrated on40 training windows. CONFIG_RESOLVED.json resolves the frozen pre-registration config and calibration file. A uses proved V6.26 reuse; B is fresh training. Expected main logical RD count1840:1380 verified reuse and460 fresh.

The inherited projected_adv/adv log keys now hold covariance statistics, not an adversarial loss; see audits/log_field_definitions.json and stat_gradient_summary.csv. No discriminator is initialized or called.
