# V6.30 execution

No model updates. Historical inputs are read-only.

Start/resume (one process lock):
```bash
cd /Huang_group/zyyz/Projects/GVC-RT
screen -dmS gvcrt_v630 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_30_fid_cache_closure_sample_robustness_audit/run_all.py
```
Read-only check (no torch):
```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_30_fid_cache_closure_sample_robustness_audit/check_once.py
```
Logs: `/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_30_fid_cache_closure_sample_robustness_audit/logs/`. Normal automatic publication uses a separate latest-main worktree. `DONE.json` records completed computation independently from publication status.
