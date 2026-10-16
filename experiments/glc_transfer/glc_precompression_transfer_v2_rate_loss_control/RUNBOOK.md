# Commands

Environment: `/Huang_group/zyyz/home_dir/.conda/envs/glc/bin/python`.

Run/resume: `python -B launch.py` from this experiment directory.

State: `python -B monitor_progress.py`.

Checkpoints and optimizer/RNG states remain under `checkpoints/resume/<branch>/latest.pt`. Completed receipts are validated before reuse. Publication uses a separate checkout and normal pushes to personal main.
