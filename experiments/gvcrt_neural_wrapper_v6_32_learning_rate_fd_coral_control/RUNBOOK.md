```bash
cd /Huang_group/zyyz/Projects/GVC-RT
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_32_learning_rate_fd_coral_control/launch32.py
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_32_learning_rate_fd_coral_control/check_once.py
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_32_learning_rate_fd_coral_control/wait_for_terminal.py
```
Resume uses the same launch command, locks, task signatures and latest optimizer/RNG/EMA transaction.

2026-10-15 takeover verified the existing screen driver PID 187912 and parallel
evaluation PID 256916; no second driver or training process was launched.
See audits/takeover_20261015.json for current code signatures, checkpoint
optimizer groups and independently decoded source dimension/hash checks.
Fixed plan indices 165, 253, 477 and 720 have source heights 960, 1038, 810
and 816. Their complete full-resolution reference, calibration and C/D
dependencies are BLOCKED under the fixed protocol. A/B crop training and its
formal evaluations continue. Incomplete EMA references are not used.
The same wait_for_terminal.py process waits for report and publication as well
as computation. A terminal BLOCKED record is an incomplete experiment, even
when available A/B results and the execution report were successfully published.
