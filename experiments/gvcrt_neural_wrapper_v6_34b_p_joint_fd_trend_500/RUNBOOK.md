# Runbook

Environment: `/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python`.

`prepare34b.py` verifies source checkpoint transactions and the 90 v6.34a point receipts. `continue_driver.py` restores each branch and runs its remaining updates to cumulative step 500, then evaluates the 36 new endpoint points and aggregates all 126 logical points. The driver is resumable and records process and task state in this directory. `check_once.py` reports status without importing PyTorch.

Only physical GPUs 4–7 are admitted. The local run used GPU 6 for N and GPU 7 for P; evaluation workers use 6/7. Large checkpoints, bitstreams and feature tensors remain on the server and are indexed by absolute path and SHA256.
