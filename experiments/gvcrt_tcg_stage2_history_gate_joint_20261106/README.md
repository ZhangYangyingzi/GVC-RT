# GVC-RT internal history gate + Router/TCG joint

This is the single joint line prescribed on20261106. B+1500 (cumulative TCG6500) is loaded exactly, including the trained fusion heads. Router is new at seed20261106 and uniform M0=.2. No previous Router or optimizer is reused.

The mask multiplies projected history before concat with the retained current projection. It is spatial and shared across all three TCGs. There is no outer residual gate. Router/TCG parameters and internal tensors are FP32 with autocast disabled; outputs return to the incoming dtype. The codec, Bridge and remaining generator retain historical precision and remain frozen.

Only Router33889+TCG137760 parameters train. Fresh AdamW groups use1e-4/1e-5, fixed hyperparameters and combined clip1.0. Four synchronized GPUs average the four sealed samples (2UVG+2ULong). Protocol, source hashes, sample plan and engineering proofs are explicit local artifacts.

Normalization uses new independent B+1500 inference with FP32 TCG. Historical FP16 B metrics remain a separate reference. Initialization does not count toward patience. Valid development checks every250updates determine raw best and the prespecified anchor/patience rule, with cap3000. Final selection is frozen before new final tests. Raw-score selection and the0.001 threshold flag are separate factual records.

Launch or resume only after checking the lifecycle lock and actual PID/start time:

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u runner.py
```

The runner performs mandatory disposable engineering checks, fresh formal initialization, matched baseline/step0 evaluation, training/validation, frozen selection, matched and joint final evaluation, diagnostics, plots, integrity and publication. Checkpoints contain both Router and TCG, optimizer, four RNG states, cursor, receipts and persisted validation state. Immutable evaluated checkpoint files are not rewritten by validation-metadata commits.

`run_status.json`, `training_status.json`, `stopping_state.json`, `active_processes.json` and `logs/runner.log` are timestamped records. COMPLETE denotes local experiment completion; `publication_status.json` separately records result publication and verified commits. No final PASS is asserted for incomplete points. No scientific effectiveness or convergence conclusion is produced.

All new writes stay under this project worktree. Historical experiments are read-only. Publication uses an explicit task-only allowlist; weights, media streams, large caches, PDFs and credentials remain local. Original/historical results are reused only after identity/window/stream/aggregation audits. Matched baseline points are freshly independently decoded and scored, never relabeled historical metrics.
