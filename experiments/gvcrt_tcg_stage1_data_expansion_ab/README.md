# Fixed TCG paired continuation and GT-flow scoring

Execution protocol: [EXECUTION_PROTOCOL.md](EXECUTION_PROTOCOL.md).

The authoritative current state is `run_status.json`. A/B budgets are assigned only after the complete ZIP qualification, historical exclusion audit and native window verification; see `protocol.json` when frozen. `runner.py` resumes completed stages and committed optimizer updates without replaying them. All historical model/data/experiment inputs are read-only.

Run/resume from this worktree with host CUDA access:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_tcg_stage1_data_expansion_ab/runner.py
```

The runner tracks an already active inventory or GT-flow scorer using exclusive locks. It does not start a second copy of the same task. Inspect `logs/runner.log`, `logs/events.jsonl`, `inventory/status.json`, A/B `training_summary.json`, evaluation `run_status.json`, `final_integrity.json`, and `publication_status.json` for stage-specific completion. Missing or incomplete stages are not PASS.

Parent GT-flow scoring can resume separately with the same Python executable and `gtflow.py`; it only reads verified parent streams and checkpoints. A/B training starts only after `audits/plans_frozen.json`, GPU preflight and numerical equivalence audits pass.

Public numerical outputs are `metrics.csv`, `paired_deltas.csv`, `warping_error.csv`, domain/QP/split summaries, numerical point counts, `training_loss.csv`, `evaluation/` per-frame/per-transition records, figures and `REPORT.md`. Complete compressed per-update rank receipts are in each arm's `receipts/`. No quality-based checkpoint selection or scientific interpretation is performed.

Checkpoints, source media, flows, streams and JPEG viewing previews remain local and are excluded from Git. JPEG previews are explicitly lossy and never used for reported metrics. Public large JSONs may be stored as `.json.gz`; decompress to the corresponding `.json` filename before reproducing a stage. Each published file is below 50MiB.
