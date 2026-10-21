# v6.40 基础损失与加入 DISTS 的对照

Single seed 20261034; Generator-only 100 committed updates per condition; Identity wrapper; frozen compression core, Bridge and quality networks. This is not the official complete training objective.

Training: Beauty and the specified U-Long identity, exactly 50 updates each, five per video×QP. One fixed historical window per video with 50 different 256×256 crops. Two conditions share all ordered input records. Only step100 is evaluated on BasketballDrive/BQTerrace, QP0/5/9: 18 points.

## Run and recover

From this directory, use `/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B prepare.py` once to generate/audit the fixed plan, then `python -B -u runner.py`. `prepare.py` verifies an existing plan rather than replacing it. Do not rerun preparation after publication because publication status is an output of that phase.

The runner first executes a step0 diagnostic without optimizer updates, then two committed smoke updates per branch, then resumes each branch from step2 to100. Physical GPUs4–7 only, one experiment GPU task per card, at least10GiB actual free memory at launch. Other tasks are never stopped. Each update atomically commits model, optimizer, buffers, full RNG, cursor and receipt before publishing the committed ledger record. Branch/task/runner locks guard duplicates. Recovery replays only uncommitted work and records discarded physical calls. Checkpoints0/50/100 are kept; no checkpoint selection.

Run `python -B -u finish.py` to wait for computation, validate, summarize, plot and publish. It can run concurrently with the runner. Publication uses the existing origin remote and a separate detached worktree, staging only this experiment. Heavy weights, source video and caches remain server-side; Git contains path/size/SHA256 indexes.

See PROTOCOL.md, protocol.json, OBJECTIVE_COMPARISON.md, plans/counts.json and audits/ for exact protocol and checks. Final outputs: evaluation.csv, paired_differences.csv, summary.csv, REPORT.md, figures/, final_integrity.json, run_status.json, publication_status.json. No mechanism claims, official undertraining claims, or significance/generalization inference.
