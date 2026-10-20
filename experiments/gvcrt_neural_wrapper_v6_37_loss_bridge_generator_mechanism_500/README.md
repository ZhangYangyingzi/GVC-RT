# v6.37 loss / bridge / generator mechanism, fixed500 updates

Only D_basicDISTS_noP_500 is newly trained. Original O, historical B/F and read-only X/Y/I diagnostics receive zero new optimizer updates. See OBJECTIVE_COMPARISON.md and MECHANISM_PROTOCOL.md for fixed definitions, source hashes and limitations.

Run/resume from this experiment directory with the gvc-rt Python environment: `python -B runner37.py`. A single runner lock, separate training/task locks, atomic checkpoints and durable worker launch/exit records support recovery. Never extend beyond500 updates or evaluate intermediate weights for this task.

State and provenance: effective_config.json, protocol.json, source_manifest.json, checkpoint_reuse_manifest.json, evaluation/task_manifest.json, runtime/, audits/, errors.jsonl. Final coverage and completion status are in final_integrity.json. Results and figures are generated only from successful task receipts; required failures/blockers are reported explicitly. Large checkpoints, tensors and example sources remain on the server in SHA-indexed artifacts; only compact results, plots, code and records are published.
