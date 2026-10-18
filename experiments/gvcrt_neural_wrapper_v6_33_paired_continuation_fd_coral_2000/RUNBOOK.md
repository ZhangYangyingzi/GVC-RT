# v6.33 execution

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_33_paired_continuation_fd_coral_2000/launch32b.py
```

The launcher respects the driver flock. Never delete a lock to replace a live process. Driver recovery verifies PID/start ticks/command and retains peak memory reservations. Each branch migrates its own verified full training checkpoint only when no new latest exists; subsequent recovery restores the new latest transaction without optimizer/EMA reset.

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_33_paired_continuation_fd_coral_2000/check_once.py
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_33_paired_continuation_fd_coral_2000/watch_event.py
```

Status reads import no torch and do not scan large directories. Watcher emits one hourly summary or an immediate failure/terminal event. The persistent driver handles migration,three branch continuations,1500/2000 evaluations,aggregation,audits and publication. Failed tasks need a documented repair before requeueing; OOM updates memory admission and retries the same committed cursor. `pipeline_status.json` separates calculation/report/publication.

Logs: `logs/driver.log`,per-worker logs,`errors.jsonl`,`gpu_schedule.jsonl`,branch `optimizer_calls.jsonl` and receipts. Checkpoints: `branches/<branch>/checkpoints/latest.pt` and fixed global milestones. Source rows are immutable in `plans/shared.json`; `plans/continuation.json` labels the second exposure. `migration_receipt.json` and `migration_equivalence.json` prove restored states. No new calibration/prefill is executed.

Large artifacts remain on this server and are indexed by path/size/SHA. Published large text is split on complete lines to512KiB; `reassemble.py` verifies original SHA through `file_index.json`. Final commit receipt is server-only `publication_status.json` to avoid self-reference. Historical B lacks a separate buffer map: every named buffer must be present in its model state_dict before migration; no unknown buffer is silently regenerated.
