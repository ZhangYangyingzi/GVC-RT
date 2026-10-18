# Runbook

The reproducible entrypoints are `launch_screen.sh`, `driver.py`, `evaluate32.py`, and `aggregate.py`. Python is `/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python`. The driver is resumable and task receipts are atomic; completed receipts are reused only when their task signature and artifact hashes match.

The evaluation manifest records six fixed v6.34 diagnostic-plan source windows. Each input frame is decoded from its source MP4 and checked against the saved full RGB hash before encoding. Each worker performs real encoding, then independent decode and frame/transition metrics using the inherited validated adapter. Physical GPU assignment and reserved memory are recorded in `gpu_schedule.jsonl`.

To check progress without loading the model environment, run `python check_once.py`. To re-run completed-safe work, run `launch_screen.sh`; receipt validation skips valid points. Aggregation is deterministic and can be repeated after all 90 receipts pass.
