# Runbook

Python: `/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python` (PyTorch 2.6.0+cu124, CUDA available).

1. `prepare35.py` verifies source manifests, step-500 indices, checkpoint files, historical records and builds fixed manifests/receipts.
2. `run_queue35.py` runs only missing N/P tasks for UVG holdout and HEVC-B. The scheduler reserves 8500 MiB per loading/running process before launch, caps at two per GPU, and checks physical GPU free memory. Each point is written to a task receipt and unique bitstream/feature files.
3. After the queue reaches a terminal state, `aggregate35.py` creates the canonical 117-point table, pooled FID, same-rate tables and figures.
4. `finalize35.py` checks point/artifact counts and writes `final_integrity.json`, `REPORT.md`, and `RESULTS_SUMMARY.md`.

Resume: rerun `run_queue35.py`. A point is skipped only if its receipt signature and artifact metadata validate. Do not run a second queue writer concurrently. No model training is part of this protocol.
