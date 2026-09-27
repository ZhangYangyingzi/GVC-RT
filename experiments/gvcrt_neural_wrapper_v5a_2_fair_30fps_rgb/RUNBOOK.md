# V5-A.2 Fair Cross-Dataset Evaluation

Evaluation only: three frozen methods, 8 fixed U-Long videos, 7 fixed UVG sequences, QP 0–9.
Canonical input: 64 lossless RGB24 PNGs at 1920x1080; evaluation duration 64/30 seconds.
U-Long frames 0–63; UVG frames 0,4,...,252 with the audited BT.709 conversion.

Run prepare_protocol.py on CPU to verify source hashes, canonical frames, and loader identity.
Then run run_pipeline.py using the existing gvc-rt Python environment from the repository directory.
The queue resumes fresh encoding on GPUs 4 and 6, then generates reports and audits.
It waits if either GPU has an unrelated workload. All point writes use exclusive file locks.
Initial copied V5-A.1 artifacts are archived under legacy_import_not_v52_results.
Only final_integrity PASS plus pipeline_status PASS/complete mean V5-A.2 is complete.

