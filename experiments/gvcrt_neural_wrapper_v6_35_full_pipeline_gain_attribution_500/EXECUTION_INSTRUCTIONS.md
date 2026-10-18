# Execution instructions

- Fixed method identities and step-500 checkpoint hashes are recorded in `source_hashes.json` and `audits/source_and_checkpoint_verification.json`.
- `prepare35.py` creates the immutable 13-video manifest, verifies all source checkpoint exports and historical original points, and records the 75 verified reuses.
- `run_queue35.py` is the only writer for new evaluation receipts. It evaluates only the fixed 42 N/P UVG/HEVC missing points. It is idempotent and validates task signatures/artifact metadata before skipping.
- `aggregate35.py` produces all-point, pooled-FID, common-rate tables and plots after all 42 new receipts pass.
- `render_examples35.py` decodes the preregistered QP5 frame 16/32/63 center crops from the saved streams; it does not encode or affect metrics.
- `finalize35.py` requires 42 new passes, 75 verified historical rows and all 27 pooled FID rows before marking computation PASS.
- Do not run two scheduler writers concurrently. If interrupted, inspect `pipeline_status.json`, active worker PIDs and GPU schedule entries before resuming.
