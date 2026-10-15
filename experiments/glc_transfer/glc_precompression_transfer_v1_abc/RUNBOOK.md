# Commands

Run from `/Huang_group/zyyz/Projects/GLC` with the existing glc environment.

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /Huang_group/zyyz/home_dir/.conda/envs/glc/bin/python -B -u experiments/glc_precompression_transfer_v1_abc/run_pipeline.py
```

Use the same command to resume. Inspect `status.json`, `branches/*/status.json`, `logs/`, `BLOCKED.json` and `publication_status.json`. Resume checkpoints remain in `checkpoints/resume/`; deployment sender weights are in `final_P/`.
