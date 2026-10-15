# Commands

Run from `/Huang_group/zyyz/Projects/GLC` with the existing glc environment.

Persistent launch or resume (checks for an existing driver before starting a new screen session):

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /Huang_group/zyyz/home_dir/.conda/envs/glc/bin/python -B experiments/glc_precompression_transfer_v1_abc/launch.py
```

Foreground execution or resume:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /Huang_group/zyyz/home_dir/.conda/envs/glc/bin/python -B -u experiments/glc_precompression_transfer_v1_abc/run_pipeline.py
```

Use the same command to resume. Inspect `status.json`, `branches/*/status.json`, `logs/`, `BLOCKED.json` and `publication_status.json`. Resume checkpoints remain in `checkpoints/resume/`; deployment sender weights are in `final_P/`.

Read-only monitoring:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /Huang_group/zyyz/home_dir/.conda/envs/glc/bin/python -B -u experiments/glc_precompression_transfer_v1_abc/wait_for_terminal.py
```

`early_original.py` schedules independent A points on GPU4–7 while B/C train. It admits up to two A workers per card and six overall only when measured memory plus the required reserve fits, including not-yet-allocated peak memory for existing workers. It stops dispatching before either training branch reaches update850 and drains owned workers; the main pipeline checks and reuses its completed points. Its PID, tasks and progress are in `early_original_status.json`; admission measurements are in `audits/parallel_admission.json`.
