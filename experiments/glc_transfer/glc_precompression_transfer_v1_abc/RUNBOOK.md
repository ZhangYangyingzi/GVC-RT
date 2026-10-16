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

Final evaluation uses `evaluation_stage.py`: initial cap four workers, up to two per card, subject to an expanded measured-peak budget including process-level CUDA memory. The mandatory free-memory reserve remains `max(2 GiB, 10% total)`. Resource preemption cools down that card for five minutes, reduces concurrency and allows at most three resource requeues per point; the separate OOM/reference-mismatch retry remains limited to one. GPU resource waits are bounded at one hour and an evaluation run at 48 hours. Frozen RGB inputs are fully rechecked using four CPU threads, with progress in `audits/evaluation_source_progress.json`.

After ten minutes without resource preemption or an encode retry, the dispatcher may raise its worker cap by one, up to four, still subject to the same measured-memory admission. Resource cooldowns, retry counts and elevated memory budgets are retained across resumes. Recovery events are recorded in `audits/concurrency_recovery_*.json`.

Successful points do not lower an elevated admission budget. Observed free-memory shortfalls also add a GPU-specific admission margin (2–8 GiB), retained on resume, so a volatile card does not repeatedly receive tasks solely because its instantaneous free memory briefly rose. Resource-only cap reductions stop at two; this does not force two workers when insufficient memory exists. OOM/reference failures may still reduce the cap to one. Neither adjustment changes encoder, decoder, frame pools, quality metrics or loss settings.

Publishing uses the existing GitHub identity over the official `ssh.github.com:443` transport, strict trusted-host-key checking and ordinary pushes. The remote remains `git@github-zyyz:ZhangYangyingzi/GVC-RT.git`. Pushes are bounded at 1800 seconds per attempt and three attempts per publication stage. No global Git or SSH configuration is changed by the publisher. Prior timeout/resource-stop records remain in `audits/`.
