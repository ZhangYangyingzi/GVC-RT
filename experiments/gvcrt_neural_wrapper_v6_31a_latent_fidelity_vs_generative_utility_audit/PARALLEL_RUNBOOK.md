# Additional MCL parallel evaluation

The original driver remains responsible for aggregation, figures, report,
integrity and normal Git publication. Evaluation settings and the original
evaluator files have not changed.

`parallel_extension.py` suspends only this experiment's original GPU7 MCL
worker. Its current and next videos are excluded. Future videos have one
exclusive worker each, using the unchanged evaluator and original task IDs.
Actual physical GPU IDs are recorded separately in point receipts.

GPU4–7 can each host multiple workers. Every launch immediately reserves its
full predicted peak. Initial reservation is 12000 MiB plus 1800 MiB margin;
full-point measurements refine admission, with a 9500 MiB minimum reservation.
OOM raises the admission estimate and retries incomplete points. No sequence,
precision, model or metric settings are changed.

The extension waits for auxiliary workers before resuming the original MCL
worker. An independent guardian also waits for recorded auxiliary processes
before restoring the original worker if the dispatcher exits unexpectedly.
The original pipeline then skips verified completed points and publishes.

## Status files

- `parallel_extension_status.json`: dispatcher, guardian, active PIDs,
  reserved memory, pending videos, retries and restoration state.
- `parallel_extension_plan.json`: video partition and unchanged protocol hash.
- `logs/parallel_gpu_schedule.jsonl`: admission and exit records.
- `logs/parallel_mcl*_gpu*_try*.log`: evaluator logs per video.
- `worker_status/parallel_video*.json`: completion and full-point peak memory.

Do not launch individual `--worker` processes while the original MCL worker is
running. Do not delete locks to take over an active dispatcher. The controller
holds an exclusive lock and verifies the original worker identity.

To launch the controller only when no extension is active:

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u \
  experiments/gvcrt_neural_wrapper_v6_31a_latent_fidelity_vs_generative_utility_audit/parallel_extension.py
```

The current controller was launched in an independent background session.
No unrelated processes are suspended or terminated.
