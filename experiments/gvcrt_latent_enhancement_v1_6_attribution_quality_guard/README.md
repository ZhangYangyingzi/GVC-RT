# GVC-RT V1.6 - Attribution and Quality Guard

V1.6 is a bounded, zero-training analysis over the frozen V1.5 candidate set. It adds exact per-frame MSE/LPIPS guards, gain attribution, matched-total-rate comparison to historical `ORIGINAL_FP32`, source attribution, and transition diagnostics.

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python run.py --output results_v1
```

The command runs the complete pipeline: input audit, Train2 repeatability/tolerance freeze, guarded DP, real ORC2 serialization, independent source-blocked reception, accounting and sanity checks, matched-rate analysis, diagnostic videos/timelines, final verification, and report generation. It performs zero network updates and zero latent/code optimization updates.
