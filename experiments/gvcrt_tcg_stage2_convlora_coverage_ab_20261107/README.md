# Generator Conv-LoRA coverage A/B

Fresh shared5 branches,A59392/B135168. Frozen B1500 TC6500,old generator/codec/Bridge. No Router.

Agent-run/resume entry:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u /Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_stage2_convlora_coverage_ab_20261107/experiments/gvcrt_tcg_stage2_convlora_coverage_ab_20261107/runner.py
```

Actual status:status.json. Do not launch a second runner while runner.lock is held. Final results:REPORT.md,metrics.csv,plots/. Existing test set is reused.
