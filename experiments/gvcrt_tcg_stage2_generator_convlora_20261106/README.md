# Generator Conv-LoRA

One LoRA-only line from strictly loaded pilot+B1500. Five rank4 branches,59392 parameters; Router absent and all original parameters frozen.

Run/resume (the agent executes this entry):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u /Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_stage2_generator_convlora_20261106/experiments/gvcrt_tcg_stage2_generator_convlora_20261106/runner.py
```

State: status.json. Results: REPORT.md. Publication status is separate. Historical SHA blocker is preserved in audits/sha_correction_20261106/.
