# Fixed A2000 half-gain: QP0/5 completion

New64points:32development videos at externalQP0/5,gain0.5. Reuse32verifiedQP9half points and192endpoints. No training/optimizer/new checkpoint/finaltest/newgain.

Run/resume (first verify runner.lock and matching PID identity):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u /Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_stage2_convlora_coverage_ab_20261107/experiments/gvcrt_tcg_stage2_convlora_A2000_qp05_gain_20261108/runner.py
```

Status:status.json;logs:logs/runner.log;protocol:protocol.json;sources:sources.json. An existing live runner must not be duplicated. Publication retry alone:/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u /Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_stage2_convlora_coverage_ab_20261107/experiments/gvcrt_tcg_stage2_convlora_A2000_qp05_gain_20261108/publish.py. All new files under this directory; SOURCE and QP9_SOURCE are read-only. Summary and figures are per externalQP; no pooled mean or automatic final selection.
