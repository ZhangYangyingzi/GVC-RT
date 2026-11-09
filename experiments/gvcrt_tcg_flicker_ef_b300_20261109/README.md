# GVC-RT E/F from B300

The task controller is `runner.py`. Run with physical GPU visibility 0,1,2,3 and PCI bus ordering. E uses0/1; F uses2/3. `status.json` records the stage; `live_pipeline_status.json` joins optimizer steps, next windows, READY count/bytes, cumulative committed cache wait, producer throughput and first-step latency. `REPORT.md` contains a factual snapshot; `complete=false` remains until training, evaluation, videos, integrity and result publication all finish.

The incremental cache producer initially started one GPU child; after explicit user authorization now runs at most one child on each GPU4/5/6/7, publishes only hash-verified atomic tensors and never changes the immutable1200-window order. Startup requires16 distinct windows (the first8 also calibrate F), followed by32-window bounded lookahead. Training rank0 waits for its next4 windows at an optimizer boundary, broadcasts readiness and both ranks handle2 windows. Explicit gradient SUM uses global loss denominators before one global norm clip. Cache eviction protects the minimum of both atomic-checkpoint committed cursors. Requests for missing development points are on demand; inherited compatible caches are read only.

Model loading follows the parent audited original/pilot/TCG/LoRA chain, then overlays every B300 trainable tensor LAST, including Bridge mlp[3]. The original Bridge and decoder become trainable with34,774,916 total parameters. Training masters stay FP32; deployment follows the parent's FP16 backbone/Bridge and FP32 TCG/LoRA arithmetic. Checkpoint loading casts once into that inherited execution precision; FP32 checkpoint values are retained intact on disk.

Only the predeclared losses, learning rates, sample plan, budget and validation/stop rules are used. No final test, automatic deployment or method-quality interpretation is performed. Videos, cached tensors, weights and streams are local only. GitHub publication uses an explicit small-file allowlist.

The detached task-private `supervise.py` watches only this experiment. It adopts active jobs, recovers abrupt interruptions from atomic checkpoints, and stops on recorded diagnostic failures. It ends after verified results_final publication; it never starts another experiment.

After interruption, resume with:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u /Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_flicker_ef_b300_20261109/experiments/gvcrt_tcg_flicker_ef_b300_20261109/runner.py
```
