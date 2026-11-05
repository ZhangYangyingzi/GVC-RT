# Router-only TCG spatial gating — 20261105

Baseline is the audited B+1500 partial TCG checkpoint, cumulative TCG optimizer step6500. The complete frozen receiver is reconstructed strictly from original I/P codec files and the pilot step0 receiver, followed by the exact B+1500 TCG overlay. It is not parent5000 or an independently original GVC-RT model.

Only the new 33889-parameter Router is optimized. See protocol.json for architecture, loss/VJP precision, causal history, immutable data schedule, stopping and evaluation definitions. Runtime and unmodified reference copies are recorded in audits/source_copy.json and source_manifest.json.

The persistent entry is runner.py. It holds a lifecycle lock, reruns only missing engineering phases, preserves atomic committed checkpoints and receipts, executes complete development validation every250 updates, freezes selection, then runs135 final real RANS evaluations with independent decoders. Source/window/QP/stream/frame/metric audits govern all original/B+1500 metric reuse.

Launch or resume, from this directory:

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u runner.py
```

`run_status.json`, `active_processes.json`, `training_status.json`, `stopping_state.json` and `logs/runner.log` give timestamped execution records. Static reports do not establish process liveness. Check the lifecycle lock and matching PID/start time. No duplicate runner may be launched.

Checkpoints, streams, caches and viewing previews remain local and are excluded by the explicit publication allowlist. Scientific conclusions and next-stage recommendations are outside this execution report.
