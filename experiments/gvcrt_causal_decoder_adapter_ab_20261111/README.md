# Frozen B300 causal decoder adapter A/B

The task protocol is `protocol.json`. Execution facts, tables and fixes are in
`REPORT.md`; current progress is in `status.json`, `readable_status.json` and
`queue/evaluation_progress.json`.

Use `/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python` from this directory.
`common_s2.env()` puts temporary files and generated caches under this experiment.
Only physical GPUs 0, 1, 2 and 3 are authorized.

To resume an interrupted execution, run `python -B -u resume.py` with host GPU
access, then `python -B -u finalize.py` if the finalizer is absent. Check the PID
registry first. Existing trainers are adopted; missing trainers load the latest
atomic adapter/AdamW/RNG/data-cursor checkpoint. Updates are counted only from
committed receipts. Do not delete or overwrite the saved checkpoints.

GPU0 trains A and GPU1 trains B in independent processes. GPU2/3 produce bounded,
shared raw-h training caches and handle development scoring. Source datasets,
past experiments, and original weights are read-only. Missing task-owned codec
caches are regenerated using independent Original code in a subprocess; training
and evaluation namespaces cannot silently substitute an adapted model for Original.

`adapter.py` implements the new branch. `experiment_model.py` restores the complete
B300 chain and freezes it. `loss.py`, `calibrate.py`, and `train.py` implement only
the requested objectives and fixed budget. `deployment_check.py` exercises actual
RANS decoding through the new adapter and the existing head LoRA. `preflight.py`
checks engineering correctness without a quality launch threshold.

`results.py` produces development CSVs and curves. `media.py` exports fixed native
FFV1 clips and unscaled ROIs. `integrity.py` checks committed data cursors, source
hashes, per-point files, media and identical rate streams. `publish.py` stages only
this experiment and rejects weights, large caches and videos before a normal push.

Weights, datasets, generated caches and large FFV1 videos stay local. Their paths
and hashes are recorded in the manifests. A fresh clone requires the pinned local
dependencies named in `dependency_imports.json` and `source_manifest.json`, plus
the original I/P checkpoint links under `runtime_tree/checkpoints/`.

This is the ULong development stage. No final-test run, model selection rule,
automatic research continuation, or claim of experimental effectiveness is added.
