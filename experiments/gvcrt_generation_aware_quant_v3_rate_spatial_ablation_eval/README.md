# v3 rate and spatial multiplier ablation

Inference-only evaluation using the immutable v1 A and v2 H11 step 300 checkpoints.
The native codec stream is wrapped in a fixed 16-byte experimental header. No
checkpoint, source, or v1/v2 result is modified.

Run with the existing `gvc-rt` Python environment:

```bash
bash run.sh
```

`driver.py` resumes from task JSON and verifies each stream SHA before skipping
it. It queues on physical GPUs 4–7, one task per GPU, waiting for any card
used by another process. Logs, codec streams, and caches are in
`server_artifacts/` and are ignored by Git. Any failed task stops its phase;
fix the failure and rerun the same driver command.

The `preflight.json` gate records the code and weight hashes. The driver
refuses to start if `task.py` or `ablation.py` changes after that gate.

Completed outputs include 36 new protocol full evaluations, 16 reused QP2/7
anchors, and 5 reused QP9 supplements. All 8 spatial rate matches use alpha=1;
no extra rate-search encodes or selected-alpha full evaluations were needed.
Preflight separately used 3 full metric runs and 2 nonunit-alpha rate-only
encodes (39 full metric runs including preflight, below the limit of 44).

`map_driver.py` only decodes persisted streams. It saves hashes for all 64
references, P latents/qmaps and RGB tensors, verifies original v2 receiver
parity for every reused stream, and renders the fixed frames. `audit_delivery.py`
checks those records, A/IDENTITY codec-state equality, source hashes and rate
selection before `make_report.py` writes the report. `environment.json` records
the existing gvc-rt Python environment and plotting dependency versions.

`task_manifest.csv` supplies canonical task keys with source, checkpoint,
runtime and configuration SHA prefixes. The original short storage names remain
aliases, bound to full SHAs in per-task configurations. Resume checks verify
those full hashes and the persisted stream before accepting a result.
