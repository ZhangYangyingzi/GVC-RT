# TCG flicker pilot A/B/C/D

Snapshot UTC: 2026-11-09T04:21:17Z

Scope:12 existing ULong development sources, external QP0/5/9. No final test evaluation. Four independent single-GPU models.

Initialization: original A2000 LoRA over pilot receiver + B+1500 FP32 TCG. Fresh AdamW. Both training/evaluation inference gain0.5.

A: TCG+LoRA; B: A+Bridge mlp[3]; C: A+fixed causal original-reconstruction RAFT history warp; D: C+Bridge mlp[3].

Main temporal metric: MCResidual_Sobel_RAFT_pixelcenter_v1_PtoP. I→P is separately recorded. Historical PWC metrics keep their prior sampler.

Standard CUDA grid_sample backward may use nondeterministic atomic accumulation; own-repeat relative gradient error is audited under1e-3. No bitwise gradient reproducibility claim.

|Group|Committed updates|Videos/groups/windows consumed|Stop reason|
|---|---:|---|---|
|A|0|0/0/0|NOT_STARTED|
|B|0|0/0/0|NOT_STARTED|
|C|0|0/0/0|NOT_STARTED|
|D|0|0/0/0|NOT_STARTED|

## Validation records

|Group|Step|Median per-video Sobel ratio|Videos ratio<1|Numeric screen|
|---|---:|---:|---:|---|

Full per-video/QP/frame/transition values: development_metrics.csv and evaluation/*/step_*/. Raw and weighted training terms plus component gradient norms: training_history.csv and compressed receipts.

The numeric screen is a pre-registered pilot threshold; it is not a significance test or visual acceptance record. No automatic expansion or deployment.

Unavailable/failed stages remain incomplete in status.json and final_integrity.json. Videos and checkpoints are local only; manifests contain SHA256.
