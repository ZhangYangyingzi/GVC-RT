# TCG flicker pilot A/B/C/D

Snapshot UTC: 2026-11-09T06:59:16Z

Scope:12 existing ULong development sources, external QP0/5/9. No final test evaluation. Four independent single-GPU models.

Initialization: original A2000 LoRA over pilot receiver + B+1500 FP32 TCG. Fresh AdamW. Both training/evaluation inference gain0.5.

A: TCG+LoRA; B: A+Bridge mlp[3]; C: A+fixed causal original-reconstruction RAFT history warp; D: C+Bridge mlp[3].

Main temporal metric: MCResidual_Sobel_RAFT_pixelcenter_v1_PtoP. I→P is separately recorded. Historical PWC metrics keep their prior sampler.

Standard CUDA grid_sample backward may use nondeterministic atomic accumulation; own-repeat relative gradient error is audited under1e-3. No bitwise gradient reproducibility claim.

|Group|Committed updates|Videos/groups/windows consumed|Stop reason|
|---|---:|---|---|
|A|31|124/124/124|FAILED: RuntimeError('EMPTY_GT_VALID_REGION_NOT_A_ZERO_ERROR')|
|B|31|124/124/124|FAILED: RuntimeError('EMPTY_GT_VALID_REGION_NOT_A_ZERO_ERROR')|
|C|31|124/124/124|FAILED: RuntimeError('EMPTY_GT_VALID_REGION_NOT_A_ZERO_ERROR')|
|D|31|124/124/124|FAILED: RuntimeError('EMPTY_GT_VALID_REGION_NOT_A_ZERO_ERROR')|

## Validation records

|Group|Step|Median per-video Sobel ratio|Videos ratio<1|Numeric screen|
|---|---:|---:|---:|---|
|Original|0|0.9826902006274723|8|NOT_PASSED|
|A2000|0|1.0|0|NOT_PASSED|
|A|0|1.0|0|NOT_PASSED|
|B|0|1.0|0|NOT_PASSED|
|C|0|1.0027490866313036|0|NOT_PASSED|
|D|0|1.0027490866313036|0|NOT_PASSED|

Full per-video/QP/frame/transition values: development_metrics.csv and evaluation/*/step_*/. Raw and weighted training terms plus component gradient norms: training_history.csv and compressed receipts.

The numeric screen is a pre-registered pilot threshold; it is not a significance test or visual acceptance record. No automatic expansion or deployment.

Unavailable/failed stages remain incomplete in status.json and final_integrity.json. Videos and checkpoints are local only; manifests contain SHA256.
