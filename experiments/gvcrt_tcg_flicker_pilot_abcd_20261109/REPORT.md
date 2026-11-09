# TCG flicker pilot A/B/C/D

Snapshot UTC: 2026-11-09T13:11:39Z

Scope:12 existing ULong development sources, external QP0/5/9. No final test evaluation. Four independent single-GPU models.

Initialization: original A2000 LoRA over pilot receiver + B+1500 FP32 TCG. Fresh AdamW. Both training/evaluation inference gain0.5.

A: TCG+LoRA; B: A+Bridge mlp[3]; C: A+fixed causal original-reconstruction RAFT history warp; D: C+Bridge mlp[3].

Main temporal metric: MCResidual_Sobel_RAFT_pixelcenter_v1_PtoP. I→P is separately recorded. Historical PWC metrics keep their prior sampler.

Empty GT-valid regions are recorded as missing, never as zero temporal error. Temporal metrics average only valid transitions; training temporal terms average valid frames within each nonempty window and then nonempty windows. Spatial and RAFT terms retain 15-frame and four-window averaging.

Standard CUDA grid_sample backward may use nondeterministic atomic accumulation; own-repeat relative gradient error is audited under1e-3. No bitwise gradient reproducibility claim.

|Group|Committed updates|Videos/groups/windows consumed|Stop reason|
|---|---:|---|---|
|A|300|1200/1200/1200|MAX_300_UPDATES_REACHED|
|B|300|1200/1200/1200|MAX_300_UPDATES_REACHED|
|C|300|1200/1200/1200|MAX_300_UPDATES_REACHED|
|D|300|1200/1200/1200|MAX_300_UPDATES_REACHED|

## Validation records

|Group|Step|Median per-video Sobel ratio|Videos ratio<1|Numeric screen|
|---|---:|---:|---:|---|
|Original|0|0.9826902006274723|8|NOT_PASSED|
|A2000|0|1.0|0|NOT_PASSED|
|A|0|1.0|0|NOT_PASSED|
|A|100|0.9525236445449621|12|NOT_PASSED|
|A|200|0.9377426267987792|12|ONE_VALIDATION_PASS|
|A|300|0.9345717197017398|12|NUMERIC_SCREEN_PASS|
|B|0|1.0|0|NOT_PASSED|
|B|100|0.9485558169132027|12|ONE_VALIDATION_PASS|
|B|200|0.9353983817368678|12|NUMERIC_SCREEN_PASS|
|B|300|0.9346453635591552|12|NUMERIC_SCREEN_PASS|
|C|0|1.0027490866313036|0|NOT_PASSED|
|C|100|0.960576714341526|12|NOT_PASSED|
|C|200|0.9435730691334381|12|ONE_VALIDATION_PASS|
|C|300|0.9423237561591414|12|NUMERIC_SCREEN_PASS|
|D|0|1.0027490866313036|0|NOT_PASSED|
|D|100|0.9558963238312959|12|NOT_PASSED|
|D|200|0.9424272416007781|12|ONE_VALIDATION_PASS|
|D|300|0.9396537635924054|12|NUMERIC_SCREEN_PASS|

Full per-video/QP/frame/transition values: development_metrics.csv and evaluation/*/step_*/. Raw and weighted training terms plus component gradient norms: training_history.csv and compressed receipts.

The numeric screen is a pre-registered pilot threshold; it is not a significance test or visual acceptance record. No automatic expansion or deployment.

Unavailable/failed stages remain incomplete in status.json and final_integrity.json. Videos and checkpoints are local only; manifests contain SHA256.

## Final execution status

Local pilot complete: all four branches committed300 updates, all scheduled36-point validations and four128-frame synchronized comparisons complete. Final integrity PASS.

Postprocessing recovery: the original final layout used an FFmpeg option absent from the installed version. Its first recovery incorrectly re-entered an engineering stage after the controller had intentionally released training caches. Both failures are preserved. This entry verified existing successful engineering checks, completed checkpoints/receipts and unchanged source hashes, then performed CPU-only composition and the unchanged final integrity checks. No training updates or metric evaluations were repeated. See audits/post_training_recovery/.

Publication is recorded separately in publication_status.json and publication_remote_receipt.json.
