# V6.2-B final generalization test

Evaluation only. The candidate is fixed to schedule_s1p0 step1000 with the SHA256 specified in the request. No training or final-test-based selection is implemented.

All 930 points are fresh runs: Original, V4.1, V6.2; 8 U-Long, 7 UVG, 8 VIRAT720, 8 VIRAT480 sequences; QP0–9. Existing raw baseline results are not imported.

U-Long and UVG use the unchanged V5-A.2 canonical RGB PNG loader and 30 fps rate accounting. VIRAT uses the existing frozen PNGs, indices and spatial conversion; both resolutions use 20 fps only in bitrate accounting. There is no temporal resampling or input re-encoding.

`preflight_audit.json` gates evaluation. Only `final_integrity.json: PASS` means all evaluation, report, source and old-artifact integrity checks completed. Partial tables must not be mistaken for final results.

From repository root:

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_2b_final_generalization_test/run_pipeline.py --launch
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_2b_final_generalization_test/run_pipeline.py --status
tail -f experiments/gvcrt_neural_wrapper_v6_2b_final_generalization_test/logs/pipeline.log
```

The detached queue allocates one process per available GPU among 4,5,6,7. GPUs with 128 MiB or more occupied memory are not allocated, and unrelated processes are never terminated.

Raw points and per-frame CSVs: `parts/<dataset>/<method>/`.
Complete bitstreams: `bitstreams/`; metric features and transitions: `features/`.
Dataset-specific numerical reports: `results/{ulong,uvg,virat720,virat480}/`.
Reconstructions use the existing decoder save helper after metric computation, under `visualizations/`; their inputs and metrics are unchanged.

Summary bitrate is the arithmetic mean of per-sequence bitrates, as in the existing protocol. Total bytes/bits are summed. FloLPIPS is transition-weighted, and dataset FID pools all frame features. Per-sequence FID is also retained. Nonmonotone or non-overlapping BD-rate comparisons are marked invalid with reasons, never repaired by sorting metric values.
