# V5-A.1 evaluation-only execution

This directory is independent of V4.1 and V5-A. No training entry point exists.
Checkpoints are resolved from completed V5-A final point metadata and hashed.
No checkpoint, architecture, loss, beta, or DPB gradient changes are made.

## Frozen evaluation protocol

- External QPs: 0–9. Actual P QP is read from each serialized frame header.
- QP0–3: all 128 prior U-Long points reused after exact artifact/hash, scalar
  metric, QP semantics and recomputed pooled-FID compatibility checks.
- U-Long: unchanged 8-video manifest and fixed 64-frame RGB sources.
- UVG: 7 locally discovered raw sequences, first 96 frames each, 672 frames.
- UVG RGB conversion matches the prior corrected UVG experiment. The three
  previously tested sequences match existing first 64 RGB frames exactly.
- Metrics use 1920×1080 RGB. Codec input pads by replication to 1920×1088.
- FID pools all dataset frames per method and external QP: 512 / 672 samples.
- FloLPIPS averages within-sequence transitions, never across sequences.
- QP scopes: 0–3 seen, 4–9 unseen, full 0–9. No point removal or monotonicization.

## Background queue

run_pipeline.py uses physical GPUs 4 and 6 only. GPUs 5 and 7 were occupied
by unrelated processes at launch and are not used. GPUs 0–3 are never assigned.
It schedules all four U-Long methods, builds U-Long reports and plots, then all
four UVG methods, builds UVG reports and plots, and runs the final integrity audit.
Completed point JSONs are atomic resume markers. A process lock prevents a
duplicate scheduler. A worker/audit failure stops the queue and records FAIL;
only that queue's own children may be terminated by the failure handler.

Inspect progress from the repository root:

    cat experiments/gvcrt_neural_wrapper_v5a_1_allqp_uvg_generalization/pipeline_status.json
    tail -n 25 experiments/gvcrt_neural_wrapper_v5a_1_allqp_uvg_generalization/logs/pipeline.log

If the queue has stopped, inspect its error and resolve it before resuming:

    cd /Huang_group/zyyz/Projects/GVC-RT
    setsid -f /data1/anaconda3_new/anaconda_program/bin/python -B -u experiments/gvcrt_neural_wrapper_v5a_1_allqp_uvg_generalization/run_pipeline.py >> experiments/gvcrt_neural_wrapper_v5a_1_allqp_uvg_generalization/logs/pipeline.log 2>&1 < /dev/null

Do not rerun prepare.py: input manifests and hashes are already frozen.
Do not start training to repair an evaluation failure.

Final requested numeric output is generated only after final_integrity PASS:

    cat experiments/gvcrt_neural_wrapper_v5a_1_allqp_uvg_generalization/logs/final_stdout.log
