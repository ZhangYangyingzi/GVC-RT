# v6.32b native-frame completion

All writes are confined to this directory. Historical v6.32 remains read-only.

## Launch or resume

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion/launch32b.py
```

The driver holds an exclusive flock; an active driver is never replaced by deleting its lock. On resume it verifies worker PID start ticks and command identity and retains their memory reservations. Workers use physical GPU4–7 via CUDA_VISIBLE_DEVICES; each sees cuda:0. Numerical failures retain the task cursor and checkpoints. Failed tasks require inspection and a recorded repair before requeueing; OOM retries increase the reservation.

## View state without importing torch or scanning large directories

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion/check_once.py
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion/wait_status.py
```

`pipeline_status.json` separates computation, report, and publication. Final PASS requires all three.

## Recovery semantics

C/D load their own atomic latest checkpoint: model, AdamW, crop/global RNG, buffers, cursor and twenty EMA states. They never reload B1 or recalibrate when resuming a committed branch. Each successful update checkpoints atomically before the receipt is published; the checkpoint also contains its last receipt for recovery after a crash. Physical optimizer calls not represented by a committed checkpoint are recorded as discarded/replayed calls. Latest is saved every update, which is stronger than the requested 25-step interval. Fixed snapshots are 0/1/250/500/1000. Inference exports exclude optimizer and EMA.

A/B weights, optimizer history and test points are imported only after hash/protocol verification. Their four missing auxiliary full-frame diagnostics per branch and zero EMA commits remain historical limitations; they are not rerun or relabelled as v6.32b execution.

## Fixed protocol

The shared and calibration plans are byte copies. C1 crop training remains unchanged. Only auxiliary full views use native valid RGB size, `CompressionModel.get_padding_size(H,W,16)`, replicate padding and exact unpadding before official Inception. Formal evaluation remains the original 1080p/64-frame, true RANS independent-decode protocol.

Training statistics use FP64/ddof0 and fixed training-only GT regularization. Formal pooled FID uses FP64/ddof1 with no training regularization. MCL29 is a derived subset and adds no encoded tasks. See protocol.json and source/hash audits.

## Logs and artifacts

`logs/driver.log`, per-task logs, `errors.jsonl`, `gpu_schedule.jsonl`, branch optimizer journals and training receipts preserve execution and failures. Large checkpoints, bitstreams and NPZ files remain on the server; publication includes their path/size/SHA256 index. Large textual results are split on complete lines and verified with `reassemble.py`.

## Final gradient-recording audit

The immutable training code assigns the combined tensors to `p.grad`, then applies
in-place global clipping. Its legacy `gradient_formula_error` field is therefore
a post-clip comparison against unscaled base/additional gradients. This raw field
is preserved and is not a pre-clip merge error. The final audit uses independently
checked composition/clip identities for every receipt plus real-network, read-only
first-step vector replays; see `audits/gradient_recording_integrity.json`.
No optimizer or EMA update is performed by these probes.

```bash
PY=/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python
EXP=/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$PY" -B "$EXP/gradient_recording_audit32b.py" --branch C_C1_lr2_FD --gpu 4
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$PY" -B "$EXP/gradient_recording_audit32b.py" --branch D_C1_lr2_mean_CORAL --gpu 4
"$PY" -B "$EXP/gradient_recording_audit32b.py" --aggregate
```

Choose an authorized GPU with the recorded full-task memory reserve available.
The probes require saved step0 checkpoints and preserve all existing training
receipts, model states, optimizer states, EMA counters and evaluation results.
