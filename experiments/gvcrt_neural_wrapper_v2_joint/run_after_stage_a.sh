#!/usr/bin/env bash
set -euo pipefail

REPO=/Huang_group/zyyz/Projects/GVC-RT
ROOT="$REPO/experiments/gvcrt_neural_wrapper_v2_joint"
PY=/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python
cd "$REPO"

test -f "$ROOT/checkpoints/joint_warmup/step_2000.pt"

"$PY" -B "$ROOT/validate_checkpoints.py" --gpu 4 --label joint_warmup \
  >"$ROOT/logs/validate_joint_warmup.log" 2>"$ROOT/logs/validate_joint_warmup.err"
"$PY" -B "$ROOT/calibrate.py" --gpu 4 \
  >"$ROOT/logs/calibrate.log" 2>"$ROOT/logs/calibrate.err"

"$PY" -B "$ROOT/train.py" --gpu 5 --stage rate --beta-label beta_low \
  >"$ROOT/logs/train_beta_low.log" 2>"$ROOT/logs/train_beta_low.err" &
pid_low=$!
"$PY" -B "$ROOT/train.py" --gpu 6 --stage rate --beta-label beta_mid \
  >"$ROOT/logs/train_beta_mid.log" 2>"$ROOT/logs/train_beta_mid.err" &
pid_mid=$!
"$PY" -B "$ROOT/train.py" --gpu 7 --stage rate --beta-label beta_high \
  >"$ROOT/logs/train_beta_high.log" 2>"$ROOT/logs/train_beta_high.err" &
pid_high=$!
wait "$pid_low"
wait "$pid_mid"
wait "$pid_high"

"$PY" -B "$ROOT/validate_checkpoints.py" --gpu 5 --label beta_low \
  >"$ROOT/logs/validate_beta_low.log" 2>"$ROOT/logs/validate_beta_low.err" &
pid_low=$!
"$PY" -B "$ROOT/validate_checkpoints.py" --gpu 6 --label beta_mid \
  >"$ROOT/logs/validate_beta_mid.log" 2>"$ROOT/logs/validate_beta_mid.err" &
pid_mid=$!
"$PY" -B "$ROOT/validate_checkpoints.py" --gpu 7 --label beta_high \
  >"$ROOT/logs/validate_beta_high.log" 2>"$ROOT/logs/validate_beta_high.err" &
pid_high=$!
wait "$pid_low"
wait "$pid_mid"
wait "$pid_high"

"$PY" -B "$ROOT/evaluate.py" --gpu 4 --tags fresh_ulong_00,fresh_ulong_01 \
  >"$ROOT/logs/eval_gpu4.log" 2>"$ROOT/logs/eval_gpu4.err" &
pid4=$!
"$PY" -B "$ROOT/evaluate.py" --gpu 5 --tags fresh_ulong_10,uvg_00 \
  >"$ROOT/logs/eval_gpu5.log" 2>"$ROOT/logs/eval_gpu5.err" &
pid5=$!
"$PY" -B "$ROOT/evaluate.py" --gpu 6 --tags uvg_01 \
  >"$ROOT/logs/eval_gpu6.log" 2>"$ROOT/logs/eval_gpu6.err" &
pid6=$!
"$PY" -B "$ROOT/evaluate.py" --gpu 7 --tags uvg_02 \
  >"$ROOT/logs/eval_gpu7.log" 2>"$ROOT/logs/eval_gpu7.err" &
pid7=$!
wait "$pid4"
wait "$pid5"
wait "$pid6"
wait "$pid7"

"$PY" -B "$ROOT/finalize.py" >"$ROOT/logs/finalize.log" 2>"$ROOT/logs/finalize.err"
cat "$ROOT/stdout.log"
