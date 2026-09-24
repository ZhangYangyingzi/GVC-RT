#!/usr/bin/env bash
set -euo pipefail
REPO=/Huang_group/zyyz/Projects/GVC-RT
ROOT="$REPO/experiments/gvcrt_neural_wrapper_v2_retest_fixed_uvg"
PY=/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python
cd "$REPO"
mkdir -p "$ROOT/logs" "$ROOT/parts"

"$PY" -B "$ROOT/prepare_sources.py" >"$ROOT/logs/prepare_sources.log" 2>"$ROOT/logs/prepare_sources.err"

"$PY" -B "$ROOT/evaluate.py" --gpu 4 --tags uvg_00,uvg_01,uvg_02 \
  >"$ROOT/logs/eval_gpu4.log" 2>"$ROOT/logs/eval_gpu4.err" & p4=$!
"$PY" -B "$ROOT/evaluate.py" --gpu 5 --tags fresh_ulong_00,fresh_ulong_01,fresh_ulong_10 \
  >"$ROOT/logs/eval_gpu5.log" 2>"$ROOT/logs/eval_gpu5.err" & p5=$!
"$PY" -B "$ROOT/evaluate.py" --gpu 6 --tags fresh_ulong_100,fresh_ulong_101,fresh_ulong_102 \
  >"$ROOT/logs/eval_gpu6.log" 2>"$ROOT/logs/eval_gpu6.err" & p6=$!
"$PY" -B "$ROOT/evaluate.py" --gpu 7 --tags fresh_ulong_103,fresh_ulong_104 \
  >"$ROOT/logs/eval_gpu7.log" 2>"$ROOT/logs/eval_gpu7.err" & p7=$!
wait "$p4"; wait "$p5"; wait "$p6"; wait "$p7"

"$PY" -B "$ROOT/finalize.py" >"$ROOT/logs/finalize.log" 2>"$ROOT/logs/finalize.err"
cat "$ROOT/stdout.log"
