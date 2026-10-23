#!/usr/bin/env bash
set -euo pipefail
branch="$1"
gpu="$2"
root="$(cd "$(dirname "$0")" && pwd)"
python_bin=/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python
while [[ ! -f "$root/server_artifacts/$branch/step_300.pt" ]]; do sleep 10; done
for step in 0 100 300; do
  for source in UVal0 UVal1; do
    for qp in 2 7; do
      "$python_bin" "$root/evaluate.py" --branch "$branch" --step "$step" --source "$source" --qp "$qp" --gpu "$gpu"
    done
  done
done
for source in ParkScene Jockey; do
  for qp in 2 7; do
    "$python_bin" "$root/evaluate.py" --branch "$branch" --step 300 --source "$source" --qp "$qp" --gpu "$gpu"
  done
done
