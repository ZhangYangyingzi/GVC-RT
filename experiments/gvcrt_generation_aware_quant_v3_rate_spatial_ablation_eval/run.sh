#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$root/../.." && pwd)"
py=/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python
cd "$repo"

wait_gpu() {
  while true; do
    used="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F, '$1 == 6 {gsub(/ /,"",$2); print $2}')"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used < 1024 )); then return; fi
    sleep 20
  done
}

if [[ ! -f "$root/preflight.json" ]]; then
  for mode in A MAP IDENTITY; do
    wait_gpu
    "$py" "$root/task.py" --mode "$mode" --source UVal0 --qp 2 --gpu 6
  done
  for mode in MEAN ROLL; do
    wait_gpu
    "$py" "$root/task.py" --mode "$mode" --source UVal1 --qp 2 --alpha-index 0 --gpu 6 --rate-only
  done
  wait_gpu
  "$py" "$root/reference_identity_audit.py"
  "$py" "$root/preflight.py"
fi

"$py" "$root/alpha_sync_audit.py" --ensure
"$py" "$root/driver.py"
"$py" "$root/finalize.py"
"$py" "$root/map_driver.py"
"$py" "$root/audit_delivery.py"
"$py" "$root/make_previews.py"
"$py" "$root/make_report.py"
