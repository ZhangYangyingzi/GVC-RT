#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
pids=()
for i in 0 1 2 3; do /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -u gvc_lowrate_all_videos/explore.py --gpu "$((i+4))" --shard "$i" > "gvc_lowrate_all_videos/logs/gpu$((i+4)).log" 2>&1 & pids+=("$!"); done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=1; done
if [ "$failed" -ne 0 ]; then echo WORKER_FAILED > gvc_lowrate_all_videos/FAILED; exit 1; fi
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -u gvc_lowrate_all_videos/report.py > gvc_lowrate_all_videos/logs/report.log 2>&1 || { echo REPORT_FAILED > gvc_lowrate_all_videos/FAILED; exit 1; }
