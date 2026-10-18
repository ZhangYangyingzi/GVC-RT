#!/usr/bin/env bash
set -euo pipefail
cd /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_34a_p_joint_fd_trend_screen
mkdir -p logs
setsid nohup /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u driver.py > logs/driver.log 2>&1 < /dev/null &
echo $! > launcher.pid
echo "driver_pid=$(cat launcher.pid)"
