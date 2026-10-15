#!/usr/bin/env bash
set -euo pipefail
cd /Huang_group/zyyz/Projects/GVC-RT
export PYTHONDONTWRITEBYTECODE=1
exec /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_33_prebottleneck_coding_mechanism_audit/launcher.py
