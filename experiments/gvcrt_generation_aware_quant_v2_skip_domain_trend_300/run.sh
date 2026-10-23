#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
cd "$root/../.."
exec /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python "$root/run.py"
