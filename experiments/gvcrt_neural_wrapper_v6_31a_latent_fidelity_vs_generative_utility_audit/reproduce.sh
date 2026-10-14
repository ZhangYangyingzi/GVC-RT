#!/usr/bin/env bash
set -euo pipefail
audit_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
audit_python=/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd -- "$audit_dir/../.."
if [[ "${1:-}" == '--resume' ]]; then
  exec "$audit_python" -B -u "$audit_dir/launch.py"
fi
"$audit_python" -B -u "$audit_dir/preflight.py"
"$audit_python" -B -u "$audit_dir/prepare_proxy.py"
/data1/anaconda3_new/anaconda_program/bin/python -B -u "$audit_dir/selftest.py"
"$audit_python" -B -u "$audit_dir/worker.py" --gpu 4 --sanity
exec "$audit_python" -B -u "$audit_dir/launch.py"
