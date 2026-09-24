#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/Huang_group/zyyz/Projects/GVC-RT/expericent_generation_input/expericent_generator_aware_latent_distortion_v11/b2_latent_direction_magnitude_v11_7/src:/Huang_group/zyyz/Projects/GVC-RT/expericent_generation_input/expericent_interface_causal_controls_v9/src:/Huang_group/zyyz/Projects/GVC-RT/expericent_generation_input/expericent_generator_aware_encoder_latent_rd_oracle_v12_debug/src
PY=/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python
$PY run_v12_1.py --gpu 5 --cells 'fresh_ulong:10:1,fresh_ulong:10:3,uvg:2:1,uvg:2:3' > logs/gpu5.log 2>&1 &
$PY run_v12_1.py --gpu 6 --cells 'fresh_ulong:0:1,fresh_ulong:0:3,fresh_ulong:1:1,fresh_ulong:1:3' > logs/gpu6.log 2>&1 &
$PY run_v12_1.py --gpu 7 --cells 'uvg:0:1,uvg:0:3,uvg:1:1,uvg:1:3' > logs/gpu7.log 2>&1 &
wait
$PY finalize.py | tee logs/finalize.log
