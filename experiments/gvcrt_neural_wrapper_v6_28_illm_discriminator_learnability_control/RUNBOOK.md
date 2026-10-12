# V6.28 execution

Launch/resume: `/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_28_illm_discriminator_learnability_control/launch_pipeline.py`

Status: `cat /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_28_illm_discriminator_learnability_control/pipeline_status.json`

Training: fixed B1; D-only TINY2000 / MAIN5000, all four branches. No new G updates or RD tests.

Completed caches/checkpoints are hash-verified; owned live scheduler is preserved.

CPU figures use the existing `/data1/anaconda3_new/anaconda_program/bin/python` plotting environment because matplotlib is absent from gvc-rt. No package or Torch/CUDA environment was changed. Model/cache/training/validation and numeric summaries use the requested gvc-rt Python.
