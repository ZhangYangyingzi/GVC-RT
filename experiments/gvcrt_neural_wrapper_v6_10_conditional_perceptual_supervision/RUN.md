V6.10 conditional perceptual supervision

Launch or resume:

    cd /Huang_group/zyyz/Projects/GVC-RT
    PYTHONDONTWRITEBYTECODE=1 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_10_conditional_perceptual_supervision/run_pipeline.py --launch

Progress: pipeline_status.json; branches/*/training_status.json.
Logs: logs/pipeline.log; logs/train_*.log; training_logs/*.jsonl.
Controller and workers hold process locks. Latest full-state snapshots are saved every
50 generator updates; named checkpoints are 1250, 1500, 2000.
Only 1500 and 2000 receive formal evaluation.

The training/codec/FID environment is gvc-rt. PNG rendering alone uses the already
installed matplotlib in /data1/anaconda3_new/anaconda_program/bin/python.

The user corrected evaluation to use the frozen V6.9 pools: UVG/U-Long 64 frames,
VIRAT 96 or 33 frames per source video and 20 fps rate accounting. The activation
script verifies the source pixels, caches, bootstrap indices and reused points
before writing audits/evaluation_protocol_resolution.json. The live scheduler
then automatically admits evaluations without restarting training.
