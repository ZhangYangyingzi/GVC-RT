# v6.19 commands

Start or resume (model is fixed; no training):

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_19_i_frame_wrapper_bypass/run_pipeline.py
```

Status:

```bash
cat experiments/gvcrt_neural_wrapper_v6_19_i_frame_wrapper_bypass/pipeline_status.json
cat experiments/gvcrt_neural_wrapper_v6_19_i_frame_wrapper_bypass/final_integrity.json
cat experiments/gvcrt_neural_wrapper_v6_19_i_frame_wrapper_bypass/publication_status.json
```

The pilot gate compares exact standard bitstreams, reconstruction hashes and metrics with v6.18. I-bypass payload and decoded initial DPB are compared with original. Only GPU 4/5/6/7 are scheduled, based on memory. Completed verified points are skipped. All 64-frame streams include SPS and frame headers. The official matrix has 480 points. First-frame diagnostics and rate interpolation use the full sequence bitrate.

Weights, full bitstreams, decoded images and feature caches remain on the server. Publication uses a fresh origin/main worktree, stages only this experiment and performs an ordinary HEAD:main push.
