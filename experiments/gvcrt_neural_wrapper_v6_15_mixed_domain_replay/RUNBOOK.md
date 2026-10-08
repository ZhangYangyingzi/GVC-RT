# V6.15 mixed-domain replay

Run commands from `/Huang_group/zyyz/Projects/GVC-RT` with access to GPU 4–7. The controller selects one training GPU and separate evaluation GPUs using available memory, without terminating other tasks.

## Prepare (once, before initialization)

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_15_mixed_domain_replay/prepare.py
```

## Start or resume

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_15_mixed_domain_replay/resume_pipeline.py
```

The controller adopts surviving worker PIDs, skips completed verified points, and resumes `checkpoints/latest.pt` with model, AdamW, independent domain RNG, sampling RNG, Python RNG, CPU and CUDA RNG states. Logs beyond the latest persisted optimizer update are archived and replayed. Checkpoint snapshots are saved at 0, 250, 500, 1000, 2000, 3000 and 5000 additional updates; latest recovery state is saved every 50 updates. Reported models are fixed at mixed_1000 and mixed_5000.

## Status

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_15_mixed_domain_replay/run_pipeline.py --status
tail -n 20 experiments/gvcrt_neural_wrapper_v6_15_mixed_domain_replay/logs/pipeline.log
cat experiments/gvcrt_neural_wrapper_v6_15_mixed_domain_replay/training_status.json
cat experiments/gvcrt_neural_wrapper_v6_15_mixed_domain_replay/final_integrity.json
```

## Protocol

Training starts from the verified v6.2 schedule_s1p0 step_1000 model with a fresh AdamW. The v6.14 objective AST, parameter scope, quantization and DPB are unchanged. Each update selects a domain with an independent seeded 50/50 Bernoulli draw, then a uniform video from that domain. U-Long uses the historical sampler; UVG uses the original four-video 64-frame pool. The exact 5000 external QPs are replayed from the historical v6.14 log. Every update records source indices, crop, losses, estimated bpp, gradients and learning rates.

Validation uses all original 32 normal and 32 hard U-Long videos at their original first 32 frames and QPs 0,2,4,6,8,9, and Bosphorus at 64 frames and QPs 0–9. Final evaluation uses 8 U-Long, 2 UVG holdout and 5 HEVC-B videos at their original 64-frame indices and QPs 0–9. Native HEVC-B fps is preserved. Real RANS bytes and independent decode are required. Quality metrics are aggregated per video; FID uses separate fixed pooled features for each final dataset. PCHIP comparisons use 100 uniform ln(bpp) samples in the six-model common interval without extrapolation. No checkpoint selection is performed.

## Publication

Only experiment code, manifests, configuration, text logs, CSV/JSON numeric outputs, plots and hash/integrity records are staged in an independent worktree based on the latest origin/main. Checkpoints/optimizer states, datasets, videos, frame images, bitstreams and feature caches stay on the server. Publication records are stored in `publication_status.json`. The original worktree is not switched or cleaned.

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_15_mixed_domain_replay/check_completion.py
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_15_mixed_domain_replay/publish_results.py
cat experiments/gvcrt_neural_wrapper_v6_15_mixed_domain_replay/publication_status.json
```
