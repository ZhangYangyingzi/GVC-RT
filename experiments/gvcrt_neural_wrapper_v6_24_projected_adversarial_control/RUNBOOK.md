# V6.24 commands

Run from `/Huang_group/zyyz/Projects/GVC-RT` using the gvc-rt Python environment.

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_24_projected_adversarial_control/prepare.py
nohup /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_24_projected_adversarial_control/run_pipeline.py > experiments/gvcrt_neural_wrapper_v6_24_projected_adversarial_control/logs/pipeline.log 2>&1 < /dev/null &
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_24_projected_adversarial_control/run_pipeline.py --status
```

The scheduler holds an exclusive lock. Reuse an existing matching scheduler. Restart the same pipeline command after an exited scheduler to restore optimizer, RNG and plan cursor and skip verified completed points. Inspect `pipeline_status.json`, `gpu_schedule.jsonl`, branch status files and `logs/`.

Training seed 20261024, calibration seed 20261025, projected initialization seed 20261026. Forty training-only windows calibrate lambda_adv once. Complete B1 loss retains lambda_rec=0.1383589248954108. Reconstruction and adversarial gradients are scaled independently by 4096, restored and weighted in FP32 before addition to unscaled base gradients, followed by one global gradient clip and optimizer update. Native I at 0, causal no-grad P1–48, detached references, supervised P49–63 divided by 15. D uses independent RNG, frozen feature/projector in eval; G freezes D parameters and buffers. No differential augmentation.

The four methods each use 16 sequences and external QP0–9, IP=-1, 64 frames. Real bytes include headers and native I. Retain native fps kbps and report kbps_20fps separately. Equal-rate tables use four-model common actual bpp intervals, 100 ln(bpp) points, PCHIP without extrapolation. Dataset FID uses fixed pooled features.

Publication uses an isolated worktree based on latest origin/main, explicit experiment paths, ordinary commits and pushes; server-only checkpoints, bitstreams, features, dependencies and datasets are excluded. `file_index.json` lists SHA256 and text shards. Run `restore_text_shards.py` in a checkout to reconstruct byte-identical oversized logs.

## Projected dependency and server-only weight

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -m pip install --no-deps --target experiments/gvcrt_neural_wrapper_v6_24_projected_adversarial_control/vendor/dependencies timm==0.4.12
```

The pretrained file is `weights/tf_efficientnet_lite1-bde8b488.pth`; source URL, exact SHA256 and strict-load verification are in `audits/projected_source.json`. Upstream commit is pinned there; required `pg_modules` source and its license are included. Feature/projector state hashes include buffers. The D spectral-normalization vectors are initialized with 50 power iterations before calibration, without parameter updates. The fixed hinge formulas are `G=-mean_scales(mean_patches(D(fake)))` and `D=0.5*(mean_scales(mean_patches(relu(1-D(GT))))+mean_scales(mean_patches(relu(1+D(fake.detach())))))`, with a mean over 15 target frames.
