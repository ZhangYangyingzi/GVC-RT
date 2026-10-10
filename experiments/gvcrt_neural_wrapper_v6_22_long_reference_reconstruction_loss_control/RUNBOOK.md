# Run and resume

```bash
python experiments/gvcrt_neural_wrapper_v6_22_long_reference_reconstruction_loss_control/restore_text_shards.py
```

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_22_long_reference_reconstruction_loss_control/run_pipeline.py
```

# Status

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_22_long_reference_reconstruction_loss_control/run_pipeline.py --status
tail -20 experiments/gvcrt_neural_wrapper_v6_22_long_reference_reconstruction_loss_control/logs/train_B.log
```

# Publication retry

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_22_long_reference_reconstruction_loss_control/publish_github.py
```

# Fixed protocol

Training plan seed: 20261022. Calibration seed: 20261023. Both branches use native I at 0, no_grad current-model P frames 1–48, and detached supervised P frames 49–63. Calibration fixes lambda_rec once using 40 training samples. Reconstruction gradients use independent scale 4096 on the unweighted term, FP32 restoration, then FP32 lambda_rec multiplication before adding base gradients and global clipping.
