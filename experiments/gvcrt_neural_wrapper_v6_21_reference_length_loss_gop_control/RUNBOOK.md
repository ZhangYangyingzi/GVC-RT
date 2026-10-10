# Run and resume

```bash
python experiments/gvcrt_neural_wrapper_v6_21_reference_length_loss_gop_control/restore_text_shards.py
```

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_21_reference_length_loss_gop_control/run_pipeline.py
```

# Status

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_21_reference_length_loss_gop_control/run_pipeline.py --status
tail -20 experiments/gvcrt_neural_wrapper_v6_21_reference_length_loss_gop_control/logs/train_B.log
```

# Publication retry

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_21_reference_length_loss_gop_control/publish_github.py
```
