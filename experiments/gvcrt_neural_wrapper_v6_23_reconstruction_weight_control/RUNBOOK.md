# V6.23 run and recovery commands

Run from /Huang_group/zyyz/Projects/GVC-RT.

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_23_reconstruction_weight_control/run_pipeline.py --launch
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_23_reconstruction_weight_control/run_pipeline.py --status
tail -n 20 experiments/gvcrt_neural_wrapper_v6_23_reconstruction_weight_control/logs/pipeline.log
```

The same --launch command resumes durable checkpoints, optimizer/RNG state and replay cursor, and validated RD points. The scheduler holds an exclusive lock; do not launch individual workers while it is running. Training restores only this experiment's latest checkpoint. Its initial AdamW is fresh.

Publication after final integrity PASS:
```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_23_reconstruction_weight_control/publish_github.py
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_23_reconstruction_weight_control/restore_text_shards.py
```

Checkpoint locations, SHA256 and inference exports are in branches/B_half/checkpoint_index.json and branches/B_double/checkpoint_index.json. Large binary assets stay on this server; file_index.json describes byte-exact restoration of published text shards.
