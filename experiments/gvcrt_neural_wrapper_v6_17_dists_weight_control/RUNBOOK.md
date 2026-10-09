# Run and resume

Run from `/Huang_group/zyyz/Projects/GVC-RT`:

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_17_dists_weight_control/run_pipeline.py
```

The same command resumes verified points and the latest optimizer/RNG/replay checkpoint. Only GPU4–7 are used, with at most one experiment worker per GPU. Other tasks are never terminated. OOM workers exit and are requeued from their latest complete state.

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_17_dists_weight_control/run_pipeline.py --status
tail -n 10 experiments/gvcrt_neural_wrapper_v6_17_dists_weight_control/logs/train.log
tail -n 5 experiments/gvcrt_neural_wrapper_v6_17_dists_weight_control/training_logs/dists_w05.jsonl
```

Preparation (new directory only), baseline verification, and publication retry:

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_17_dists_weight_control/prepare.py
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_17_dists_weight_control/reuse.py
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_17_dists_weight_control/publish_github.py
```
