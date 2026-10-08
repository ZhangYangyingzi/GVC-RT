# Run and resume

From `/Huang_group/zyyz/Projects/GVC-RT`, use the existing environment:

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_16_dists_gradient_verification/run_pipeline.py
```

The same command resumes checked points and the latest complete optimizer/RNG checkpoint. Following the user's subsequent GPU authorization, it may share GPU 4–7 when memory is sufficient, with at most one experiment worker per GPU, and never terminates other tasks. The authorization is recorded in `audits/gpu_policy_amendment.json`.

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_16_dists_gradient_verification/run_pipeline.py --status
tail -n 20 experiments/gvcrt_neural_wrapper_v6_16_dists_gradient_verification/logs/train.log
tail -n 5 experiments/gvcrt_neural_wrapper_v6_16_dists_gradient_verification/training_logs/dists_fixed.jsonl
```

CPU input replay audit and standalone independent probe:

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_16_dists_gradient_verification/replay_audit.py
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_16_dists_gradient_verification/probes.py --cpu
```

After a final terminal integrity record, retry only publication:

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_16_dists_gradient_verification/publish_github.py
```
