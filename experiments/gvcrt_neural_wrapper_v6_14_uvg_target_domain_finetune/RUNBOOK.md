# 启动与恢复

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_14_uvg_target_domain_finetune/run_pipeline.py --launch
```

# 状态检查

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_14_uvg_target_domain_finetune/run_pipeline.py --status
tail -n 20 experiments/gvcrt_neural_wrapper_v6_14_uvg_target_domain_finetune/logs/pipeline.log
```
