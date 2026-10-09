# Run and resume

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_18_long_sequence_data_coverage/run_pipeline.py
```

The same command resumes preparation, verified points, branch model/optimizer/
RNG states and replay cursors, aggregation and publication. Only the new
experiment is written. The original 1024 ULong pool is cleaned to 1020 here;
the prior blocked metadata is in `archived_before_cleanup/`.

```bash
tail -n 3 experiments/gvcrt_neural_wrapper_v6_18_long_sequence_data_coverage/logs/prepare.log
tail -n 3 experiments/gvcrt_neural_wrapper_v6_18_long_sequence_data_coverage/logs/train_A.log
tail -n 3 experiments/gvcrt_neural_wrapper_v6_18_long_sequence_data_coverage/logs/train_B.log
tail -n 3 experiments/gvcrt_neural_wrapper_v6_18_long_sequence_data_coverage/logs/train_C.log
```

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_18_long_sequence_data_coverage/run_pipeline.py --status
```
