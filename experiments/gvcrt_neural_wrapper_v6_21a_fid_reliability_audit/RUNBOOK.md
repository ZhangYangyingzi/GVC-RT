# Commands

From /Huang_group/zyyz/Projects/GVC-RT:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_21a_fid_reliability_audit/prepare.py
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_21a_fid_reliability_audit/statistics.py --test
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_21a_fid_reliability_audit/run_pipeline.py
cat experiments/gvcrt_neural_wrapper_v6_21a_fid_reliability_audit/pipeline_status.json
```

Verified points and completed statistics resume automatically. Do not start a second controller while the first PID is alive. GPU4–7 only. Up to two codec plus one statistical worker per device when observed free memory permits. No training.
