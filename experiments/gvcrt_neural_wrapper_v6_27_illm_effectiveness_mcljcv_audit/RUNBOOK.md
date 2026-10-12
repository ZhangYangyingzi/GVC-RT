# V6.27

No optimizer updates. Historical experiments are read only.

Start or resume:
```bash
cd /Huang_group/zyyz/Projects/GVC-RT
nohup /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_27_illm_effectiveness_mcljcv_audit/run_pipeline.py > /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_27_illm_effectiveness_mcljcv_audit/logs/pipeline.log 2>&1 < /dev/null &
```
Status: read `pipeline_status.json`; tail `logs/pipeline.log`. The scheduler uses flock and validates PID start times.
