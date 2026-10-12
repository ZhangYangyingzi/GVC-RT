# V6.27 run and resume

No G/D training or optimizer updates. All historical experiment files are read only.
Do not rerun setup.py over an existing run. The source hashes, fixed plan and30-video manifest are already frozen.

Start/resume without duplicating a live scheduler:
```bash
cd /Huang_group/zyyz/Projects/GVC-RT
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_27_illm_effectiveness_mcljcv_audit/launch_pipeline.py
```

Status:
```bash
cat /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_27_illm_effectiveness_mcljcv_audit/pipeline_status.json
tail -n30 /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_27_illm_effectiveness_mcljcv_audit/logs/run_pipeline.log
tail -n20 /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_27_illm_effectiveness_mcljcv_audit/logs/diagnostic_postprocess.log
tail -n20 /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_27_illm_effectiveness_mcljcv_audit/logs/pooled_worker.log
```

The detached session survives SSH disconnection. Scheduler flock and PID start-time checks preserve active-worker reservations. No lock deletion or other-process termination is used. Each whole64-frame point resumes only after artifact/hash validation.

CPU precomputation is optional for throughput; the scheduler's final summarize.py reruns/reuses all completed statistics and pooled points before plots, independent visual decode, final integrity and ordinary publication.

Individual numerical stages, after all required raw points exist:
```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_27_illm_effectiveness_mcljcv_audit/summarize.py
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_27_illm_effectiveness_mcljcv_audit/plots.py
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_27_illm_effectiveness_mcljcv_audit/visuals.py
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u /Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_27_illm_effectiveness_mcljcv_audit/finalize.py
```
Do not run concurrent publishers. The persistent pipeline calls publish_github.py and verify_final.py automatically.

Reassemble text shards using restore_text_shards.py and file_index.json; large model/feature/stream/RGB artifacts stay local and are SHA-indexed.
