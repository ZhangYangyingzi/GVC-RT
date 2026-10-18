# v6.33 fixed paired continuation

Training/evaluation counters and numeric tables; no model selection or research judgement.

Additional/cumulative updates: {'B_C1_lr2': {'additional': 1000, 'cumulative': 2000}, 'C_C1_lr2_FD': {'additional': 1000, 'cumulative': 2000}, 'D_C1_lr2_mean_CORAL': {'additional': 1000, 'cumulative': 2000}}.
Physical/replay accounting: {'B_C1_lr2': {'physical_optimizer_calls': 1000, 'committed_additional_updates': 1000, 'cumulative_adaptation_updates': 2000, 'discarded_or_replayed_calls': 0, 'parent_training_checkpoint': '/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_32_learning_rate_fd_coral_control/branches/B_C1_lr2/checkpoints/adaptation_step_1000.pt'}, 'C_C1_lr2_FD': {'physical_optimizer_calls': 1000, 'committed_additional_updates': 1000, 'cumulative_adaptation_updates': 2000, 'discarded_or_replayed_calls': 0, 'parent_training_checkpoint': '/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion/branches/C_C1_lr2_FD/checkpoints/adaptation_step_1000.pt'}, 'D_C1_lr2_mean_CORAL': {'physical_optimizer_calls': 1000, 'committed_additional_updates': 1000, 'cumulative_adaptation_updates': 2000, 'discarded_or_replayed_calls': 0, 'parent_training_checkpoint': '/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion/branches/D_C1_lr2_mean_CORAL/checkpoints/adaptation_step_1000.pt'}}.
Historical read-only points2300; historical fallback encodes0; planned new encodes1794; logical4094/4094.

Historical GT reference1000/1000 and calibration40/40 are verified read-only aliases; additional prefill0 and calibration0. Exact source checkpoints,full optimizer moments/steps,RNG,buffers and C/D EMA states were migrated. The same frozen1000 windows are exposed a second time; zero new source videos/windows. B EMA is N/A. C/D each add1000 EMA commits and finish2000 with100/stratum. Missing historical B auxiliary views are neither retrofilled nor relabelled.

## Fixed protocol

See REPORT.md and equal_rate_summary.csv for the prescribed numeric comparisons.
