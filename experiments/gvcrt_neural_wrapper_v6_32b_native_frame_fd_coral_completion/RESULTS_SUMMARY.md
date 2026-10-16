# v6.32b execution and numerical record

Computation, training, real evaluation and numerical audits: PASS. Publication is tracked independently in publication_status.json and final_integrity.json.

Committed updates: {'A_C1_lr1': 1000, 'B_C1_lr2': 1000, 'C_C1_lr2_FD': 1000, 'D_C1_lr2_mean_CORAL': 1000}. Physical/replay accounting: {'A_C1_lr1': {'physical_optimizer_calls': 1000, 'committed_updates': 1000, 'discarded_or_replayed_calls': 0, 'execution_origin': '/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_32_learning_rate_fd_coral_control/branches/A_C1_lr1'}, 'B_C1_lr2': {'physical_optimizer_calls': 1000, 'committed_updates': 1000, 'discarded_or_replayed_calls': 0, 'execution_origin': '/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_32_learning_rate_fd_coral_control/branches/B_C1_lr2'}, 'C_C1_lr2_FD': {'physical_optimizer_calls': 1000, 'committed_updates': 1000, 'discarded_or_replayed_calls': 0, 'execution_origin': '/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion/branches/C_C1_lr2_FD'}, 'D_C1_lr2_mean_CORAL': {'physical_optimizer_calls': 1000, 'committed_updates': 1000, 'discarded_or_replayed_calls': 0, 'execution_origin': '/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion/branches/D_C1_lr2_mean_CORAL'}}.
Logical evaluation3312/3312; historical read-only reuse2116; newly encoded1196. Shared B1 prefill1000/1000, calibration40/40, twenty complete strata.

## Protocol revision and historical provenance

See PROTOCOL_REVISION.md, protocol_revision.json and effective_protocol.json for explicit scoping of inherited, unused metadata fields; frozen runtime hashes and actual raw source bytes remain preserved.
Only the auxiliary training full-frame view changes to each source native valid H×W. The four plan indices165/253/477/720 retain original source, hashes, frame indices and crop; no resize/substitution. Formal testing remains1920×1080/64frames. Official alignment helper pads each native view and unpads before Inception. Ordinary1080p forward equivalence is recorded in geometry_audit.json.

A/B are historical read-only1000-step C1 runs, not v6.32b reruns. Each retains four missing auxiliary full views and zero diagnostic EMA commits. These known historical auxiliary gaps are not counted as completed diagnostics and do not remove formal test points. C/D each start from the same original v6.22 B1, not B_1000, and receive1000 new AdamW updates.

Fixed training-only coefficients: lambda_FD=0.0044560228041899182; alpha_mu=29.09603986536024; alpha_cov=1413.4647491834271. No test-based coefficient or checkpoint selection.


See equal_rate_summary.csv and REPORT.md for all fixed comparisons.
