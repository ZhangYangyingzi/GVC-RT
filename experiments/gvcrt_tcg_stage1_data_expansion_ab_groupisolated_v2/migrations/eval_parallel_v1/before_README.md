# TCG source-group isolation A/B continuation

Active protocol: `protocol_v3.json`, `PROTOCOL_REVISION_V3.md`, `ACTIVE_PROTOCOL.json`.
Active launcher: `launch_v3.py`; read-only status: `v3_status.py`.
Live execution state: `run_status.json`, `logs/live_status_v3.json`, `REPORT.md` (dated snapshot until final integrity).

This is an in-progress mid-training revision, not a completed result. A retained its committed595-step four-GPU prefix; B resumed the common parent5000 state. The unchanged frozen A/B plans and native_geometry_v1 provenance remain protected. Original v2 protocol files remain historical and do not define v3 scheduling or stopping.

Preferred runtime is concurrent A on physical GPUs4,5 and B6,7, with two physical ranks per arm and four preserved logical samples/RNG streams per optimizer update. The independent numerical probe passed the unchanged relative-L2<1e-3 criterion. Four-GPU alternating validation segments remain the registered fallback.

12710 additional updates is a safety cap. Fixed development validation occurs every500 additional updates and at the cap if reached; each arm independently uses normalized development FloLPIPS with min_delta0.005 and patience3. Raw development best includes parent5000. Largest-common validated step and independent dev-best comparisons are separate. Final test runs only after both arms stop and model selection is frozen.

The previous README is preserved in `migrations/online_earlystop_v3/before/README.md`. Historical `EXECUTION_PROTOCOL.md`, `protocol.json`, source manifests and failed/blocked records retain their original meaning. Complete results require `final_integrity_v3.json` and verified results publication; code publication alone is not experiment completion.

All execution paths and actual launch provenance are recorded under `migrations/online_earlystop_v3/`; checkpoint files remain local and are excluded from Git publication. No scientific interpretation is supplied.
