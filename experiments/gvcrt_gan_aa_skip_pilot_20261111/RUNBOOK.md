# Execution and recovery

This experiment is already running on the server. `runner.py status` reads its
status; `runner.py resume` uses the singleton lock and adopts matching surviving
workers by PID, exact argv, and cwd. Use the protocol's Python interpreter and
this directory as cwd. The runner does not require an attached terminal.

The fixed training order is `plans/train.json.gz`: 2,000 real windows, batch 4,
ULong/Vimeo 1:1, external QP0–9 balanced. `plans/calibration.json.gz` has 24
separate windows. The grouped Vimeo list excludes its recorded historical
heldout source groups and is intersected with the audited official train list.
`audits/data_isolation.json` records the actual H development composition:
32 ULong videos, not a multi-dataset development set. No final test is scheduled.

Each group's atomic `training/latest.pt` is the committed optimizer cursor.
G, D, both optimizers, G/D RNG, sampling cursor, ramp position, protocol/plan/
calibration hashes are saved together. Archives are saved every 25 updates.
Normal execution stops at 500 G updates. Waiting workers unload their CUDA
contexts and resume from the committed cursor when their next four windows are
ready. They never take samples in cache-arrival order.

There is one shared producer with a 48-window lookahead measured from the
slowest active consumer. Native evaluation runs on a released training card.
At 250, J0 releases GPU0 for step0/250 development evaluation while other groups
can continue within the shared cache horizon; J0 resumes after that evaluation.
The final evaluation adds the 500-step states. Already completed exact-identity
points are reused. All evaluation is from the H development manifest.

`status.json` is refreshed every five seconds and contains actual committed
steps, process IDs, stage, speed and compute ETA. That ETA excludes cache waits,
evaluation and media export. It is not the completion ETA of the whole task.
`receipts/processes/` and `logs/` identify each task-owned process. GPU0–3 only
are visible to these workers. Existing unrelated processes remain untouched.

The reconstruction path is PyTorch FP32, with the fixed codec in FP16. The
common I frame is the independent original codec's FP16 I reconstruction.
The standalone Original P receiver loads only original I/P weights and uses
FP32 reconstruction, without pilot, TCG or LoRA. Recorded reference-path speed
does not describe the optimized CUDA deployment implementation.

`audits/early_development_result.json` records the first prescribed point
evaluated concurrently with training. The early driver temporarily suspended
only its identity-checked J0 trainer and resumed it in `finally`.
`audits/Original_reference_latency.json` records a separate timing run with
bitwise comparison to the scored Original output.

Reporting-only source revisions are recorded in `source_manifest.json` and
`audits/reporting_only_source_revision.json`. Generator, losses, calibration,
optimizer, sampler and metric arithmetic were not changed by those revisions.
Bootstrap cache identity enrichment is recorded in `audits/cache_key_alias/`;
the immutable codec/RAFT tensors were retained after receipt/source checks.

Outputs are `REPORT.md`, per-video/per-dataset/paired CSVs, `plots/`, and
`media/{native,crops,panels}`. `audits/GT_video_exports.json` is a partial GT-only
media receipt; it does not imply candidate videos are complete.
`video_manifest.json` is produced after the prescribed full export.

Publication uses only this experiment's paths on
`codex/gvcrt-gan-aa-skip-pilot-20261111`, without force-push. Weights, cached
tensors, source media, binaries and lossless videos remain on the server.
Their paths/hashes are in the audit and integrity receipts.
`publication_status.json` records the latest verified remote commit.

Numerical, data, codec or implementation faults retain their records. A group
fault stops that group while independent groups continue; a common producer
fault requests a safe stop of affected work. `COMPLETE` requires the prescribed
training, evaluation and delivery. Partial progress remains `RUNNING` or a
recorded fault state; no quality threshold changes the objective or budget.
