# TCG group-isolated data expansion paired continuation v2

Authoritative current state: run_status.json. This is a new long-term worktree experiment; P and E1 are read-only. E1's historical BLOCKED result is retained. Revision details: EXECUTION_PROTOCOL.md and revision_diff.json.

The complete source graph includes archive candidates, reserves, historical exposure and aliases before exclusion. Reserved-touching components are quarantined without waiting for original-upload IDs. A_clean_old_pool is the cleaned parent pool; B uses all remaining qualified local training identities. Counts and S are assigned after the fixed dispersed-content screen and frozen window plans, not inferred from reference counts.

Run/resume in this worktree, with host CUDA access:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_tcg_stage1_data_expansion_ab_groupisolated_v2/runner.py
```

Exclusive stage locks prevent duplicate work. The runner tracks existing v2 graph workers and resumes verified profiles. Training latest/checkpoint receipts control optimizer recovery; no committed update is repeated. Cached PASS receipts are only from this v2 experiment and have source hashes; E1 receipts are not adopted as v2 acceptance.

`current_status.py` prints read-only objective status JSON. Committed training counts come from atomic latest checkpoints rather than an in-flight log. Counts remain null until source pools exist; prepared pool counts are explicitly labeled unfrozen until full plan-window acceptance. It does not launch any work or inspect metrics to select data/checkpoints.

Per-stage logs remain local. Public outputs include bounded log summaries, immutable compressed plans and receipts, source-group evidence, audits, numerical point/frame/transition data, plots and an objective REPORT. Checkpoints/media/flows/streams/viewing previews stay local. Each public file is below50MiB. Fresh A/B evaluation and reused parent156 points are explicitly distinguished. No quality-driven tuning or scientific interpretation is performed.


Startup/resume repair: use `launch_repaired.py` with the same environment and Python as the command above. It creates only deterministic experiment-local output parents and then execs the unchanged runner, forwarding arguments and exit status. `--prepare-only` initializes directories while tracking an already-active runner. Exact source/invocations are in audits/launcher_initialization_*.json and audits/runner_resume_geometry_*.json.

A later strict native-geometry assertion stopped A before additional update11 after10 committed updates. ENGINEERING_REPAIR.md and repairs/native_geometry_v1/ document15 sources/313 slots/31 unique windows whose appended1080 metadata disagreed with audited native heights. The immutable protocol, plans and splits are unchanged; the original manifest/code and checkpoint10 are archived. The current source manifest explicitly records the three-source revision. Resume accepts the exact predecessor checkpoint10 SHA and receipt only, then continues at11. No committed optimizer update is replayed and no tolerance, crop/window, loss, precision, LR or budget is changed.

`observe_run.py --watch` is a CPU-only status observer of the existing task, not a second trainer. Its lease prevents duplicate observers. It reads atomic latest checkpoints and actual lifetime locks, writes logs/live_status.json, and updates dated REPORT snapshots during engineering checks and training. It stops writing REPORT before evaluation/summarization/publication owns that file; observer logs/live_status are excluded from the Git allowlist. Its exact source hash and invocation are in audits/status_observer_launch_*.json. The observer makes no optimizer update and changes no model or training RNG.
