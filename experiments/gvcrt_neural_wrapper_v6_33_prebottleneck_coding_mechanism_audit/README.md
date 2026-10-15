# v6.33 pre-bottleneck coding mechanism audit

Evaluation only, fixed B1 weights, no optimizer or checkpoint writes.
Seven methods × 46 fixed videos × ten external QPs = **3220 points**.
Each formal point has 64 causal frames, native I0 bypass, persisted real RANS
stream, fresh independent decode, actual file bytes, per-frame payload and
P-frame quality/fidelity/coding diagnostics. Input sampling is copied exactly
from v6.31a; 20fps is a bitrate-display convention only.

`B1_alpha0` removes only source P and retains the B1 receiver. It is neither
Original GVC-RT nor a fair independently optimized No-P baseline. Quality at
alpha !=1 is a local intervention without receiver readaptation.
Teacher identity is **paper_compatible_teacher_proxy**, not a verified exact
original GVC-RT training teacher. No Strong No-P claim is permitted.

The background driver continues through evaluation, aggregation, bootstrap,
figures, REPORT.md, integrity verification and normal current-branch commit/push.
No background process calls Codex, an LLM or a model API. Existing unrelated
processes and worktree modifications are preserved. Failed groups preserve
receipts; reusing `launcher.py` resumes remaining groups after a failure has
been diagnosed. Code/protocol signatures cannot silently change on resume.

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_33_prebottleneck_coding_mechanism_audit/launcher.py
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_33_prebottleneck_coding_mechanism_audit/check_once.py
```

Launch gates: preflight.json, selftest.json and sanity/sanity.json PASS.
The first full-64-frame formal point is also a resource canary before admitting
multiple workers. GPUs4–7 only, one visible GPU per process, measured peak
memory/RSS reservations with 25% margin and GPU headroom. OOM causes evaluation
group backoff/resume, without changing frames, precision or codec settings.

Full receipts are in server `parts/**/point.json` and eventually bundled into
`point_receipts.jsonl.gz`. Streams, full reference tensors and logs stay on the
server. Publication includes only this experiment's code, configuration,
provenance, tables, figures, receipts and report. Large CSVs are losslessly gzip
archived for publication; `restore_large_text.py` verifies SHA before restoration.
Publication conflict produces publication_status.json and preserves local results;
there is no force push or unrelated-change commit.
