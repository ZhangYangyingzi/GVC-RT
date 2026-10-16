# Effective v6.32b protocol and frozen dependency metadata

The executable numerical protocol is specified by `effective_protocol.json`, `config.json` branch-specific fields, `branch_manifest.json`, exact plans and implementation hashes. `protocol.json` remains byte-frozen because already-running calibrated workers and verified imported receipts use its SHA256. `protocol_revision.json` documents inherited metadata that is not used by the numerical pipeline.

- `new_start` was inherited from v6.30 and is not used. Formal frame indices are exactly `eval_manifest.json`, with no resampling.
- Inherited `new_encodes=2392` describes all A/B/C/D adaptation points in the old combined matrix. This run has1196 new C/D encodes plus2116 historical reused points,3312logical points.
- Fixed `global_view`, `canvas`, `padding`, `crop` describe formal testing, not the revised auxiliary training full view. The latter is native H×W, official align16 and unpadding; `auxiliary_view` and geometry audits govern it.
- Inherited resource entries are historical metadata. The new driver actually uses1.2×measured/estimated peak+2048MiB, immediately reserves loading tasks and additionally subtracts live reservations from free memory. It allows multiple workers per card, sets2CPUthreads, and retries OOM at most3times with raised estimates. Actual reservations/peaks/PIDs are in `gpu_schedule.jsonl`.
- Config `optimizer` contains the historical A default; `branch_lrs` is authoritative: A3e-6/5e-7, B/C/D6e-6/1e-6. The complete C/D objectives are in `branch_manifest.json` and frozen `calibration/final.json`; config's inherited objective string describes C1 only.

These are explicit metadata-scope corrections. No sample, loss, coefficient, reference chain, precision, optimizer update or cached numerical artifact is changed by this addendum. Checkpoints retain the exact frozen runtime configuration/protocol hashes; those original bytes remain available for reproduction.
