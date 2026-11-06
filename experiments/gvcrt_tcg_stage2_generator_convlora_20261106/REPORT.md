# Generator Conv-LoRA execution snapshot

UTC: 2026-11-06T15:56:49Z

Status:READY; stage:engineering_complete_ready_for_formal; committed optimizer updates:0.

One LoRA-only line; rank4, alpha/r1,59392 parameters, learning rate5e-5. Original generator/TCG/codec/Bridge and quality networks frozen. No Router.

Baseline is strict original I/P + pilot receiver + B1500 TCG overlay, cumulative frozen TCG6500. It is not original GVC-RT weights.

Source cursor1500, effective batch4 (2UVG+2ULong), inherited loss and VJP4096. Max3000, validation250, patience4, min_delta.001.

Every baseline/candidate point independently decodes the same verified pre-encoded stream. Exact bpp/hash checks precede scoring.

Full acceptance and final test are pending; no completion claim. Historical SHA blocker and explicit user correction are preserved in audits/sha_correction_20261106/.
