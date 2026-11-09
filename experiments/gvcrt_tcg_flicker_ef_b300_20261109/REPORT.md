# GVC-RT E/F B300 execution report

Factual snapshot UTC epoch: 1794235526.9112213.

Status: INCOMPLETE

Initialization: B300 `/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_flicker_pilot_abcd_20261109/experiments/gvcrt_tcg_flicker_pilot_abcd_20261109/B/training/checkpoints/step_0300.pt`; SHA256 `7465b57f20bfe2a8d6064a87155cf45c31a488cdeaa68111a228c28f6335ea85`. Source commit `2e70bdc4ad99934d987fe8c484f8430345792b83`.

E GPU0/1; F GPU2/3. Fresh AdamW; global batch4 (2 windows/rank), maximum300 additional updates. Independent original codec reference trajectory. No final test evaluation.

Scope: ULong开发子集12视频, QP0/5/9,64 frames. Original/A2000/B300 inherited36-point records retained with explicit provenance. B300 supplemental detail is reported only when actually computed.

Incremental cache: initially one GPU producer; now 4 controlled producers on GPUs [4, 5, 6, 7], authorized by the user.16 distinct startup windows,32-window prefetch, bounded bytes and backpressure. Rank readiness coordinated at full optimizer boundaries. Both cursors protect cache retention.

E: additional optimizer updates=35; completed development points=0; stop reason=not stopped/completed.
F: additional optimizer updates=35; completed development points=0; stop reason=not stopped/completed.

No method superiority conclusion is made. MP4 preview bytes are excluded from R-D data. All neural codec bpp values originate from verified real RANS streams.

See status.json for current stage, audits for measured checks, development_metrics.csv and aggregate_by_QP.csv for bounded-scope results, training_history.csv for formal optimizer updates.
