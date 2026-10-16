# GLC precompression transfer v2 rate/loss control

Status: preparation. Four fixed latent-sender branches continue the same v1 C step1000 weights with fresh optimizers. Source experiment and codec stay read-only.

Training seed 20261016; mixed clean U-Long/UVG pool; 1000 added updates per branch; 64-frame clips, P49–63 supervision averaged over 15 frames. Actual evaluation rate is `real_local_nonofficial_rans`. The common training rate backend is frozen only after paired training-pool audits. No test-data calibration or checkpoint selection.

See protocol.json, config.json, audits/, RUNBOOK.md and status.json. Preparation publication does not mean training or evaluation is complete.
