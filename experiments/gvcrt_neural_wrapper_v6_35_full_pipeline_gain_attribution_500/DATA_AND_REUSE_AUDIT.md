# Data and reuse audit

The fixed evaluation manifest combines the six exact U-Long clips from v6.34b with v6.33 UVG holdout entries 2 and 3 and all five v6.33 HEVC-B entries. Source paths, source hashes, frame indices, per-frame RGB hashes, and loader metadata are retained unchanged. Each actual task reloads and rechecks the 64 RGB frames before evaluation.

The 75 reused points are individually checked against the fixed manifest and source receipts. For U-Long, v6.34b task receipts for original/N500/P500 are checked, including complete bytes, bitstream and feature SHA, RGB/frame identity, QP schedule, compression hash and decode/state-sync flags. For UVG/HEVC original, v6.24 original rows are checked against v6.33 source/frame metadata, actual QP sequence, model hashes, bitstream size/hash, feature hash, frame metric/transition hashes, and decode/state synchronization.

Training/test isolation is reported from the v6.34b first-500 shared-plan identities and the historical split manifests. The supplied training and test lists are cross-checked by source path, video identity and source SHA. This audit does not claim exhaustive content-level near-duplicate search or full historical pretrained-data exposure knowledge.
