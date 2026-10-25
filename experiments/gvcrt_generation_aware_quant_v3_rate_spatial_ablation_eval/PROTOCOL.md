# Fixed protocol

- Checkpoints: v1 `A/step_300.pt`; v2 `H11/step_300.pt`. All MAP, MEAN,
  ROLL, and IDENTITY runs use the same H11 G and H weights. Inference mode only.
  No visual-condition V, optimizer, training step, or checkpoint update is used.
- Sources: exact v2 `evaluation_sources.json` source hashes, dimensions, RGB
  handling, 64 frame indices, and source FPS. Native I at frame 0, IP=-1,
  external QP shifted by the unchanged eight-frame v2 schedule.
- Header: 16 bytes, big-endian `>4sBBBB8s`: `GV3R`, version 1, variant ID,
  alpha index, zero reserved byte, eight zero bytes. Variant IDs are A=0,
  MAP=1, MEAN=2, ROLL=3, IDENTITY=4. Alpha indices 0..4 map to
  `[0.8, 0.9, 1.0, 1.1, 1.25]`. A, MAP, and IDENTITY always use index 2.
  Every method pays all 16 header bytes. The native codec stream follows
  unchanged. The independent decoder reads mode and alpha from this file.
- MEAN averages the actual H multiplier over the full spatial map of each
  decoded frame, then broadcasts it. The original base quantization step
  retains all of its own spatial and channel variation.
- ROLL circularly shifts the actual H multiplier by
  `4 * floor(floor(height/4)/2)` and likewise in width. No shift search.
- MEAN and ROLL use `clamp(alpha * transformed_multiplier, 0.5, 2.0)`;
  `q_new = clamp_min(q_base * effective_multiplier, 0.5)` followed by the v2
  dtype conversion. The two-pass coordinate ratio is always `q_new/q_base`
  after these operations.
- Stage 1: A QP3,4,5,6 and MAP QP4,6 on all four fixed sources, 24 new
  full tasks. Existing QP2/7 anchors are rewrapped with exact provenance.
  A interpolation targets remain each MAP QP2/7 measured rate.
  Reuse the four existing A QP9 points and the existing MAP UVal0 QP9 point
  after native-stream/weight/source verification. Compare sparse A QP2/7/9
  interpolation against the densified curve at the unchanged targets.
- Stage 2: MEAN, ROLL, IDENTITY at alpha 1 on UVal1 and Jockey, QP2/7,
  12 full tasks. Each MEAN/ROLL comparison targets MAP total bytes at the
  same source and QP. If alpha 1 is farther than 2%, run only the four other
  fixed candidates as rate-only streams. Select minimum absolute total byte
  distance; no image metric enters selection. The selected nonunit alpha then
  gets one full evaluation. A failed 2% match is retained and labelled.
- Metrics: original v2 LPIPS, DISTS, FloLPIPS and PSNR implementations;
  native P payload and analytically estimated P y/z bits are reported
  separately. Interleaved RANS does not provide exact y/z payload bytes.
- Comparison: same QP, nearest measured rate, and bracketed linear/log bpp
  interpolation are distinct observations. No extrapolation or formal BD-rate.
- Final receiver audits decode all 57 persisted protocol streams without
  recomputing metrics. All reused streams must match fresh original-v2 RGB,
  reference, latent and qmap tensor hashes exactly for all 64 frames. The four
  A/IDENTITY source/QP pairs must match native streams and codec state hashes.
  Receiver diagnostics check raw/effective multipliers, actual post-dtype rho,
  clamp and both skip fractions on every P frame. Fixed receiver views and
  their audits are resumable only when their code and stream hashes match.
- Per-task canonical keys include full provenance through source, weight,
  runtime and configuration SHA prefixes. Short file names are storage aliases
  retained from the existing runs; `task_manifest.csv` and `configs/` bind
  each alias to the full hashes. No existing result is renamed or re-encoded.
