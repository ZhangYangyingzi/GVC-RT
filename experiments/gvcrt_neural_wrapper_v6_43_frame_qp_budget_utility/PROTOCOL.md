# Fixed protocol

The full authorized specification is in REQUEST.txt. Machine-readable conditions are in protocol.json. This is an offline GT-assisted oracle diagnostic, with zero training updates.

Actual QPs are copied from the v6.42 q5 streams; only listed frame indices are changed, with no shift or clamp. Every stream is encoded from native I0 and all dependent P frames are re-encoded. Every persisted stream is independently decoded using its own QP fields. All state comparisons and baseline reproduction use exact SHA256 checks. Frame packet accounting assigns preceding SPS bytes to that frame, including the initial SPS in frame 0.

Whole PSNR comes from summed RGB SSE. Frame/transition metrics are separate. LPIPS is the primary selection metric; DISTS and L1 are supplemental. Positive efficiencies use a shared window-bit denominator and require positive whole-stream and window cost. Nonpositive costs and negative gains remain in raw tables. Transfer rules, grid, tie breaks and budget intervals are fixed before encoding.

Two videos and multiple within-video interventions are not independent video samples. Same QP offsets do not imply equal bit costs; impulse gains do not imply fixed-budget gains. QP affects encoding features and Generator conditions together. No pure-latent causal interpretation, deployable policy claim, FID, BD-rate or speed evaluation.
