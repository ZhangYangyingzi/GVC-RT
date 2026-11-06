# Internal history gate + Router/TCG joint execution

This protocol implements the single joint line prescribed on20261106. The exact settings are in protocol.json. The source manifest freezes executable code before formal updates.

B+1500 is cumulative TCG6500. Load original I/P, pilot receiver/buffers and exact B TCG; create a new seed20261106 Router at M0=.2. No historical Router or optimizer is loaded; trained fusion heads are retained.

Gate only previous_adapter(previous), before concat with current_adapter(current). Keep one spatial mask, shared across3layers; remove outer residual gating. M1 zeros direct history, not the whole residual. Inputs remain decoded codec features. All Router/TCG computations and parameters are FP32 with disabled autocast; output returns to incoming dtype. Parent half() preserves FP32 values. Other modules retain historical precision.

Only Router33889 and TCG137760 parameters update, total171649. Fresh AdamW groups1e-4/1e-5, fixed settings, combined clip1.0, SUM4/4. Preserve historical64frame causal detach and exact loss/VJP4096 ordering; no added groups/modules/losses/BPTT.

Use audited pool8470/quarantine236 and identical fixed splits/windows, source cursor1500, batch4=2UVG+2ULong. Revisited data are recorded, not claimed independent. Baseline normalization uses new matched-FP32 B inference, not historical FP16 scores.

Validate96points every250 updates, cap3000, patience4, min_delta.001. Trained raw best excludes initialization; score ties use earlier checkpoint. Final choice is raw dev-score argmin against independent matched baseline; threshold flag is separately recorded and selection does not assert effective improvement. Freeze selection before new final-test quality scoring. Always test and plot the best trained joint checkpoint.

Final45videos×QP0/5/9 use original RANS/native64frames and matching RGB FP32-first scores. Fresh joint encodes and independent decoders; matched baseline may decode audited old B streams. Preserve separate historical vs matched precision results. Four separate dataset plots and45 per-video plots use actual measured points, no BD-rate/extrapolation. Fixed prior diagnostic sequences/frames use0–1 mask colors.

All new writes are within the project. Preserve historical experiments and current dirty work. Atomic checkpoints include both Router andTCG, optimizer,4RNG states, cursor, receipts and validation state. No duplicate committed updates. Publish only allowlisted small task artifacts, no weights/media/cache/PDF/secrets, to the authorized new branch, with verified remote HEAD/core blobs.
