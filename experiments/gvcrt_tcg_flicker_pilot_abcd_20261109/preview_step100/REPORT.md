# Step100 intermediate viewing export

Snapshot: 2026-11-09 17:43:07 +0800. Preview COMPLETE; full pilot remains RUNNING.

12 existing ULong development sources; equal video weights within each QP. LPIPS is lower-is-better. Metrics are reused from complete36-point evaluations, not recomputed from MP4 previews.

|Model|LPIPS QP0|LPIPS QP5|LPIPS QP9|
|---|---:|---:|---:|
|Original GVC-RT|0.212518|0.134805|0.110587|
|A2000 s0.5|0.196417|0.121862|0.099873|
|A +100|0.196208|0.122119|0.100075|
|B +100|0.196020|0.122003|0.099986|
|C +100|0.197596|0.122776|0.100629|
|D +100|0.197293|0.122603|0.100495|

Fixed clip:24747b35-04ca-416c-b1d0-0bc670fbac99, the protocol preselected low-motion high-texture first development source. Native1920x1080, frames0--63,30fps, external QP5. No candidate-quality-based video selection.

Shared GVC-RT stream:66704bytes,250.14kbps. MP4 preview sizes are separate from codec stream size. GT has no codec bitrate.

Seven individual native-resolution lossless FFV1 videos and H264 viewing previews are saved locally. Candidate raw reconstruction and float32 RGB hashes match the existing formal measured records for every frame. See integrity.json and manifest.json.

Comparisons:

- /Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_flicker_pilot_abcd_20261109/experiments/gvcrt_tcg_flicker_pilot_abcd_20261109/preview_step100/24747b35-04ca-416c-b1d0-0bc670fbac99_QP5_step100_stream66704B_scaled_overview_comparison.mp4
- /Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_flicker_pilot_abcd_20261109/experiments/gvcrt_tcg_flicker_pilot_abcd_20261109/preview_step100/24747b35-04ca-416c-b1d0-0bc670fbac99_QP5_step100_stream66704B_native_center_crop_comparison.mp4

The overview scales each panel to960x540. The native center-crop comparison uses the same fixed640x360 region in each model without resizing. Both previews carry explicit BT709 limited-range metadata.

The initial composition attempt failed because this FFmpeg lacks the xstack fill option. The original failure is preserved under audits/. CPU-only composition uses the identical seven-panel positions and an explicit black eighth tile. Frozen model/training/evaluation sources are unchanged.

These64-frame previews supplement the pending128-frame final videos. No optimizer or checkpoint update is performed by this export.
