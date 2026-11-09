# Intermediate viewing preview

Fixed source: the first entry in the existing `quick_development.json`, selected
before candidate results by the `low_motion_high_texture` GT-only rule. This is
sequence `24747b35-04ca-416c-b1d0-0bc670fbac99`, native1920x1080, frames0--63,
30fps, external QP5. These are64 original consecutive frames, about2.13seconds.

Models: GT, independent Original GVC-RT, fixed A2000 s0.5, and all four branches
at100 committed updates. No model selection or optimizer update occurs here.
This preview supplements the planned128-frame videos after formal training;
it does not replace those deliverables.

Candidate models independently decode the existing audited RANS stream from
the first I frame. All64 reconstruction and RGB tensor hashes must match the
saved point records from the existing formal evaluation. The Original clip uses
the cache of the independently verified Original decoder. Every clip uses the
same native source window, internal QP schedule and codec reference trajectory.

Individual `.mkv` files are lossless FFV1 RGB8 viewing exports. First-frame RGB
roundtrip is checked exactly. `.mp4` files are H264 previews for viewing, using
explicit full-RGB to limited-BT709 conversion; neural metrics remain those of
the measured float32 decoder tensors. Codec stream bytes and kbps in filenames
refer to the GVC-RT stream, rather than the MP4 preview file size. GT has no
codec bitrate label.

The overview reduces each panel to960x540. The center-crop comparison instead
uses the fixed native640x360 center region with no spatial resizing. It uses
the same crop in every model. Both are labeled; individual clips retain full
native resolution. Neither display is selected by measured candidate quality.

See `manifest.json` and the per-model `manifests/` for actual paths, checkpoint
and stream identities, image hash checks, and viewing-file hashes. Large
videos remain local. This worker shares GPU4 with the ongoing experiment;
its elapsed time is not an isolated inference benchmark.

`measured_quality_step100.csv` contains the already measured36-point quality
means per model and QP (12 ULong development videos with equal video weights).
It is a summary of the existing results, not an additional metric evaluation.
