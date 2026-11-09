# Person/dog and Cactus: fixed B300 vs independent Original

Snapshot UTC: 2026-11-09T14:35:05Z

User-requested post-hoc comparison of existing held-out/historical sources. Fixed B300 checkpoint; no training, checkpoint selection or loss changes.128 native consecutive frames, real I0 plus127 P frames, external QP0/5. Person/dog uses source128..255 at30000/1001fps; Cactus uses0..127 at50fps. These128-frame results are separate from previous64-frame tables.

Original: pinned b2a0895 original I/P weights without pilot,TCG orLoRA. B300: original codec plus pilot receiver,B+1500 TCG,A2000 LoRA,and the completed300-step branchB TCG/LoRA/Bridge_mlp3 overlay. No Router or history alignment. Inference gain0.5. Codec reference trajectory remains the verified original trajectory, matching the completed pilot protocol.

RGB scoring: float before clamp and normalization, then native crop. Spatial metrics use128frames; temporal metrics127transitions. PSNR comes from total pixel SSE. Old PWC samplers unchanged; new MCResidual metrics retain their distinct RAFT pixel-center version.

|Video|QP|Model|bpp|kbps|PSNR ↑|MS-SSIM ↑|LPIPS ↓|DISTS ↓|FloLPIPS ↓|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
|person_dog|0|Original GVC-RT|0.001166|72.465659|17.765278|0.532348|0.331681|0.158942|0.291566|
|person_dog|0|B300 TCG+LoRA+Bridge_last s0.5|0.001166|72.465659|17.940826|0.545224|0.311632|0.139660|0.279954|
|person_dog|5|Original GVC-RT|0.004092|254.291958|19.387329|0.699491|0.239059|0.111665|0.200048|
|person_dog|5|B300 TCG+LoRA+Bridge_last s0.5|0.004092|254.291958|19.666546|0.708296|0.213236|0.085123|0.186659|
|Cactus|0|Original GVC-RT|0.002041|211.621875|21.791820|0.766705|0.287697|0.096590|0.336093|
|Cactus|0|B300 TCG+LoRA+Bridge_last s0.5|0.002041|211.621875|21.878289|0.769745|0.286038|0.103635|0.333603|
|Cactus|5|Original GVC-RT|0.006842|709.337500|24.064176|0.856424|0.188417|0.067327|0.213094|
|Cactus|5|B300 TCG+LoRA+Bridge_last s0.5|0.006842|709.337500|24.277652|0.857779|0.193678|0.072964|0.220022|

Both old PWC warping errors and new RGB/Sobel temporal residuals, I-to-P and P-to-P detail, real stream hashes and paired absolute/relative deltas are retained in metrics.csv,paired_deltas.csv and the per-frame/per-transition records. No cross-video/dataset pooling or BD-rate estimate.

Full source geometry1920x1080. The overview uses640x360 panels. Fixed native640x640 crops use no resizing. Additional320x320 crops use explicitly labeled2x nearest-neighbor magnification. ROI coordinates were defined from GT-only first/middle/last views before model metrics. All models use identical windows/crops within each video/QP.

FFV1 files retain lossless RGB8 viewing exports. H264 CRF12 MP4 files are viewing previews; neural metrics are computed on original float32 decoder outputs. Codec stream bytes and kbps in filenames do not refer to MP4 container size. Large videos/cache/weights remain local.

Integrity: PASS.8/8points,4 freshly encoded and independently verified canonical RANS streams,1024scoredframes,1016transitions. B300 independently decoded each matching stream; measured bpp/bytes/stream hashes/codec traces identical for each paired video/QP. No optimizer or weight updates.

See comparison_manifest.json for exact full-native/overview/crop/zoom viewing paths; rois.json for GT-defined coordinates; sources.json for protected source hashes. Publication is separately recorded in publication_status.json.
