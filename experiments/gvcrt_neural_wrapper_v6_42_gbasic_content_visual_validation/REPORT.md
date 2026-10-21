# Fixed G_basic content and visual validation

24/24 supplementary evaluation points completed; zero new optimizer updates. The fixed G_basic checkpoint SHA256 is 89c1a5baa195c8ffc9695af495d43389e00021b926eae13a0337d3b2eb117fd4.

## Supplementary four-video equal summary

|Metric|O_original mean|G_basic mean|G_basic − original|Improved / tied / degraded|
|---|---:|---:|---:|---:|
|LPIPS|0.1920207481|0.1833526992|-0.008668048831 metric units|12 / 0 / 0|
|DISTS|0.1061913222|0.1015630721|-0.004628250065 metric units|12 / 0 / 0|
|FloLPIPS|0.1886491082|0.1777291334|-0.01091997477 metric units|12 / 0 / 0|
|PSNR|26.25101107|26.38157266|0.1305615936 dB|12 / 0 / 0|
|L1|0.03669535109|0.03605948261|-0.0006358684744 metric units|12 / 0 / 0|
|actual_bpp|0.01420876837|0.01420876837|0 bpp|0 / 12 / 0|

## Per-video results

|Video|Metric|Mean delta|Improved / tied / degraded QP points|
|---|---|---:|---:|
|Kimono1|LPIPS|-0.01111116928 metric units|3 / 0 / 0|
|Kimono1|DISTS|-0.006976456381 metric units|3 / 0 / 0|
|Kimono1|FloLPIPS|-0.01044114557 metric units|3 / 0 / 0|
|Kimono1|PSNR|0.1121173751 dB|3 / 0 / 0|
|Kimono1|L1|-0.0004275708634 metric units|3 / 0 / 0|
|Kimono1|actual_bpp|0 bpp|0 / 3 / 0|
|ParkScene|LPIPS|-0.009135070024 metric units|3 / 0 / 0|
|ParkScene|DISTS|-0.003272518826 metric units|3 / 0 / 0|
|ParkScene|FloLPIPS|-0.01781453697 metric units|3 / 0 / 0|
|ParkScene|PSNR|0.07211551723 dB|3 / 0 / 0|
|ParkScene|L1|-0.0005021319957 metric units|3 / 0 / 0|
|ParkScene|actual_bpp|0 bpp|0 / 3 / 0|
|HoneyBee|LPIPS|-0.004415322522 metric units|3 / 0 / 0|
|HoneyBee|DISTS|-0.003914926201 metric units|3 / 0 / 0|
|HoneyBee|FloLPIPS|-0.004248010142 metric units|3 / 0 / 0|
|HoneyBee|PSNR|0.2161715884 dB|3 / 0 / 0|
|HoneyBee|L1|-0.001002747701 metric units|3 / 0 / 0|
|HoneyBee|actual_bpp|0 bpp|0 / 3 / 0|
|Jockey|LPIPS|-0.0100106335 metric units|3 / 0 / 0|
|Jockey|DISTS|-0.004349098851 metric units|3 / 0 / 0|
|Jockey|FloLPIPS|-0.01117620641 metric units|3 / 0 / 0|
|Jockey|PSNR|0.1218418936 dB|3 / 0 / 0|
|Jockey|L1|-0.0006110233371 metric units|3 / 0 / 0|
|Jockey|actual_bpp|0 bpp|0 / 3 / 0|

Per-QP and separate HEVC-B / UVG summaries are in summary.csv. All fixed point differences are in paired_differences.csv. The original BasketballDrive/BQTerrace 12 points are separately listed in original_reference_12_points.csv and are excluded from these summaries.

## Material entry points

Local HTML: `/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_42_gbasic_content_visual_validation/index.html`. 32 static groups, each with a three-model ROI-box overview and two native-size ROI panels. Eight synchronized 64-frame video groups provide a uniformly resized full overview and two native-size ROI views at the manifest timeline (HEVC-B 24fps, UVG 30fps). H264 is display encoding; lossless PNG and FFV1 ROI sources remain on the server. See visual_index.json and server_artifacts_index.json for paths, sizes and SHA256.

## Audits and limits

Same complete RANS bytes/SHA/consumption, y/f/q, Bridge output and Generator inputs, I0 equality, eval mode, absent parameter gradients and unchanged weights/buffers: PASS. The first dual-model point was validated before the remaining queue. No training or additional model/data scale was introduced.

Four supplementary videos and three QPs per video are a fixed small evaluation; QP points are not independent videos. These videos were absent from the actual v6.40 fine-tuning plan. Official pretraining exposure is unknown; prior evaluation use is not excluded. PSNR deltas are in dB. Perceptual percentage changes are not rate savings. No FID, BD-rate or FPS performance was computed. No subjective visual improvement claim, mechanism explanation, significance, generalization or next experiment is asserted.
