# Fixed protocol

O_original and the exact v6.40 G_basic step100 are evaluated on Kimono1, ParkScene, HoneyBee and Jockey, QP0/5/9, 64 frames. Native I0, P1–63, IP=-1, no extra reset, 1920×1080 with inherited 1088 padding. Models deploy in the inherited float16 precision. No training, optimizer update, weight tuning or bit allocation experiment.

All 12 shared original RANS streams were verified; two models independently decode each. Hooks verify exact y/f/q, Bridge output and Generator inputs; I output and full consumed byte counts agree. PSNR uses aggregate RGB SSE; metrics use floating output. Visual PNG rule is clamp[0,1], round(255*x), uint8, identical across GT/models. ROIs and frames were fixed using GT before reconstruction.

Outcomes use absolute metric delta tolerance 1e-12. Perceptual percentage changes are not bitrate savings. Supplementary videos and the original two reference videos remain separate. Official pretraining exposure is unknown.
