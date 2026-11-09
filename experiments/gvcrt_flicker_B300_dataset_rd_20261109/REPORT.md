# B300 data-set-level measured R-D comparison

Scope: the existing ULong development subset of 12 videos. This is not the full ULong data set, the 32-video held-out set, or a cross-data-set test. No new decoding or training was performed.

Fixed states: independent Original GVC-RT; A2000 s0.5; flicker-pilot branch B step300 s0.5 (TCG + LoRA + Bridge.mlp[3]). B300 is not the earlier expanded-pool B+1500 TCG checkpoint.

108 reused measured points, 36 per model. Native 64 frames per video, QP0/5/9. Spatial metrics use64frames; temporal metrics use63transitions; sustained P-to-P residual metrics use62transitions. The separate person_dog/Cactus 128-frame comparison is not pooled here.

Within each QP, all12videos have equal weights. Per-video PSNR derives from total pixel SSE; the data-set PSNR is the equal-video mean of those PSNRs. All matching points have identical source/RGB, actual QP schedule, real stream SHA/bytes/bpp, codec trace and metric networks. Curves connect only the three actual measured points. No BD-rate or extrapolation.

|QP|Mean bpp|Model|PSNR|MS-SSIM|LPIPS|DISTS|FloLPIPS|
|---:|---:|---|---:|---:|---:|---:|---:|
|0|0.00241627|Original GVC-RT|23.4247080|0.8269027|0.2125176|0.1198621|0.2122307|
|0|0.00241627|A2000 s0.5|23.0458867|0.8238708|0.1964168|0.1069659|0.1906915|
|0|0.00241627|B300 s0.5|23.4689332|0.8295578|0.1953932|0.1060431|0.1900098|
|5|0.00854806|Original GVC-RT|25.6868332|0.9005805|0.1348055|0.0803040|0.1260936|
|5|0.00854806|A2000 s0.5|25.5675835|0.9001417|0.1218623|0.0693437|0.1143044|
|5|0.00854806|B300 s0.5|25.9624067|0.9034013|0.1215191|0.0684796|0.1139659|
|9|0.01919020|Original GVC-RT|26.6056401|0.9229633|0.1105867|0.0677083|0.0996859|
|9|0.01919020|A2000 s0.5|26.5790265|0.9226557|0.0998728|0.0585365|0.0904715|
|9|0.01919020|B300 s0.5|26.9717901|0.9254038|0.0995713|0.0575704|0.0900559|

B300 differences below use 100*(candidate mean/reference mean-1). Negative values mean a numerical reduction; metric direction is explicitly retained. PSNR absolute differences are in dB.

|Reference|QP|LPIPS change %|DISTS change %|FloLPIPS change %|Sobel residual change %|PSNR difference dB|
|---|---:|---:|---:|---:|---:|---:|
|Original|0|-8.0579|-11.5291|-10.4702|-11.5978|+0.04423|
|Original|5|-9.8559|-14.7245|-9.6180|-7.6062|+0.27557|
|Original|9|-9.9609|-14.9729|-9.6603|-6.5583|+0.36615|
|A2000|0|-0.5212|-0.8627|-0.3575|-10.0770|+0.42305|
|A2000|5|-0.2816|-1.2461|-0.2961|-8.4154|+0.39482|
|A2000|9|-0.3020|-1.6504|-0.4593|-7.8863|+0.39276|

All per-video differences, including increases and decreases, are retained in paired_deltas_by_video_qp.csv. No new model selection or training is authorized by this descriptive aggregation. B300 has not been evaluated on the complete HEVC/UVG/MCL or32-video ULong held-out grids in this package.
