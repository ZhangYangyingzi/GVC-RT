# V6.27 MS-ILLM diagnostic and MCL-JCV evaluation

No new G/D training or optimizer updates. Diagnostic points: 240/240; MCL-JCV: 1500/1500, N=30. Numerical work complete; publication is verified separately in final_integrity.json.

## Protocol and limitations

Seen:40 windows. Unseen:20 windows (10 U-Long and10 UVG), outside v6.26 source intervals. UVG uses additional canonical windows of the same permitted source videos with the inherited stride4/color semantics; no new training. The first50 diagnostic windows and their valid results are preserved with an audited plan revision. Unseen means outside that round's sampled source intervals, not historically never seen.

Raw original YUV_source files,8-bit yuv420p,1920×1080, native24/25/30fps from filename with frame-byte/archive-member checks. First64 continuous frames, no temporal resampling. All methods use fixed historical BT709 limited-to-full conversion; untagged raw files do not establish intrinsic matrix/range metadata.

Sources: [USC MCL-JCV](https://mcl.usc.edu/mcl-jcv-dataset/), [DCVC test conditions](https://github.com/microsoft/DCVC/blob/main/DCVC-family/test_conditions.md). The local valid source count governs this evaluation.

Confirmed overlap:1; suspected fingerprint candidates:0. Full and confirmed-overlap-excluded statistics are both saved. Sparse fingerprints cannot exclude unknown, shifted or re-encoded exposure; official pretraining exposure is unknown.

## Implementation and local supervision

No confirmed historical MS-ILLM training/condition/gradient implementation error in audited paths. Two v627 CPU summary field-parser errors were found and fixed, with logs and diffs preserved; weights and raw scores were unchanged. This is not proof against all untested errors.

insufficient evidence for reliable local content learning across available panels; report real content accuracy and frequency references separately from fake detection.

|Domain|Panel|Real content top1|Majority top1|Real content top5|Majority top5|Fake AUROC|CE real|Label entropy|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|ulong|seen|0.004596|0.018145|0.021253|0.062614|0.500122|8.611779|6.689524|
|uvg|seen|0.000732|0.002822|0.005267|0.013525|0.499767|8.685367|6.816116|
|ulong|unseen|0.003079|0.005143|0.013991|0.021803|0.500711|8.629280|6.770894|
|uvg|unseen|0.000495|0.004505|0.003665|0.018965|0.498889|8.601298|6.763064|

Under normal matching conditions, mean reconstruction−real CE_content=+0.000001, p_fake=-0.000002. These paired responses are directly measured; fake recognition is evaluated separately from content accuracy.

zero/shuffled probes are OOD; effects alone do not establish a conditional shortcut; current evidence does not isolate excessive reliance.

## Gradients and generator response

No parameters or buffers changed during the audit. Restored complete15-frame BG gradient ratio max=0.0500000000; reconstruction and adversarial unit scale4096; FP32 restoration; one simulated global clip, no optimizer.step. Image and BG gradients and deepest DISTS contribution passed. C_1000 mean cropped reconstruction MAE versus fixed B1=0.00440202.

Full1000-step A/B/C logs, four step-segment distributions and bridge/generator parameter-change tables are provided. A5% gradient norm ratio is not a5% AdamW parameter-update bound; B1/GAN gradient cosine does not directly measure FID/LPIPS gradient conflict.

## MCL-JCV same-rate results

Signed delta=method−reference; all four metrics are lower better. Percent=100*(dataset method_mean−reference_mean)/reference_mean; not BD-rate. Paired metrics use equal-video weighting of per-video five-model common intervals; FID uses fixed pooled features and dataset-mean bpp.100 uniform ln(bpp) points, PCHIP, no extrapolation.

|Method|Reference|Metric|Method mean|Reference mean|Delta|Change (%)|
|---|---|---|---:|---:|---:|---:|
|B_decoder_conditional_projected_1000|A_decoder_continue_1000|LPIPS|0.15140696|0.15126220|0.00014475|0.09569759|
|B_decoder_conditional_projected_1000|A_decoder_continue_1000|DISTS|0.09754833|0.09743381|0.00011452|0.11753116|
|B_decoder_conditional_projected_1000|A_decoder_continue_1000|FloLPIPS|0.15026605|0.15008843|0.00017762|0.11834431|
|C_decoder_illm_1000|B_decoder_conditional_projected_1000|LPIPS|0.15148342|0.15140696|0.00007646|0.05050242|
|C_decoder_illm_1000|B_decoder_conditional_projected_1000|DISTS|0.09751922|0.09754833|-0.00002911|-0.02984309|
|C_decoder_illm_1000|B_decoder_conditional_projected_1000|FloLPIPS|0.15039969|0.15026605|0.00013363|0.08893187|
|C_decoder_illm_1000|A_decoder_continue_1000|LPIPS|0.15148342|0.15126220|0.00022122|0.14624834|
|C_decoder_illm_1000|A_decoder_continue_1000|DISTS|0.09751922|0.09743381|0.00008540|0.08765299|
|C_decoder_illm_1000|A_decoder_continue_1000|FloLPIPS|0.15039969|0.15008843|0.00031126|0.20738143|
|A_decoder_continue_1000|original|LPIPS|0.15126220|0.16061428|-0.00935207|-5.82269184|
|A_decoder_continue_1000|original|DISTS|0.09743381|0.10113957|-0.00370576|-3.66400152|
|A_decoder_continue_1000|original|FloLPIPS|0.15008843|0.16000002|-0.00991159|-6.19474076|
|A_decoder_continue_1000|B1_initial|LPIPS|0.15126220|0.15104972|0.00021248|0.14066992|
|A_decoder_continue_1000|B1_initial|DISTS|0.09743381|0.09774969|-0.00031588|-0.32314922|
|A_decoder_continue_1000|B1_initial|FloLPIPS|0.15008843|0.14946486|0.00062357|0.41720292|
|B_decoder_conditional_projected_1000|original|LPIPS|0.15140696|0.16061428|-0.00920732|-5.73256643|
|B_decoder_conditional_projected_1000|original|DISTS|0.09754833|0.10113957|-0.00359124|-3.55077670|
|B_decoder_conditional_projected_1000|original|FloLPIPS|0.15026605|0.16000002|-0.00973397|-6.08372757|
|B_decoder_conditional_projected_1000|B1_initial|LPIPS|0.15140696|0.15104972|0.00035724|0.23650212|
|B_decoder_conditional_projected_1000|B1_initial|DISTS|0.09754833|0.09774969|-0.00020136|-0.20599786|
|B_decoder_conditional_projected_1000|B1_initial|FloLPIPS|0.15026605|0.14946486|0.00080119|0.53604096|
|C_decoder_illm_1000|original|LPIPS|0.15148342|0.16061428|-0.00913086|-5.68495910|
|C_decoder_illm_1000|original|DISTS|0.09751922|0.10113957|-0.00362035|-3.57956014|
|C_decoder_illm_1000|original|FloLPIPS|0.15039969|0.16000002|-0.00960033|-6.00020607|
|C_decoder_illm_1000|B1_initial|LPIPS|0.15148342|0.15104972|0.00043370|0.28712398|
|C_decoder_illm_1000|B1_initial|DISTS|0.09751922|0.09774969|-0.00023047|-0.23577947|
|C_decoder_illm_1000|B1_initial|FloLPIPS|0.15039969|0.14946486|0.00093483|0.62544955|
|B_decoder_conditional_projected_1000|A_decoder_continue_1000|FID|32.94967211|32.99246803|-0.04279593|-0.12971423|
|C_decoder_illm_1000|B_decoder_conditional_projected_1000|FID|33.05102220|32.94967211|0.10135009|0.30759060|
|C_decoder_illm_1000|A_decoder_continue_1000|FID|33.05102220|32.99246803|0.05855417|0.17747738|
|A_decoder_continue_1000|original|FID|32.99246803|31.39620339|1.59626465|5.08426012|
|A_decoder_continue_1000|B1_initial|FID|32.99246803|33.11759484|-0.12512681|-0.37782578|
|B_decoder_conditional_projected_1000|original|FID|32.94967211|31.39620339|1.55346872|4.94795087|
|B_decoder_conditional_projected_1000|B1_initial|FID|32.94967211|33.11759484|-0.16792274|-0.50704992|
|C_decoder_illm_1000|original|FID|33.05102220|31.39620339|1.65481881|5.27076091|
|C_decoder_illm_1000|B1_initial|FID|33.05102220|33.11759484|-0.06657264|-0.20101895|

A_decoder_continue_1000 vs original: all three LPIPS/DISTS/FloLPIPS means non-increasing; FID mean higher. This refers only to the measured fixed64-frame MCL-JCV panel.

A_decoder_continue_1000 vs B1_initial: not all three LPIPS/DISTS/FloLPIPS means non-increasing; FID mean non-increasing. This refers only to the measured fixed64-frame MCL-JCV panel.

B_decoder_conditional_projected_1000 vs original: all three LPIPS/DISTS/FloLPIPS means non-increasing; FID mean higher. This refers only to the measured fixed64-frame MCL-JCV panel.

B_decoder_conditional_projected_1000 vs B1_initial: not all three LPIPS/DISTS/FloLPIPS means non-increasing; FID mean non-increasing. This refers only to the measured fixed64-frame MCL-JCV panel.

C_decoder_illm_1000 vs original: all three LPIPS/DISTS/FloLPIPS means non-increasing; FID mean higher. This refers only to the measured fixed64-frame MCL-JCV panel.

C_decoder_illm_1000 vs B1_initial: not all three LPIPS/DISTS/FloLPIPS means non-increasing; FID mean non-increasing. This refers only to the measured fixed64-frame MCL-JCV panel.

Three perceptual metrics and FID are reported independently. Per-video and low/middle/high equal-ln-rate-band tables are saved; interpolation samples are not independent experiments.

## FID mean/covariance decomposition

Inception2048, fixed GT pool, float64 sample covariance ddof1. Shared linear ln(bpp) component interpolation is diagnostic; total main FID remains PCHIP.

|Method|Reference|Mean delta D_mu|Mean delta D_cov|Mean diagnostic delta FID|
|---|---|---:|---:|---:|
|B_decoder_conditional_projected_1000|A_decoder_continue_1000|+0.003789|-0.046520|-0.042731|
|C_decoder_illm_1000|B_decoder_conditional_projected_1000|-0.001513|+0.102864|+0.101350|
|C_decoder_illm_1000|A_decoder_continue_1000|+0.002276|+0.056343|+0.058619|
|A_decoder_continue_1000|original|-0.040628|+1.639562|+1.598934|
|A_decoder_continue_1000|B1_initial|-0.026400|-0.100257|-0.126658|
|B_decoder_conditional_projected_1000|original|-0.036839|+1.593042|+1.556203|
|B_decoder_conditional_projected_1000|B1_initial|-0.022611|-0.146778|-0.169389|
|C_decoder_illm_1000|original|-0.038353|+1.695906|+1.657553|
|C_decoder_illm_1000|B1_initial|-0.024125|-0.043914|-0.068039|

Covariance changes alone do not identify blur or texture mechanisms. Fixed actual-rate visual panels are direct decoded images with measured bpp labels; no interpolated images.

## One controlled follow-up

current one D update per G versus two D updates per G, same source model/shared plan,1000 G updates,rho0.05 and other losses unchanged. This proposed comparison has not been run in this audit; all other settings would remain matched.

## Files

`diagnostics/`: complete classification, probes, gradients, class distributions and historical update statistics.
`results/mcljcv/`: raw points, per-video, pooled FID, common grids, bands and decomposition.
`plots/`: curves and fixed first/middle/last source panels.
`audits/`: sources, exact state immutability, colors, overlaps and decode checks.
`file_index.json` and `restore_text_shards.py`: complete UTF8 shards, reconstruction and SHA verification.
