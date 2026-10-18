# Evaluation implementation and saved data

The reused evaluator is `experiments/gvcrt_parallel_v5a3_v60/engine.py`,
`Runtime.run`; dependencies and their frozen source copies are indexed in
`manifests/dependency_sources.json`. Inputs are the fixed canonical RGB clips
in `manifests/videos.json`. All branches evaluate 16 frames at 1920x1080,
after removing the bottom padding from the codec's 1920x1088 canvas.

LPIPS uses the frozen AlexNet implementation from `lpips.LPIPS(net="alex")`,
with `normalize=True` on RGB values in [0,1]. DISTS uses the frozen
`DISTS_pytorch.DISTS` implementation on the same floating-point RGB crop.
Both are evaluated per frame and arithmetically averaged over 16 frames.
`core.quality_models` is in `gvcrt_neural_wrapper_v2_joint/core.py`;
the frame evaluator is inherited through `gvcrt_neural_wrapper_v4_progressive_joint/eval_core.py`.

FloLPIPS uses the inherited `flolpips_compat` model loader referenced by
`gvcrt_neural_wrapper_v4_1_high_convergence_metrics/metric_runtime.py`.
For each adjacent pair the evaluator computes reference flow minus decoded
flow, checks its magnitude is positive and finite, and passes it to the
perceptual model with `normalize=True`. The sequence score averages the
15 saved transition scores. No transition subsampling is used.

The runtime exports 2048-dimensional Inception features using the inherited
official FID Inception loader (`training32.official_model`), but does not
calculate FID for this 16-frame sample. All FID and delta-FID fields remain
null with `NOT_COMPUTED_INSUFFICIENT_SAMPLES`. Feature extraction does not
make these small clips a valid FID benchmark.

Actual total bits are eight times the persisted experimental stream size.
They include native SPS/IP headers, RANS payloads, 10 fixed extension bytes
and 11 bytes per signalled decision. Original strategy A uses the native
stream without this extension. bpp divides bits by 16*1920*1080; kbps uses
the inherited 30 fps accounting convention. Frame-level bit counts sum to
the entire stream, including control bytes charged to frame zero. Baseline
tables retain both wrapped and native rates; allocation budgets use native
rates. Nonzero rate differences are excluded from equal-rate comparisons.

Raw spatial scores are in `metrics/per_frame/`; temporal scores are copied
to `metrics/per_transition/` during validation. `metrics/timing/` stores
CUDA-synchronized codec-call timings, including read-only tensor observers.
Model loading, disk I/O, video export and metric inference are excluded from
those codec timings and included in the separately recorded task wall time.
Metric weight paths and hashes are recorded in `manifests/metric_weights.json`.

`src/verify_saved_controls.py` independently checks each saved envelope,
quantization-step rule, original/modified ROI symbol record, frame-bit sum,
and LPIPS/DISTS aggregation. Finalization runs it before publication.
