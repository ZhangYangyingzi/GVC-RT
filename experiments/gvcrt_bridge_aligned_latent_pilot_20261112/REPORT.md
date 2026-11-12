# Bridge aligned latent pilot: factual execution record

Completed one paired seed (20261112): 500 matcher pretraining updates and 300 committed updates each for S/U/T. No full-dataset training, second seed, architecture/loss search, convergence claim, efficacy claim or test135 run.

B300 composition: Original → pilot → B1500 TCG → A2000 LoRA → B300 last (52 incremental keys). LoRA gain 0.5. Composite parameter SHA256 `743218b24af1efe631d71e3f0626a6ffdf423cef2175f33fc41470fb064c8552`. All codec/entropy/Bridge/de-tokenizer/TCG/LoRA parameters frozen; per-group final hashes are in `checkpoints/*/freeze_audit.json`.

Training coverage: 32 distinct approved ULong source groups, 128 unique nonoverlapping 16-frame temporal windows, 128 fixed original-RGB 256×256 spatial crops, 384 QP instances (external0/5/9, original actual-QP schedule retained). Original frames were cropped before frozen codec/Bridge; no whole-frame latent crop equivalence claim.

Training consumes 1200 fixed 4-frame segments per group (4800 supervised frame occurrences), reusing 384 fixed window×QP instances. Per-instance/per-window repeat counts and exact spatial/temporal identities are in `metrics/training_coverage.csv` and `training_plan.json`. Four segments are accumulated equally per update. Candidate-specific causal warmup is replayed from the window I-frame reset, TCG gradients are kept inside each segment, and all state is detached between segments.

Only the new aligner (U/T), reliability head and residual adapter train. S retains B300 TCG and has the same fusion capacity; S has no new historical content, U retains matching information while forcing latent warp displacement to zero. Parameter counts: {"S_current": 169459, "U_unaligned": 489017, "T_aligned": 489017}. The matcher/RMS/QP statistics and optimizer settings are fixed in protocol.json.

Loss: LPIPS_Alex(normalize=True)+0.5 DISTS+0.1383589248954108 RGB_L1 + 4.63349060748*(GT-flow-masked motion-compensated RGB change error +0.5 signed /8 Sobel change error); U/T additionally use0.1 multiscale pseudo-flow Charbonnier. This is the verified existing baseline spatial objective, not a claim of an unchanged original-paper loss. Lambda is0.1 times the median reconstructed-RGB output gradient ratio on eight fixed train segments. GT RAFT flow is a pseudo-label, never inference input.

All candidate and B300 reconstruction/metrics/new modules use FP32 PyTorch reference operators; codec keeps audited FP16. Decoder and perceptual parameters are frozen but their input gradients are preserved. Zero-output branches matched B300 bitwise across4 synthetic and7 real consecutive P frames. Correlation direction, fractional sampling, displacement units, all-invalid complete fallback, original-only history cache and cross-frame TCG gradients passed. Candidate reconstruction leaves codec DPB untouched. Grid-sample CUDA backward uses deterministic warn_only; seed pairing does not imply bitwise deterministic backward.

Evaluation scope: four predeclared ULong development clips, each16 consecutive frames from a reset I frame, full native1920×1080 at source fps; QP5 at0/50/100/200/300. At300, externalQP0/5/9 additionally cover the same four ULong sources and one preexisting UVG-validation Bosphorus source (native120fps). These are short clips, not full original videos. No verified legal HEVC/MCL development split was available without using reserved/final lists; those curves are absent. Selection used existing GT content categories, never candidate errors; existing development use does not make these an independent final test.

|Dataset|Model|step|QP|sources|PSNR|LPIPS|DISTS|FloLPIPS|MC RGB|MC Sobel|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ulong|B300|0|5|4|28.117043|0.097233441|0.058406011|0.085535393|0.010369271|0.0034327722|
|ulong|Original|0|5|4|27.951916|0.10431186|0.06712845|0.091298987|0.010454828|0.0034511378|
|ulong|S_current|0|5|4|28.117043|0.097233441|0.058406011|0.085535393|0.010369271|0.0034327722|
|ulong|S_current|50|5|4|28.261718|0.09676|0.057724839|0.08494612|0.010284022|0.0033952827|
|ulong|S_current|100|5|4|28.295208|0.096759401|0.057498834|0.084952858|0.010252608|0.003380641|
|ulong|S_current|200|5|4|28.301955|0.097938233|0.057531182|0.086310509|0.010224459|0.0033511908|
|ulong|S_current|300|5|4|28.333917|0.098221703|0.057579989|0.086608874|0.010229823|0.0033487042|
|ulong|T_aligned|0|5|4|28.117043|0.097233441|0.058406011|0.085535393|0.010369271|0.0034327722|
|ulong|T_aligned|50|5|4|28.327442|0.09660114|0.057560297|0.084690746|0.010219226|0.0033714057|
|ulong|T_aligned|100|5|4|28.410221|0.096600204|0.057362659|0.084690243|0.010146215|0.0033427433|
|ulong|T_aligned|200|5|4|28.401264|0.097566859|0.057728586|0.085864198|0.010093817|0.0033156902|
|ulong|T_aligned|300|5|4|28.411914|0.097766788|0.057685877|0.086154128|0.01009197|0.0033102582|
|ulong|U_unaligned|0|5|4|28.117043|0.097233441|0.058406011|0.085535393|0.010369271|0.0034327722|
|ulong|U_unaligned|50|5|4|28.328309|0.096603199|0.057559219|0.084687587|0.010213835|0.0033697105|
|ulong|U_unaligned|100|5|4|28.392451|0.096648822|0.057415616|0.084744592|0.010145863|0.0033438199|
|ulong|U_unaligned|200|5|4|28.37241|0.097567963|0.057743637|0.085885391|0.010096838|0.0033194808|
|ulong|U_unaligned|300|5|4|28.381282|0.097769368|0.057701146|0.08617573|0.010095609|0.003314577|
|uvg|B300|0|5|1|29.50991|0.094610721|0.06663046|0.075121123|0.012729602|0.0061547151|
|uvg|Original|0|5|1|29.529755|0.096651328|0.068878278|0.077507984|0.012897794|0.0062430062|
|uvg|S_current|300|5|1|29.57533|0.093155435|0.064424768|0.073228349|0.012490365|0.0059987273|
|uvg|T_aligned|300|5|1|29.635138|0.093234054|0.064363573|0.073334199|0.01240226|0.005944681|
|uvg|U_unaligned|300|5|1|29.613143|0.093211479|0.064373322|0.073316415|0.012422245|0.0059590432|

All unrounded per-video/step/QP values, per-frame metrics and signed paired differences are in metrics/. Dataset R-D curves use identical sequence sets within each dataset and preserve Original/B300; the three observed QPs are connected without extrapolation. No cross-dataset averaged curve or full-rate-range conclusion is made.

|ULong QP5, step300|ΔPSNR vs B300|ΔLPIPS|ΔDISTS|ΔFloLPIPS|ΔMC RGB|ΔMC Sobel|
|---|---:|---:|---:|---:|---:|---:|
|S_current|0.21687376|0.00098826142|-0.00082602259|0.0010734811|-0.00013944794|-8.4067978e-05|
|U_unaligned|0.2642394|0.00053592667|-0.00070486497|0.00064033698|-0.00027366147|-0.00011819523|
|T_aligned|0.29487124|0.00053334667|-0.000720134|0.00061873465|-0.00027730097|-0.00012251397|

Matcher pretraining is counted separately. Pseudo-label EPE samples are fixed four train pairs and one pair per ULong dev source at local frame8; they are not dense flow ground truth or full-clip average EPE. Raw0/100/250/500 values are in metrics/aligner_*.csv.

|Matcher step|train EPE8|dev EPE8|
|---:|---:|---:|
|0|0.7998676|1.3998542|
|100|0.76566864|1.3196616|
|250|0.75860924|1.2889218|
|500|0.74784292|1.2580036|

Mechanism statistics (q quantiles/near0/1, displacement, valid history fraction, delta RMS and per-channel latent mean/std drift) are observations, not calibrated reliability or proof of useful history. Invalid/newly exposed-region error uses the frozen teacher consistency mask as a proxy, not true occlusion labels. FloLPIPS uses the audited official PWC+Alex implementation; MC residuals use the fixed GT-RAFT protocol.

Extra persistent state at this native padded shape: 34010880 bytes for one raw f and one uncorrected Bridge latent plus metadata. No corrected latent is stored as future history. Runtime breakdown and actual RAM-bitstream end-to-end decode definition are in audits/runtime_profile.json.

|Pipeline|codec ms/frame|external complete reconstruction ms/frame|end-to-end ms/frame|
|---|---:|---:|---:|
|B300|26.6936|123.818|150.512|
|T300|26.3507|158.158|184.509|

End-to-end timing includes RAM bitstream parsing, entropy/feature decoding, Bridge, candidate and complete de-tokenizer (batch1), including I-frame handling. P-frame codec DPB stores original features with frame=None; bitwise feature identity with independent Original decode was checked. Candidate RGB/latent never enters DPB. Component branch latency alone is not decode speed; this FP32 reference is not an optimized deployment speed claim.

Timing is separated: cache construction and teacher seconds are recorded per cache; training update seconds exclude cache generation/checkpoint serialization/diagnostics/evaluation; evaluation has its own receipts. Per-group updates and measured durations are in metrics/execution_times.json.

Implementation exceptions: initial Original snapshot omitted protocol-verified C++ sources (stopped with0 caches); inherited restore reset deterministic warn_only (stopped with0 optimizer updates); a coarse mask had no valid positions at attempted pretrain update50 (49 committed, resumed after safe empty-set reduction). Full all-invalid fallback metadata was completed before joint update0. These are logged in audits/implementation_corrections.json; no quality-driven architecture/loss change was made.

Limitations: one seed,32 training source groups,128 fixed windows,16-frame development clips, only three QPs for the final display, one legal UVG validation source, and no HEVC/MCL development curves. The two visual diagnostics independently roll through64 frames; quantitative checkpoint results remain16-frame pilot clips.300 updates is a cap, not convergence. No final135 test, human temporal blind study or significance test was performed.

Locations: protocol.json; source_manifest.json; data_manifest.json; training_plan.json; additional_development_manifest.json; audits/; metrics/; figures/; VIDEO_INDEX.md; REPRODUCE.md. Large caches/weights/videos are local and listed in large_artifacts.json; ordinary Git upload excludes them.

Runtime component means (FP32, synchronized, native padded input): matching=41.1673 ms, warp=2.11464 ms, fusion=2.51422 ms, adapter_only=0.547506 ms, reliability_only=0.747306 ms. Fusion includes reliability, adapter, scale restoration and final residual addition; the separately measured parts do not include the full decoder.

Publication: branch `codex/bridge-aligned-latent-pilot-20261112`; results commit [9c67059eb5ea7ed4eb5057f9d00b2117676e55a8](https://github.com/ZhangYangyingzi/GVC-RT/commit/9c67059eb5ea7ed4eb5057f9d00b2117676e55a8). Remote branch SHA was read back and matched this results commit. Large caches, weights and FFV1/MP4 media were not uploaded; their paths, sizes and SHA256 values are in large_artifacts.json. Final publication metadata is recorded in a subsequent ordinary commit; no force push or main-branch change.
