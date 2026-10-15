# Fixed mechanism protocol

The machine-readable preregistration is protocol.json. It was fixed before
observing formal results. The dataset/frame identities, source sampling, teacher
proxy and native-I real-RANS evaluator are inherited from v6.31a, v6.29 and v6.22.

P_alpha(x) = clamp(x + alpha*(P(x)-x), 0,1), alpha=0,.25,.5,.75,1,1.25.
Endpoints use exactly x and P(x), respectively, to avoid gratuitous floating-point
cancellation at the no-op controls. I0 bypasses the wrapper. Compression core,
P, bridge, generator and teacher are frozen. Main adapted model is v6.22
B1_reconstruction adaptation_step_1000; no checkpoint is updated.

Actual 64-frame bpp = 8*file_bytes/(64*1920*1080), including SPS/headers/I0.
P-frame bpp = sum(P-frame consumed bits)/(63*1920*1080).
Payload rates exclude headers and are explicitly separate.
Fresh independent decode verifies full consumption, actual QP, online/independent
reconstruction synchronization and unchanged modules. QP9 has the historical
higher-rate meaning; no traditional inverse-QP assumption is made.

Source-frame residual/frequency bands and generator-input/teacher distances
reuse v6.31a. Teacher always takes original GT, never the modified source.
Primary teacher target is fixed EMA Open-MAGVIT2 continuous pre-LFQ latent,
18×68×120, labeled paper_compatible_teacher_proxy. The actual generator-input
hook runs only in the third, fresh persisted-stream decode model instance.

Quantization semantics are established from actual executable source:
raw analysis-y is normalized by clamped predicted q_dec, centered at the actual
conditional mean in each of two masked groups, rounded, optionally forced to
zero when scale<=.12, and clipped to [-128,127]. The returned y_hat restores
conditional means and q_dec. Integer symbol changes are measured on centered
integer y_q, never by treating continuous generator input as transmitted tokens.
Rounding-boundary diagnostics use actual centered normalized residuals, with
active/non-saturated eligibility, threshold .05. Skip and saturation statistics
are separate. No half-integer rule is applied to raw unnormalized y.

Entropy diagnostics observe actual packed symbol/index values passed into native
RANS. Integer CDF bin mass gives in-support bits and escape-event mass; escape
bypass bits and hyperlatent-z bits are not assigned to those diagnostics. No
per-channel actual stream allocation is claimed. RGB prediction residual is
unavailable: conditional context is not a defined RGB/source-latent prediction.

Analysis uses same-video/same-QP controls, alpha trends, and 2000 video-cluster
bootstrap resamples (seed20261015). Rate bands are alpha0-control bpp tertiles
within dataset, fixed across alpha. Bootstrap means weight videos equally;
pooled-pair medians and video-cluster intervals are reported separately.
Single-video strata have degenerate cluster intervals, explicitly labeled.

Only B1_alpha100 versus Original supplies the main system-level matched-rate
comparison: per-video common measured bpp support, 100 uniform log-bpp points,
PCHIP, no extrapolation or duplicate-rate deletion. LPIPS, DISTS and FloLPIPS
are primary; PSNR/MS-SSIM are auxiliary. FID extraction is retained because it
is already part of the audited evaluator, but it does not drive this mechanism
conclusion or delay analysis with additional FID pooling.

CASE A requires consistent same-QP coding change, actual internal symbol change,
>=2 primary matched-rate metrics with negative video-cluster95% upper bounds,
and teacher-worse/rate-lower at >=50% of video/QP pairs. Otherwise changed coding
is CASE B; no coding effect is CASE C. Missing data are not CASE C. The
preregistered rule is conservative and never chooses a checkpoint or method.

This is not Strong No-P. No conclusion about insufficient baseline training is
excluded, and no optimal representation or incorrect teacher/supervision claim
is allowed. Source/rate/fidelity/quality associations are not causal channel
attribution.
