# Fixed v4 inference-only protocol

Read-only templates: v3 rate/spatial ablation and v2 skip/domain H11. Checkpoints
are the explicitly indexed v1 A step300 and v2 H11 step300. Dependencies and
weights have SHA256 manifests and are checked before and after execution.

A uses v1 A G, no H; IDENTITY uses H11 G and m=1; FRAME averages raw H output
over C/H/W; CHANNEL averages over H/W; SPATIAL retains the full raw H output.
The original H has one output channel; its map broadcasts to all 128 main-latent
channels, so FRAME and CHANNEL are the same operator on these weights. No new
channel-dependent H is synthesized or trained. Raw/variant arrays retain their
native singleton channel and full spatial dimensions; actual qmap/rho arrays
retain all128 channels.

All five modes use the v2 H11 coordinates: q_base=clamp_min(original_step,.5);
q_new=clamp_min(q_base.float()*variant,.5).to(original_dtype); rho=q_new/q_base.
First-pass mean/scale divide by rho. First-pass reconstructed u0 multiplies rho
before the original spatial prior, whose mean/scale divide by rho for pass2.
Latent reconstruction uses q_new. Effective multiplier means the actual native
dtype ratio rho, including q clamp/dtype effects; alpha is always1.0.

Mode is fixed shared encoder/decoder experiment configuration. No multiplier
map or per-video side information is transmitted, and no experimental header
is added. The decoder recomputes all maps from its common prior. The RANS coder,
original frozen probability weights, native I, compressed reference update rule,
IP=-1, color/size and original eight-frame QP schedule are unchanged. G output
does not replace the compressed reference feature. Actual QP offsets are
[0,2,0,1,0,1,0,1].

Videos and true frame indices are byte-copied from v2 evaluation_sources.json:
UVal0/UVal1/Jockey/ParkScene, each64 full causal frames at externalQP2/7.
All40 encodings are newly produced. Every mode first passes full UVal0/QP2;
these five tasks count in40. GPU4–7 queues wait behind existing compute jobs.
No validation selection or training is performed.

Each task durably saves native stream, 63 full map NPZ files, estimated y/z bits,
two-pass skip/clamp diagnostics, all64 encoder/decoder state hashes and metrics.
It releases the encoder before loading a fresh independent decoder. Every
decoder checks reference, P latent/qmap/rho and all multiplier maps exactly.
A and IDENTITY additionally decode through the original v2 A runtime and check
the exact original A stream SHA. This parity concerns compressed states and
bitstream; their RGB outputs can differ because their G weights differ.
All five native I payloads must match at each video/QP.

Metrics: global RGB-MSE PSNR,64-frame mean LPIPS/DISTS,63-transition FloLPIPS.
20fps kbps is explicitly a conversion. Actual RANS y/z P payload is combined;
estimated analytic y/z bits are separate. P-only gaps exclude I and headers.
The inherited native z likelihood uses p+1e-5, y uses a1e-9 floor. FID is
not_computed. Encoding timing includes map/diagnostic instrumentation and memory
includes metrics/native parity work, so this is not a throughput benchmark.

Measured-point dominance is reported before interpolation. Compare each pair
at low/mid/high points of their per-video observed rate overlap using both
linear-bpp and linear-log-bpp quality interpolation, with no extrapolation.
Two QPs cannot establish formal BD-rate, significance, or full-curve dominance.
Quality rankings are metric/sequence specific; report all favorable and
unfavorable differences. Generator and multiplier contributions are separated
by A/IDENTITY and IDENTITY/SPATIAL comparisons. An IDENTITY gain fraction is
descriptive only, can exceed1 or be negative, and is unavailable near zero.

Previews are fixed in advance: frame15,QP2,crop(y200,x400,256×256), all modes.
There is no quality-selected frame or checkpoint. Historical training exposure,
single checkpoint/seed, two-QP interpolation uncertainty, and degenerate
FRAME/CHANNEL are explicit limitations on causal/generalization conclusions.
