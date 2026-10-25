# v4 quantization granularity ablation

Inference only: five modes × four videos × QP2/7 = 40 new 64-frame encodings,
including the five UVal0/QP2 preflight tasks. No optimizer, extra QP sweep,
alpha search, epoch, or checkpoint update is performed.

Run `driver.py` with the indexed gvc-rt Python on the host with NVIDIA access.
It checks immutable source/weight SHA256, completes every preflight mode,
then runs the remaining tasks on unoccupied physical GPUs4/5/6/7. A process
lock prevents duplicate runners. A task is skipped on resume only when its
code/config signature and every stream/table/state/map artifact SHA match.
Interrupted tasks are rerun; completed tasks are retained.

`finalize.py` performs the full artifact and mathematical audit, produces
tables and REPORT.md, and generates plots and fixed frame15 previews.
`publish.py` commits only this directory on
`experiment/gvcrt-quant-v4-granularity-ablation`, pushes without force,
verifies the remote SHA, and finally writes local COMPLETE.json. Logs,
full streams, every-P-frame map arrays and COMPLETE.json remain on the server.
Their actual paths and SHA256 are indexed by artifact_index.json.

The specified A reference is v1 A step300, using original codec quantization
and its trained G. It is not the untouched official generator checkpoint.
All other modes use v2 H11 step300 G/H, with original frozen codec probabilities.
The H11 multiplier has shape B×1×H×W, broadcast over the 128 latent channels.
Consequently FRAME and CHANNEL are identical by construction. They are still
encoded independently and their streams, full states, and metrics are checked.
These weights cannot identify an independent channel-scale contribution.

See PROTOCOL.md, preflight.json, cross_mode_audit.json, delivery_audit.json,
evaluation.csv, matched-rate_comparisons.csv, observed_dominance.csv and REPORT.md.
