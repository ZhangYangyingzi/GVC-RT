# Fixed v6.40 protocol

基础损失与加入 DISTS 的对照；非官方完整训练目标。

See `protocol.json` for the fixed conditions, optimizer, seed, temporal semantics and evaluation. `plans/first100.json` is the immutable shared plan; its SHA256 and 50/50 and video×QP counts are in `plans/counts.json`. Each video contributes one historical 64-frame window and 50 distinct spatial crops, not 50 independent windows.

Step0 source SHA256: 18394bc0145e4bfc9eb18fb009fdb9dcc7f626536fe2367fb19b95e471f09074. Both branches restore the identical original tensors, buffers and full training RNG, with fresh AdamW. Diagnostic optimizer calls: zero. Smoke updates 1 and 2 count toward the exact 100 committed updates. Only step100 is evaluated.

18 independent decoding points use six complete audited real RANS streams. Native I, IP=-1, inherited padding/reset/color/metrics. Every point has 64 frame records and 63 real adjacent-frame transitions. Aggregate RGB MSE determines PSNR. No FID, seed search, extra updates or extra conditions.

O_original is freshly reevaluated, rather than relying on historical evaluation equivalence. Historical experiments, including v6.39, are read-only.
