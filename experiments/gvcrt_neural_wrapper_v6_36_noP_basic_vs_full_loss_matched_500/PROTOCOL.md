# v6.36 protocol

A single no-P branch starts from the original published model initialization, with a fresh matched AdamW optimizer, and performs exactly 500 updates on the first 500 rows of the sealed v6.34 shared plan. It uses the existing 64-frame causal window, native I0 bypass, P1–48 no-grad warmup, P49–63 supervision, detached references, and crop size 256.

The training target is LPIPS + weighted reconstruction L1 + inherited beta(q) rate term + proxy L1. DISTS, its spatial statistic, and FD are not computed during this branch. The existing full-loss no-P step-500 model and original model are reused after pointwise verification.

The fixed evaluation set is the v6.35 13-video manifest at QP 0/5/9. Newly trained B gets real RANS encode and independent decode. Existing original/F points are read-only aliases only after hashes and protocol fields verify.
