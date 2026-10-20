# Objective and module control

All B/G/BG updates use the exact copied v6.37 `basic_crop` LPIPS+DISTS fused VJP and separately scaled reconstruction VJP. DISTS explicitly uses `require_grad=True`; quality network parameters remain frozen.

`LPIPS + 0.5 DISTS + 0.1383589248954108 reconstruction_L1 + beta_q(schedule_s1p0) rate_bpp + 0.01 proxy_L1`. Proxy is Identity (L1=0), rate is the inherited frozen-codec differentiable training estimate and can have no receiver gradient. Actual evaluation bpp comes exclusively from persisted full RANS bytes.

B updates only the real wrapper mlp bridge, G updates only the real recon-generation decoder, BG updates both. A frozen generator remains in the differentiable forward path so B can receive input gradients; it has no trainable parameters or optimizer state. Frozen module hashes and absence of gradients are asserted at every update. Native I/frozen codec reference semantics are inherited.

Fresh AdamW: bridge6e-6, generator1e-6, constant LR, betas0.9/0.999,eps1e-8,weight decay1e-4. Only trainable module groups are present in each optimizer. Clip combined gradients once to global norm1.

No P/FD/legacy-stat/Inception training forward, no extra branch, no additional seed, no training beyond100, no intermediate checkpoint evaluation. Comparisons are fixed: B/O,G/O,BG/O,BG/B,BG/G on identical real streams.
