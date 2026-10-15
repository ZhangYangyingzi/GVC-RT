# GLC precompression transfer v1 A/B/C

Execution in progress. A is the original GLC model; B and C train only a new identity-initialized RGB/latent residual module, respectively. Historical GVC wrapper weights are not used.

Training: balanced U-Long/UVG, 1000 updates each for B/C, 64-frame causal clips, 256 crops, supervised P49–63. Independent tests: U-Long, UVG holdout, HEVC-B, MCL-JCV. UVG validation is separately reported; MCL29 is a sensitivity subset.

The local GLC checkout includes a non-official real-rANS extension. Its availability and round-trip behavior must pass preflight before reporting actual bitstream rates. No estimated probability rate will be labeled real bitrate.

Check status.json for actual execution status. No training is allowed before weight/gradient/data checks and an actual initial GitHub push.
