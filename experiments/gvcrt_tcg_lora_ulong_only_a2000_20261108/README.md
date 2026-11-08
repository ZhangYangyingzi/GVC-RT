# ULong-only A2000 LoRA continuation

This task restores the audited A2000 LoRA weights, AdamW moments, absolute optimizer step 2000 and four logical RNG streams. Only 59,392 LoRA parameters can update. TCG remains frozen at cumulative step 6500. The parent has historical UVG exposure; this continuation consumes ULong only.

Training gain is explicitly 1.0 and development/final gain is explicitly 0.5. The new immutable source-group/time-bin plan has an independent seed of 20261108, four ULong samples from four distinct source groups per update, and no old mixed-plan cursor. Native decoded window hashes are written separately on actual reads.

`protocol.json` freezes configuration and `source_manifest.json` seals code, sources, manifests and plan hashes. The persistent `runner.py` first checks three A2000 half-gain development endpoints, new MCL real I/P coding without quality scoring, restored optimizer/four-rank aggregation, individual loss gradients and runtime throughput. Disposable updates never advance formal progress.

Development validation is 96 points every 250 updates. Qualification requires DISTS ratios <=1 at QP0/5/9 and mean normalized FloLPIPS <1, relative to A2000 s0.5. Qualified improvement larger than 0.0005 resets patience; otherwise four complete valid checks stop training, with cap 3000 additional updates. Selection is frozen before test quality computation.

Each distinct final state has 225 points: previous 45 videos and 30 MCL-JCV sources, each at QP0/5/9. MCL uses the real I-frame bitstream, not the historical native-RGB bypass. Source FPS, BT709 limited-to-full conversion and native geometry are preserved. Results use 64-frame spatial aggregation, 63 temporal transitions and total pixel-SSE PSNR. Dataset/split curves remain separate.

Launch/resume (the execution agent runs this command):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONDONTWRITEBYTECODE=1 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u /Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_lora_ulong_only_a2000_20261108/experiments/gvcrt_tcg_lora_ulong_only_a2000_20261108/runner.py
```

The runner lock, process start identity, heartbeat, atomic latest checkpoint and committed receipts prevent duplicate ownership and update replay. GPU use is limited to physical 4–7. Cache, temporary files, code, checkpoints and results are local to this experiment. Publication uses an explicit allowlist; weights, media, streams and caches are never uploaded. `status.json` and `publication_status.json` distinguish execution from publication.
