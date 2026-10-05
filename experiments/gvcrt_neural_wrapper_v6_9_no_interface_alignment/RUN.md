Run or resume from the GVC-RT repository directory:

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_9_no_interface_alignment/run_pipeline.py --launch
```

The pipeline uses GPU 4, 5, 6, 7, allows multiple processes on a card subject
to memory reservations, and runs at most four CPU FID workers. Exact worker
commands, working directories, log paths and launch times are recorded in
`logs/commands.jsonl`. Runtime status is stored in `pipeline_status.json`.

Matched branches:

| New branch | V6.8b reference branch | Generator learning rate |
| --- | --- | --- |
| N50_no_interface | G50_interface_preserve | 2.5e-7 |
| N25_no_interface | G25_interface_preserve | 1.25e-7 |

Both branches restore the identical B1000 parameter tensors, AdamW state and
RNG states. Each runs steps 1001–1500 using the original 500 training plans,
with checkpoints at steps 1250 and 1500. The wrapper, bridge and generator
are trainable, and the compression core is frozen. Training calls the
existing V6.6 `L_B` binder directly; no teacher or interface loss is computed.

Formal evaluation preserves the 31 videos, external QPs 0–9, actual QP
schedule, real-RANS encode/decode, crops, frame pools and metric implementations
of V6.8b. Dataset FID uses the same GT caches and 20 bootstrap draws. The
separate Vimeo held-out cohort uses 32 clips and external QPs 0, 4, 9.

Artifacts:

- `config.json`, `evaluation/config.json`, `branches/*/training_config.json`
- `branches/*/checkpoints/step_1250.pt`, `step_1500.pt`, checkpoint hash files
- `training_logs/*.jsonl`, `training_logs/*.csv`, `logs/*.log`
- `parts/<dataset>/<method>/video_*_qp*.json` and per-frame CSV files
- `bitstreams/`, `features/`, `heldout_bitstreams/`, `heldout_features/`
- `parts/fid/`, raw bootstrap CSV files
- `results/<dataset>/raw_rd.csv` and `results/<dataset>/per_video/*.csv`
- `results/ablation_raw_rd.csv`, `baseline_raw_rd.csv`, `dataset_macro_rd.csv`
- `results/fid_raw.csv`, `fid_bootstrap.csv`, `fid_bootstrap_summary.csv`
- `results/vimeo_heldout_raw.csv`
- `results/comparison_*_same_qp.csv`: paired raw values at the same external QP
- `audits/`: source, protocol, disposable gradient check and GPU schedule records
- `final_integrity.json`: completeness and artifact validation status

The V6.8b final-report failure is recorded in `audits/baseline_reuse.json`.
Reuse is gated on individual baseline artifact validation, not that failed
report status. Previous experiment artifacts are read-only inputs.
