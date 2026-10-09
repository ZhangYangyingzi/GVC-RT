# v6.20 execution

Start/resume both branches, verification, evaluation, numerical outputs and publication:

```bash
cd /Huang_group/zyyz/Projects/GVC-RT
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u experiments/gvcrt_neural_wrapper_v6_20_native_i_reference_adaptation/run_pipeline.py
```

Status:

```bash
/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B experiments/gvcrt_neural_wrapper_v6_20_native_i_reference_adaptation/run_pipeline.py --status
cat experiments/gvcrt_neural_wrapper_v6_20_native_i_reference_adaptation/final_integrity.json
cat experiments/gvcrt_neural_wrapper_v6_20_native_i_reference_adaptation/publication_status.json
```

Both branches replay the exact v6.18 AB plan and cleaned 1020-video manifest. References are initialized once, from actual I reconstruction under no_grad; all 15 P references remain detached. Step0 is checked against a fresh baseline on the same GPU and v6.19 persisted points, HoneyBee/Jockey QP0/4. Fixed step1000 is used for the 800-point official evaluation. All bit accounting includes the I frame and SPS/header bytes. Relative metric change is distinct from BD-rate.

Model/optimizer/RNG checkpoints, full RANS streams, full images and feature caches remain local. Publication stages only this new directory in a fresh origin/main worktree, performs an ordinary HEAD:main push and verifies remote files.
