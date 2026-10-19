# RUNBOOK

Run training: `python basic_train.py --gpu 6`

After the 500-step inference export is saved, run the 39 tasks in `task_manifest.json` via `evaluate35.py --task '<json task>' --gpu <4|5|6|7>`, then run `aggregate36.py`, `finalize36.py`, and the scoped publisher. All outputs are rooted in this directory.
