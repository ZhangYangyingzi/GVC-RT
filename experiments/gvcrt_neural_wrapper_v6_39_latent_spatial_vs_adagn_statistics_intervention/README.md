# v6.39 read-only latent spatial vs AdaGN statistics intervention

This experiment holds the v6.34b trained generator fixed and swaps two inputs
from the same verified v6.37 RANS replay: the latent at `conv_in`, and the latent
used as the second argument of five `AdaptiveGroupNorm` modules. It performs no
training update and no new encoding. The source video, decoded codec feature,
frame QP, `q_recon`, and complete stream bytes are identical within each
video/QP comparison.

Run `prepare39.py`, then `run39.py --pilot`, followed by `run39.py`,
`aggregate39.py`, and `figures39.py` with the recorded gvc-rt Python. The
worker `intervene39.py` is resumable through per-task receipts and locks.
Historical v6.37 Python files are imported from their original directory for
reading; new writes go only to this directory.

The gvc-rt environment has no Matplotlib. Only `figures39.py`, which loads
finished CSV and PNG files without a model, is run with the server `python`
environment (Matplotlib 3.9.2, NumPy 1.26.4, Pillow 10.4.0). Inference and
metric extraction use the specified gvc-rt Python (PyTorch 2.6.0+cu124).

`tasks/`, `raw/`, `features/`, `artifacts/`, and `logs/` contain server-side
receipts, full path checks, features, fixed frames, and logs. The small
tables, audits, figures, and report are published to GitHub. Absolute binary
paths and SHA256 values are recorded in the manifests and receipts.
