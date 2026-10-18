# GVC-RT recoverability oracle feasibility

Fixed original GVC-RT weights. No training, P module or weight updates. Four inherited evaluation identities;16 native-resolution frames;QP2/7. Audit and executable configuration are under audit/ and configs/.

Run: `/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B src/driver.py` from this directory (GPU4/5/6/7 only). Individual commands and stdout/stderr are saved under logs/ and manifests/. Completed atomic receipts are resumable.

Baseline: baseline/. Local interventions: oracle/local_sensitivity/. Allocation trials: oracle/rate_allocation/. Raw tensors, complete streams and lossless sample reconstructions: artifacts/ (not Git; indexed with server paths and SHA256). Metrics are computed from floating-point native reconstructions before video export. FID is not computed for these small pools. Experimental block control must pass decoder/causality correctness before allocation is enabled. Exact actual-rate matches are labelled separately from unmatched trials. See run_status.json for status; no population or optimality conclusions are made.
