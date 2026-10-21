# v6.43 fixed G_basic frame-QP budget utility

[Protocol](PROTOCOL.md) · [Full fixed request](REQUEST.txt) · [Objective results](REPORT.md)

Zero optimizer updates. Two historical q5 baselines, 24 single-frame interventions, and the fixed-budget transfer candidates recorded in transfer_plan.json. Real streams are included. Large model files and input video sources remain at the server paths indexed by the manifests.

To resume: run queue43.py smoke, then queue43.py impulse, analyze43.py select, queue43.py transfer, in the project conda environment; run finish43.py with /data1/anaconda3_new/anaconda_program/bin/python (the existing CPU plotting environment). Task signatures, file hashes, locks and atomic writes protect completed results. Publication uses a separate worktree and only this experiment directory.
