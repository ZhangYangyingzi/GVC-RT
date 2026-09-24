#!/usr/bin/env python3
"""Start Stage B on GPU4 only after Stage A has completed successfully."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
done = ROOT / "parts/train_stage_a_done.json"
checkpoint = ROOT / "checkpoints/stage_a/step_5000.pt"
while True:
    if done.is_file() and checkpoint.is_file():
        result = json.loads(done.read_text())
        if result["status"] == "PASS" and result["steps"] == 5000:
            break
    time.sleep(30)

command = [sys.executable, "-B", str(ROOT / "train.py"), "--gpu", "4", "--stage", "stage_b"]
with open(ROOT / "logs/train_stage_b.log", "a") as log:
    subprocess.run(command, cwd=ROOT.parents[1], stdout=log, stderr=subprocess.STDOUT, check=True)
