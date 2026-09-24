#!/usr/bin/env python3
"""Complete checkpoint selection, final test, and reports after validation."""
import csv
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SYSTEM_PYTHON = "/data1/anaconda3_new/anaconda_program/bin/python"
GVC_PYTHON = "/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python"


def row_count(stage):
    keys = set()
    for path in (ROOT / "parts").glob(f"checkpoint_validation_{stage}_gpu*.csv"):
        with open(path, newline="") as handle:
            keys.update((row["stage"], row["step"], row["video_index"], row["qp"])
                        for row in csv.DictReader(handle))
    return len(keys)


while True:
    done = ROOT / "parts/train_stage_b_done.json"
    if done.is_file() and json.loads(done.read_text()).get("status") == "PASS":
        counts = {stage: row_count(stage) for stage in ("original", "stage_a", "stage_b")}
        if counts == {"original": 12, "stage_a": 48, "stage_b": 72}:
            break
    time.sleep(30)

with open(ROOT / "logs/summarize_validation.log", "a") as log:
    subprocess.run([SYSTEM_PYTHON, str(ROOT / "summarize_validation.py")],
                   cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)

videos = json.loads((ROOT / "test_manifest.json").read_text())["videos"]
tags = [f"fresh_ulong_{int(video['video_id']):02d}" for video in videos]
groups = {5: tags[:3], 6: tags[3:6], 7: tags[6:]}
processes = []
for gpu, subset in groups.items():
    log = open(ROOT / "logs" / f"final_gpu{gpu}.log", "a")
    command = [GVC_PYTHON, "-B", str(ROOT / "final_evaluate.py"),
               "--gpu", str(gpu), "--tags", ",".join(subset)]
    processes.append((gpu, subprocess.Popen(command, cwd=REPO, stdout=log,
                                           stderr=subprocess.STDOUT), log))
for gpu, process, log in processes:
    code = process.wait()
    log.close()
    if code:
        raise RuntimeError(f"final GPU{gpu} evaluation failed with exit code {code}")

with open(ROOT / "logs/finalize.log", "a") as log:
    subprocess.run([SYSTEM_PYTHON, str(ROOT / "finalize.py")],
                   cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
