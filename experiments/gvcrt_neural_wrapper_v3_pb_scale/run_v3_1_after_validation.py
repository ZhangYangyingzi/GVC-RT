#!/usr/bin/env python3
"""Finish V3.1 only after 50k training and all new validation points exist."""
import csv
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SYSTEM_PYTHON = "/data1/anaconda3_new/anaconda_program/bin/python"
GVC_PYTHON = "/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python"


def validation_count():
    keys = set()
    for gpu in (5, 6):
        path = ROOT / f"parts/v3_1_validation_gpu{gpu}.csv"
        if path.is_file():
            with open(path, newline="") as handle:
                keys.update((r["step"], r["video_index"], r["qp"]) for r in csv.DictReader(handle))
    return len(keys)


while True:
    done = ROOT / "parts/train_stage_b_done.json"
    if done.is_file():
        result = json.loads(done.read_text())
        if result.get("status") == "PASS" and result.get("steps") == 50000 and validation_count() == 36:
            break
    time.sleep(30)

with open(ROOT / "logs/v3_1_resume_audit.log", "a") as log:
    subprocess.run([GVC_PYTHON, "-B", str(ROOT / "v3_1_resume_audit.py")],
                   cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
with open(ROOT / "logs/v3_1_select.log", "a") as log:
    subprocess.run([SYSTEM_PYTHON, str(ROOT / "v3_1_select.py")],
                   cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)

videos = json.loads((ROOT / "test_manifest.json").read_text())["videos"]
tags = [f"fresh_ulong_{int(video['video_id']):02d}" for video in videos]
groups = {5: tags[:3], 6: tags[3:6], 7: tags[6:]}
processes = []
for gpu, subset in groups.items():
    log = open(ROOT / "logs" / f"v3_1_final_gpu{gpu}.log", "a")
    command = [GVC_PYTHON, "-B", str(ROOT / "final_evaluate.py"),
               "--gpu", str(gpu), "--tags", ",".join(subset),
               "--selection-file", "v3_1_selected_checkpoint.json",
               "--output-prefix", "v3_1_final", "--method", "v3_1_selected"]
    processes.append((gpu, subprocess.Popen(command, cwd=REPO, stdout=log,
                                           stderr=subprocess.STDOUT), log))
for gpu, process, log in processes:
    code = process.wait()
    log.close()
    if code:
        raise RuntimeError(f"V3.1 final GPU{gpu} evaluation failed: {code}")

with open(ROOT / "logs/v3_1_finalize.log", "a") as log:
    subprocess.run([SYSTEM_PYTHON, str(ROOT / "v3_1_finalize.py")],
                   cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
with open(ROOT / "v3_1_stdout.log", "w") as log:
    subprocess.run([SYSTEM_PYTHON, str(ROOT / "v3_1_print_status.py")],
                   cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
