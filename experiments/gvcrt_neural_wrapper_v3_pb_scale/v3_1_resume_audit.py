#!/usr/bin/env python3
"""Check optimizer and sampling continuity across the 20k resume boundary."""
import csv
import json
import random
import argparse
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--sample-only", action="store_true")
args = parser.parse_args()
old = torch.load(ROOT / "checkpoints/stage_b/step_20000.pt", map_location="cpu", weights_only=True)
new = None if args.sample_only else torch.load(ROOT / "checkpoints/stage_b/step_50000.pt", map_location="cpu", weights_only=True)
records = json.loads((ROOT / "train_manifest.json").read_text())["videos"]
with open(ROOT / "parts/training_stage_b.csv", newline="") as handle:
    first = next(row for row in csv.DictReader(handle) if int(row["step"]) == 20001)

rng = random.Random()
rng.setstate(old["sampling_rng_state"])
plan = None
for _ in range(20):
    record = rng.choice(records)
    capture = cv2.VideoCapture(record["path"])
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if count < 4 or width < 256 or height < 256:
        continue
    plan = {"video": record["filename"], "start": rng.randrange(count - 3),
            "crop_x": rng.randrange(width - 255), "crop_y": rng.randrange(height - 255)}
    break
if plan is None:
    raise RuntimeError("cannot reproduce first resumed sample")
qp = rng.randrange(4)
sampling_match = all(str(first[key]) == str(value) for key, value in plan.items()) and int(first["qp"]) == qp
if args.sample_only:
    print(json.dumps({"sampling_rng_state_restored": sampling_match, "expected": {**plan, "qp": qp},
                      "observed": {key: first[key] for key in (*plan, "qp")}}, indent=2))
    if not sampling_match:
        raise RuntimeError("first resumed sample does not match saved RNG state")
    raise SystemExit(0)

def steps(payload):
    return {int(value["step"].item()) for value in payload["optimizer"]["state"].values()
            if "step" in value}

old_states = old["optimizer"]["state"]
new_states = new["optimizer"]["state"]
audit = {
    "resume_checkpoint": str(ROOT / "checkpoints/stage_b/step_20000.pt"),
    "resume_step": old["step"], "final_step": new["step"],
    "optimizer_parameter_states_at_20k": len(old_states),
    "optimizer_parameter_states_at_50k": len(new_states),
    "optimizer_steps_at_20k": sorted(steps(old)),
    "optimizer_steps_at_50k": sorted(steps(new)),
    "optimizer_state_restored": len(old_states) > 0 and len(new_states) == len(old_states)
                                and steps(old) == {20000} and steps(new) == {50000},
    "sampling_rng_state_restored": sampling_match,
    "torch_rng_state_saved": old["torch_rng_state"].numel() > 0,
    "cuda_rng_state_saved": old["cuda_rng_state"].numel() > 0,
}
audit["status"] = "PASS" if (audit["resume_step"] == 20000 and audit["final_step"] == 50000
    and all(value for value in audit.values() if isinstance(value, bool))) else "FAIL"
(ROOT / "v3_1_resume_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
if audit["status"] != "PASS":
    raise RuntimeError(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
