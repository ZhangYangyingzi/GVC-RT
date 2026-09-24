#!/usr/bin/env python3
"""Evaluate the selected V3 checkpoint on frozen U-Long test frames."""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
RETEST = ROOT.parent / "gvcrt_neural_wrapper_v2_retest_fixed_uvg"
V2 = ROOT.parent / "gvcrt_neural_wrapper_v2_joint"
sys.path.insert(0, str(V2))
from core import quality_models
from eval_core import load_joint, run_stream


def tag(video):
    return f"{video['dataset']}_{int(video['video_id']):02d}"


def load_frames(video_tag):
    base = RETEST / "source_frames_fixed" / video_tag
    frames = []
    for index in range(1, 65):
        image = Image.open(base / f"im{index}.png").convert("RGB")
        array = np.asarray(image, dtype=np.uint8).copy()
        frames.append(torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float() / 255)
    return frames


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(4, 5, 6, 7), required=True)
    parser.add_argument("--tags", required=True, help="Comma-separated frozen test tags")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    selected = json.loads((ROOT / "selected_checkpoint.json").read_text())
    if selected["stage"] != "stage_b" or selected["used_final_test"]:
        raise RuntimeError("checkpoint was not selected solely from validation")
    checkpoint = Path(selected["checkpoint"])
    joint = load_joint(checkpoint, device)
    checkpoint_digest = file_hash(checkpoint)
    quality = quality_models(device)
    test = json.loads((ROOT / "test_manifest.json").read_text())["videos"]
    lookup = {tag(video): video for video in test}
    tags = args.tags.split(",")
    if not set(tags) <= set(lookup):
        raise ValueError("tag outside frozen test set")
    output = ROOT / "parts" / f"final_v3_gpu{args.gpu}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output if output.exists() else os.devnull, newline="") as handle:
        complete = {(row["video_tag"], int(row["qp"])) for row in csv.DictReader(handle)}
    fields = None
    for video_tag in tags:
        video = lookup[video_tag]
        pending = [qp for qp in range(4) if (video_tag, qp) not in complete]
        if not pending:
            continue
        frames = load_frames(video_tag)
        for qp in pending:
            bitstream = ROOT / "bitstreams" / "final" / video_tag / f"v3_beta_low_qp{qp}.bin"
            save_root = (ROOT / "reconstruction_frames" / video_tag / f"v3_beta_low_qp{qp}") if qp in (1, 3) else None
            summary, _ = run_stream(frames, qp, joint, device, float(video["fps"]),
                                    bitstream, save_root=save_root, quality=quality)
            row = {"dataset": video["dataset"], "video": video["name"],
                   "video_id": video["video_id"], "video_tag": video_tag,
                   "method": "v3_beta_low", "stage": "stage_b",
                   "beta": "beta_low", "checkpoint": str(checkpoint),
                   "checkpoint_sha256": checkpoint_digest, "qp": qp,
                   "num_frames": 64, "fps": video["fps"], **summary}
            if fields is None:
                fields = list(row)
            with open(output, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            print(video_tag, qp, summary["bytes"], flush=True)


if __name__ == "__main__":
    main()
