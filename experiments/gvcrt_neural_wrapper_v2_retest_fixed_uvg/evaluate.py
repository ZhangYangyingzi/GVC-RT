#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V2 = REPO / "experiments/gvcrt_neural_wrapper_v2_joint"
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))
from core import csv_write, quality_models
from eval_core import load_joint, run_stream

METHODS = ("original", "stage_a", "beta_low", "beta_mid", "beta_high")
CHECKPOINTS = {
    "stage_a": V2 / "checkpoints/joint_warmup/step_2000.pt",
    "beta_low": V2 / "checkpoints/beta_low/step_5000.pt",
    "beta_mid": V2 / "checkpoints/beta_mid/step_5000.pt",
    "beta_high": V2 / "checkpoints/beta_high/step_5000.pt",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tag(video):
    return f"{video['dataset']}_{int(video['video_id']):02d}"


def load_frames(video):
    base = ROOT / "source_frames_fixed" / tag(video)
    frames = []
    for index in range(1, 65):
        image = Image.open(base / f"im{index}.png").convert("RGB")
        array = torch.from_numpy(__import__("numpy").asarray(image, dtype="uint8").copy())
        frames.append(array.permute(2, 0, 1).unsqueeze(0).float() / 255)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--tags", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    manifest = json.loads((ROOT / "test_manifest.json").read_text())
    lookup = {tag(video): video for video in manifest["videos"]}
    quality = quality_models(device)
    joints = {"original": None}
    for method in METHODS[1:]:
        checkpoint = CHECKPOINTS[method]
        joints[method] = load_joint(checkpoint, device)
    summaries, frames_out = [], []
    for video_tag in args.tags.split(","):
        video = lookup[video_tag]
        frames = load_frames(video)
        for method in METHODS:
            stage = "original" if method == "original" else ("stage_a" if method == "stage_a" else "rate")
            beta = "none" if method in ("original", "stage_a") else method
            checkpoint = "" if method == "original" else str(CHECKPOINTS[method])
            checkpoint_hash = "" if method == "original" else sha256(CHECKPOINTS[method])
            for qp in range(4):
                stream_path = ROOT / "bitstreams" / video_tag / f"{method}_qp{qp}.bin"
                save_root = None
                if qp in (1, 3):
                    save_root = ROOT / "reconstruction_frames" / video_tag / f"{method}_qp{qp}"
                    if method != "original":
                        save_root = ROOT / "reconstruction_frames" / video_tag / f"{method}_qp{qp}"
                summary, frame_rows = run_stream(
                    frames, qp, joints[method], device, float(video["fps"]),
                    stream_path, save_root=save_root, quality=quality)
                summaries.append({
                    "dataset": video["dataset"], "video": video["name"],
                    "video_id": video["video_id"], "video_tag": video_tag,
                    "test_group": video["test_group"], "method": method,
                    "stage": stage, "beta": beta, "checkpoint": checkpoint,
                    "checkpoint_sha256": checkpoint_hash, "qp": qp,
                    "num_frames": 64, "fps": video["fps"], **summary,
                })
                frames_out.extend({
                    "dataset": video["dataset"], "video": video["name"],
                    "video_id": video["video_id"], "video_tag": video_tag,
                    "test_group": video["test_group"], "method": method,
                    "stage": stage, "beta": beta, "qp": qp, **row,
                } for row in frame_rows)
                print(video_tag, method, qp, summary["bytes"], flush=True)
        # Preserve source frames separately for manual inspection.
        source_dir = ROOT / "source_only" / video_tag
        source_dir.mkdir(parents=True, exist_ok=True)
        for index in range(1, 65):
            src = ROOT / "source_frames_fixed" / video_tag / f"im{index}.png"
            dst = source_dir / f"frame_{index - 1:06d}.png"
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
    csv_write(ROOT / "parts" / f"final_rd_gpu{args.gpu}.csv", summaries)
    csv_write(ROOT / "parts" / f"final_frames_gpu{args.gpu}.csv", frames_out)
    (ROOT / "parts" / f"eval_gpu{args.gpu}_done.json").write_text(json.dumps({
        "status": "PASS", "tags": args.tags.split(","), "streams": len(summaries)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
