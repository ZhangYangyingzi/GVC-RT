#!/usr/bin/env python3
import argparse
import json
import os

import torch

from core import ROOT, csv_write, quality_models
from eval_core import load_joint, matched_frames, run_stream


METHOD_CHECKPOINTS = {
    "original": None,
    "stage_a": "checkpoints/joint_warmup/step_2000.pt",
    "beta_low": "checkpoints/beta_low/step_5000.pt",
    "beta_mid": "checkpoints/beta_mid/step_5000.pt",
    "beta_high": "checkpoints/beta_high/step_5000.pt",
}


def tag(video):
    return f"{video['dataset']}_{int(video['video_id']):02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--tags", required=True)
    parser.add_argument("--methods", default=",".join(METHOD_CHECKPOINTS))
    parser.add_argument("--frames", type=int, default=64)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    manifest = json.loads((ROOT / "test_manifest.json").read_text())
    lookup = {tag(video): video for video in manifest["videos"]}
    quality = quality_models(device)
    methods = args.methods.split(",")
    unknown = set(methods) - set(METHOD_CHECKPOINTS)
    if unknown:
        raise RuntimeError(f"unknown methods: {sorted(unknown)}")
    joints = {}
    for method in methods:
        checkpoint = None if METHOD_CHECKPOINTS[method] is None else ROOT / METHOD_CHECKPOINTS[method]
        if checkpoint is not None and not checkpoint.exists():
            raise RuntimeError(f"missing required checkpoint: {checkpoint}")
        joints[method] = load_joint(checkpoint, device) if checkpoint is not None else None
    summaries, frame_rows = [], []
    for video_tag in args.tags.split(","):
        video = lookup[video_tag]
        frames = matched_frames(video_tag, args.frames)
        for method in methods:
            stage = "original" if method == "original" else ("A" if method == "stage_a" else "B")
            beta = method if method.startswith("beta_") else "none"
            checkpoint = "" if METHOD_CHECKPOINTS[method] is None else str(ROOT / METHOD_CHECKPOINTS[method])
            for qp in range(4):
                bitstream = ROOT / "bitstreams/test" / video_tag / f"{method}_qp{qp}.bin"
                save_root = None
                if qp in (1, 3):
                    save_root = ROOT / "parts/saved_frames" / video_tag / f"{method}_qp{qp}"
                summary, frame_data = run_stream(
                    frames, qp, joints[method], device, float(video["fps"]),
                    bitstream, save_root=save_root, quality=quality)
                summaries.append({
                    "dataset": video["dataset"], "video": video["name"],
                    "video_id": video["video_id"], "video_tag": video_tag,
                    "method": method if method == "original" else f"v2_{method}", "stage": stage, "beta": beta,
                    "checkpoint": checkpoint, "qp": qp,
                    "num_frames": args.frames, "fps": video["fps"], **summary,
                })
                frame_rows.extend({
                    "dataset": video["dataset"], "video": video["name"],
                    "video_id": video["video_id"], "video_tag": video_tag,
                    "method": method if method == "original" else f"v2_{method}", "stage": stage, "beta": beta,
                    "qp": qp, **row,
                } for row in frame_data)
                print(video_tag, method, qp, summary["bytes"], flush=True)
    csv_write(ROOT / "parts" / f"final_rd_gpu{args.gpu}.csv", summaries)
    csv_write(ROOT / "parts" / f"final_frames_gpu{args.gpu}.csv", frame_rows)
    (ROOT / "parts" / f"eval_gpu{args.gpu}_done.json").write_text(json.dumps({
        "status": "PASS", "tags": args.tags.split(","), "methods": methods,
        "streams": len(summaries), "frames": args.frames,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
