#!/usr/bin/env python3
import argparse
import json
import os

import torch

from core import ROOT, REPO, csv_write, quality_models
from eval_core import load_joint, run_stream, video_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--label", choices=("joint_warmup", "beta_low", "beta_mid", "beta_high"), required=True)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--steps", default=None,
                        help="Comma-separated override used only for smoke testing.")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    config = json.loads((ROOT / "config.json").read_text())
    stage = "A" if args.label == "joint_warmup" else "B"
    beta = 0.0 if stage == "A" else float(config["beta_weights"][args.label])
    default_steps = config["validation"]["stage_a_steps" if stage == "A" else "stage_b_steps"]
    steps = [int(value) for value in args.steps.split(",")] if args.steps else default_steps
    frame_count = args.frames or int(config["validation"]["frames"])
    split = json.loads((REPO / "expericent_generation_input/expericent_generator_aware_latent_distortion_v11/data_split.json").read_text())
    video = split["splits"]["validation"][0]
    frames = video_frames(video["local_path"], frame_count)
    fraction = video["probe"]["avg_frame_rate"].split("/")
    fps = float(fraction[0]) / float(fraction[1])
    quality = quality_models(device)
    rows = []
    for step in steps:
        checkpoint = ROOT / "checkpoints" / args.label / f"step_{step:04d}.pt"
        if not checkpoint.exists():
            raise RuntimeError(f"missing required checkpoint: {checkpoint}")
        joint = load_joint(checkpoint, device)
        for qp in config["validation"]["qps"]:
            bitstream = ROOT / "bitstreams/validation" / args.label / f"step_{step:04d}_qp{qp}.bin"
            summary, _ = run_stream(frames, qp, joint, device, fps, bitstream, quality=quality)
            rows.append({"stage": stage, "beta": args.label if stage == "B" else "none",
                         "beta_value": beta, "step": step,
                         "video": video.get("name", video["filename"]),
                         "qp": qp, **summary})
            print(args.label, step, qp, summary["bytes"], flush=True)
    output = ROOT / "parts" / f"checkpoint_rans_{args.label}.csv"
    csv_write(output, rows)
    print(json.dumps({"status": "PASS", "label": args.label, "rows": len(rows),
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
