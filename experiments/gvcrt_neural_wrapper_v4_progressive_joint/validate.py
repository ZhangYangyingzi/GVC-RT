#!/usr/bin/env python3
"""Run resumable real-RANS validation on the frozen V3 split."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "gvcrt_neural_wrapper_v2_joint"
sys.path.insert(0, str(V2))
from core import quality_models
sys.path.insert(0, str(ROOT))
from eval_core import load_joint, run_stream, video_frames

FIELDS = ("stage", "step", "video", "video_index", "qp", "real_bytes", "kbps", "bpp",
          "LPIPS", "DISTS", "PSNR", "MS_SSIM", "SSIM", "bitstream_path",
          "bitstream_sha256", "bytes_consumed", "decode_status", "state_sync_pass",
          "compression_hash_before", "compression_hash_after", "checkpoint", "checkpoint_sha256",
          "reconstruction_sha256", "independent_decode_pass", "num_frames")
STEPS = {"original": (0,), "v3": (20000,), "beta_low": (0, 1000, 2000, 5000, 10000),
         "beta_mid": (0, 1000, 2000, 5000, 10000),
         "beta_high": (0, 1000, 2000, 5000, 10000)}


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(4, 5, 6, 7), required=True)
    parser.add_argument("--stage", choices=tuple(STEPS), required=True)
    parser.add_argument("--steps", help="Comma-separated checkpoint steps")
    parser.add_argument("--video-indices", default="0,1,2,3,4,5")
    parser.add_argument("--wait-checkpoints", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    cfg = json.loads((ROOT / "config.json").read_text())
    videos = json.loads((ROOT / "validation_manifest.json").read_text())["videos"]
    steps = tuple(int(x) for x in args.steps.split(",")) if args.steps else STEPS[args.stage]
    indices = tuple(int(x) for x in args.video_indices.split(","))
    if not set(steps) <= set(STEPS[args.stage]) or not set(indices) <= set(range(len(videos))):
        raise ValueError("invalid step or video index")
    output = ROOT / "parts" / f"checkpoint_validation_{args.stage}_gpu{args.gpu}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = open(ROOT/"parts"/f"validation_{args.stage}.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    completed = set()
    for shard in sorted((ROOT / "parts").glob(f"checkpoint_validation_{args.stage}_gpu*.csv")):
        with open(shard, newline="") as handle:
            for row in csv.DictReader(handle):
                if row["decode_status"] != "PASS":
                    raise RuntimeError(f"existing validation row failed: {shard}")
                key = (row["stage"], int(row["step"]), int(row["video_index"]), int(row["qp"]))
                if key in completed:
                    raise RuntimeError(f"duplicate validation row: {key}")
                completed.add(key)
    quality = quality_models(device)
    for step in steps:
        checkpoint = ROOT / "checkpoints" / args.stage / f"step_{step:04d}.pt"
        if args.stage == "v3":
            checkpoint = Path(cfg["initial_checkpoint"])
        if args.stage != "original":
            previous_size = -1
            while True:
                current_size = checkpoint.stat().st_size if checkpoint.is_file() else 0
                if current_size > 100_000_000 and current_size == previous_size:
                    break
                if not args.wait_checkpoints and not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                print(f"waiting for {checkpoint}", flush=True)
                previous_size = current_size
                time.sleep(10 if args.wait_checkpoints else 1)
            joint = load_joint(checkpoint, device)
            checkpoint_digest = file_hash(checkpoint)
        else:
            joint, checkpoint_digest = None, ""
        for index in indices:
            video = videos[index]
            pending = [qp for qp in cfg["validation"]["qps"]
                       if (args.stage, step, index, qp) not in completed]
            if not pending:
                continue
            frames = video_frames(video["path"], cfg["validation"]["frames"])
            fps = float(Fraction(video["fps"]))
            for qp in pending:
                stream = ROOT / "bitstreams" / "validation" / args.stage / f"step_{step:04d}" / f"video_{index}_qp{qp}.bin"
                summary, _ = run_stream(frames, qp, joint, device, fps, stream, quality=quality)
                row = {"stage": args.stage, "step": step, "video": video["filename"],
                       "video_index": index, "qp": qp, "real_bytes": summary["bytes"],
                       "kbps": summary["kbps"], "bpp": summary["bpp"],
                       "LPIPS": summary["LPIPS"], "DISTS": summary["DISTS"],
                       "PSNR": summary["PSNR"], "MS_SSIM": summary["MS_SSIM"],
                       "SSIM": summary["SSIM"], "bitstream_path": str(stream),
                       "bitstream_sha256": summary["bitstream_sha256"],
                       "bytes_consumed": summary["bytes_consumed"],
                       "decode_status": summary["decode_status"],
                       "state_sync_pass": summary["state_sync_pass"],
                       "compression_hash_before": summary["compression_hash_before"],
                       "compression_hash_after": summary["compression_hash_after"],
                       "checkpoint": str(checkpoint) if joint else "",
                       "checkpoint_sha256": checkpoint_digest,
                       "reconstruction_sha256": summary["reconstruction_sha256"],
                       "independent_decode_pass": summary["independent_decode_pass"],
                       "num_frames": len(frames)}
                with open(output, "a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS)
                    if handle.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)
                completed.add((args.stage, step, index, qp))
                print(args.stage, step, index, qp, summary["bytes"], flush=True)


if __name__ == "__main__":
    main()
