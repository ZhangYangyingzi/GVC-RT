#!/usr/bin/env python3
import argparse
import csv
import hashlib
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
from eval_core import load_joint, run_stream, video_frames

FIELDS = ("stage", "step", "video", "video_index", "qp", "real_bytes", "kbps", "bpp",
          "LPIPS", "DISTS", "PSNR", "MS_SSIM", "SSIM", "bitstream_path",
          "bitstream_sha256", "bytes_consumed", "decode_status", "state_sync_pass",
          "compression_hash_before", "compression_hash_after", "checkpoint", "checkpoint_sha256")


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(4, 5, 6, 7), required=True)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--video-indices", default="0,1,2,3,4,5")
    parser.add_argument("--wait-checkpoints", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    steps = tuple(int(x) for x in args.steps.split(","))
    indices = tuple(int(x) for x in args.video_indices.split(","))
    videos = __import__("json").loads((ROOT / "validation_manifest.json").read_text())["videos"]
    output = ROOT / f"parts/v3_1_validation_gpu{args.gpu}.csv"
    completed = set()
    if output.exists():
        with open(output, newline="") as handle:
            completed = {(int(r["step"]), int(r["video_index"]), int(r["qp"])) for r in csv.DictReader(handle)}
    quality = quality_models(device)
    for step in steps:
        checkpoint = ROOT / "checkpoints/stage_b" / f"step_{step:05d}.pt"
        while not checkpoint.is_file() or checkpoint.stat().st_size < 100_000_000:
            if not args.wait_checkpoints:
                raise FileNotFoundError(checkpoint)
            print(f"waiting for {checkpoint}", flush=True)
            time.sleep(20)
        joint = load_joint(checkpoint, device)
        digest = sha(checkpoint)
        for index in indices:
            pending = [qp for qp in (1, 3) if (step, index, qp) not in completed]
            if not pending:
                continue
            video = videos[index]
            frames = video_frames(video["path"], 32)
            fps = float(Fraction(video["fps"]))
            for qp in pending:
                stream = ROOT / "bitstreams/v3_1_validation" / f"step_{step:05d}" / f"video_{index}_qp{qp}.bin"
                summary, _ = run_stream(frames, qp, joint, device, fps, stream, quality=quality)
                row = {"stage": "stage_b", "step": step, "video": video["filename"],
                       "video_index": index, "qp": qp, "real_bytes": summary["bytes"],
                       "kbps": summary["kbps"], "bpp": summary["bpp"], "LPIPS": summary["LPIPS"],
                       "DISTS": summary["DISTS"], "PSNR": summary["PSNR"], "MS_SSIM": summary["MS_SSIM"],
                       "SSIM": summary["SSIM"], "bitstream_path": str(stream),
                       "bitstream_sha256": summary["bitstream_sha256"], "bytes_consumed": summary["bytes_consumed"],
                       "decode_status": summary["decode_status"], "state_sync_pass": summary["state_sync_pass"],
                       "compression_hash_before": summary["compression_hash_before"],
                       "compression_hash_after": summary["compression_hash_after"],
                       "checkpoint": str(checkpoint), "checkpoint_sha256": digest}
                with open(output, "a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS)
                    if handle.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)
                completed.add((step, index, qp))
                print(step, index, qp, summary["bytes"], flush=True)


if __name__ == "__main__":
    main()
