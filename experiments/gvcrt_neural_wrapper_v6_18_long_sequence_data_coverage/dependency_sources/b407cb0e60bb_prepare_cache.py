#!/usr/bin/env python3
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch

from common import ROOT, config, sha256, split, torch_load, write_json
from gvc_hooks import decode_records, encode, iter_frames, load_models


def training_plan():
    destination = ROOT / "analysis/training_plan.json"
    if destination.exists():
        return json.loads(destination.read_text())
    cfg = config()
    rng = random.Random(cfg["seed"])
    rows = []
    for epoch in range(5):
        videos = list(range(16))
        qps = list(cfg["qps"]) * 4
        rng.shuffle(videos)
        rng.shuffle(qps)
        for video_index, qp in zip(videos, qps):
            frame = rng.randint(2, cfg["frames_per_video"] - 1)
            rows.append({"step": len(rows) + 1, "epoch": epoch + 1,
                         "video_index": video_index, "requested_qp": qp,
                         "frame": frame, "crop_y": rng.randint(0, 68 - cfg["latent_crop"]),
                         "crop_x": rng.randint(0, 120 - cfg["latent_crop"])})
    assert len(rows) == cfg["training_updates"]
    write_json(destination, {"seed": cfg["seed"], "updates": rows,
                             "qp_counts": {str(qp): sum(row["requested_qp"] == qp for row in rows)
                                           for qp in cfg["qps"]}})
    return {"seed": cfg["seed"], "updates": rows}


def ensure_stream(video, group, qp, models, device):
    name = Path(video["path"]).stem
    path = ROOT / f"cache/bitstreams/{group}/{name}_qp{qp}.bin"
    metadata = path.with_suffix(".json")
    if path.exists() and metadata.exists():
        payload = json.loads(metadata.read_text())
        if payload["sha256"] != sha256(path):
            raise RuntimeError(f"Stream hash mismatch: {path}")
        return path, payload
    frames = list(iter_frames(video, 96, device))
    frame_bits, actual_qps = encode(frames, *models, path, qp)
    payload = {"video": video["path"], "group": group, "requested_qp": qp,
               "frames": 96, "sha256": sha256(path), "actual_bits": path.stat().st_size * 8,
               "visible_bpp": path.stat().st_size * 8 / (96 * 1920 * 1080),
               "frame_bits": frame_bits, "actual_qps": actual_qps}
    write_json(metadata, payload)
    del frames
    torch.cuda.empty_cache()
    return path, payload


def prepare_train(models, device, start, limit):
    plan = training_plan()["updates"]
    videos = split()["train"]
    grouped = defaultdict(list)
    for row in plan:
        grouped[(row["video_index"], row["requested_qp"])].append(row)
    stop = 16 if limit is None else min(16, start + limit)
    for (video_index, qp), steps in sorted(grouped.items()):
        if not start <= video_index < stop:
            continue
        video = videos[video_index]
        name = Path(video["path"]).stem
        stream, metadata = ensure_stream(video, "train", qp, models, device)
        needed = {index for row in steps for index in (row["frame"] - 2, row["frame"] - 1, row["frame"])}
        records = decode_records(stream.read_bytes(), 96, *models, keep_feature=True,
                                 record_indexes=needed)
        by_frame = {record["frame"]: record for record in records}
        step_payloads = {}
        for row in steps:
            y, x = row["crop_y"], row["crop_x"]
            cropped = []
            for frame in (row["frame"] - 2, row["frame"] - 1, row["frame"]):
                record = by_frame[frame]
                feature = record["bridge_feature"]
                cropped.append({"frame": frame, "frame_type": record["frame_type"],
                                "actual_qp": record["actual_qp"], "quant": record["quant"],
                                "codeword": record["codeword"][:, :, y:y + 16, x:x + 16].clone(),
                                "bridge_feature": (feature[:, :, 2*y:2*(y + 16), 2*x:2*(x + 16)].clone()
                                                   if feature is not None else None)})
            step_payloads[row["step"]] = {"plan": row, "records": cropped, "targets": {}}
        frame_to_steps = defaultdict(list)
        for row in steps:
            for frame in (row["frame"] - 1, row["frame"]):
                frame_to_steps[frame].append(row)
        for frame, target in enumerate(iter_frames(video, 96, device)):
            for row in frame_to_steps.get(frame, ()):
                py, px = row["crop_y"] * 16, row["crop_x"] * 16
                step_payloads[row["step"]]["targets"][frame] = target[:, :, py:py + 256, px:px + 256].cpu()
        for step, payload in step_payloads.items():
            payload.update({"video": video, "stream_sha256": metadata["sha256"],
                            "additional_transmitted_bits": 0})
            destination = ROOT / f"cache/receiver/train_steps/step_{step:03d}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, destination)
        print(f"train video{video_index} qp{qp}: {len(steps)} steps", flush=True)
        del records, by_frame, step_payloads
        torch.cuda.empty_cache()


def prepare_eval(group, start, limit, models, device):
    if group == "test" and not (ROOT / "checkpoints/final/SELECTIONS_FROZEN.json").exists():
        raise RuntimeError("V9 TEST is sealed until all validation selections are frozen")
    videos = split()[group][start:]
    if limit is not None:
        videos = videos[:limit]
    for offset, video in enumerate(videos):
        name = Path(video["path"]).stem
        for qp in config()["qps"]:
            destination = ROOT / f"cache/receiver/{group}/{name}_qp{qp}.pt"
            stream, metadata = ensure_stream(video, group, qp, models, device)
            if destination.exists():
                payload = torch_load(destination, map_location="cpu", weights_only=False)
                if payload["stream_sha256"] != sha256(stream):
                    raise RuntimeError(f"Cached stream mismatch: {destination}")
                status = "verified-reuse"
            else:
                records = decode_records(stream.read_bytes(), 96, *models, keep_feature=True)
                if metadata["actual_qps"] != [record["actual_qp"] for record in records]:
                    raise RuntimeError("Requested/actual QP trajectory mismatch")
                if max(record["rerun_max_abs"] for record in records) != 0:
                    raise RuntimeError("Frozen generator rerun mismatch")
                payload = {"video": video, "group": group, "requested_qp": qp,
                           "stream_path": str(stream), "stream_sha256": metadata["sha256"],
                           "actual_bits": metadata["actual_bits"], "visible_bpp": metadata["visible_bpp"],
                           "additional_transmitted_bits": 0, "records": records}
                destination.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, destination)
                status = "generated"
                del records, payload
            print(f"{group} {start + offset} qp{qp} {status}", flush=True)
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--gpu", type=int, choices=(4, 5, 6, 7), default=6)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    models = load_models(device)
    if args.group == "train":
        prepare_train(models, device, args.start, args.limit)
    else:
        prepare_eval(args.group, args.start, args.limit, models, device)


if __name__ == "__main__":
    main()
