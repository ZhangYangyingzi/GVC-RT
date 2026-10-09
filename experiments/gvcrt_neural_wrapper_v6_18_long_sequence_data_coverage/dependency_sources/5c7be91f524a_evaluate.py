#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from common import ROOT, V6, config, sha256, split, torch_load, write_csv, write_json
from gvc_hooks import iter_frames, load_models
from methods import (baseline_image, candidate_specs, load_candidate, load_historical_v8,
                     render)
from metrics import spatial_metrics, temporal_delta_l1, torch_msssim_rgb, unit

sys.path.insert(0, str(V6 / "src"))
from fid_metric import InceptionPool3
from flolpips_metric import load_models as load_flolpips


def quality_models(device):
    import lpips
    from DISTS_pytorch import DISTS
    return (lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False),
            DISTS().to(device).eval().requires_grad_(False), InceptionPool3().to(device),
            *load_flolpips(device))


def save_visual_comparison(name, qp, frame_index, reference, outputs, branch_names):
    labels = ["reference"] + branch_names
    tensors = [reference] + [outputs[branch] for branch in branch_names]
    cell_width, cell_height, label_height, columns = 480, 270, 24, 4
    rows = (len(labels) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_width, rows * (cell_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, tensor) in enumerate(zip(labels, tensors)):
        array = (tensor[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        image = Image.fromarray(array).resize((cell_width, cell_height), Image.Resampling.LANCZOS)
        x, y = (index % columns) * cell_width, (index // columns) * (cell_height + label_height)
        canvas.paste(image, (x, y + label_height))
        draw.text((x + 6, y + 5), label, fill="black")
    destination = ROOT / f"outputs/visual_comparisons/{name}_qp{qp}_frame{frame_index:03d}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def branches_for(phase, codec, device):
    if phase == "candidate":
        specs = candidate_specs()
        if len(specs) != 30:
            raise RuntimeError(f"Expected 30 trained candidates, found {len(specs)}")
        branches = {}
        metadata = {}
        for spec in specs:
            branches[spec["candidate"]], payload = load_candidate(
                spec["method"], spec["path"], codec, device)
            metadata[spec["candidate"]] = spec | {
                "method": spec["method"]}
        return branches, metadata
    selection_path = (ROOT / "checkpoints/final/SELECTIONS_FROZEN.json"
                      if phase == "test" else ROOT / "analysis/validation_selection.json")
    selections = json.loads(selection_path.read_text())
    branches, metadata = {}, {}
    for method, selected in selections["selected"].items():
        branches[method], payload = load_candidate(method, selected["path"], codec, device)
        metadata[method] = selected | {
            "method": method,
            "checkpoint_sha256": selected.get("sha256", sha256(selected["path"]))}
    if phase == "test":
        branches["m6"] = load_historical_v8(device)
        metadata["m6"] = {"method": "m6", "path": config()["historical_v8_checkpoint"],
                          "checkpoint_sha256": config()["historical_v8_sha256"]}
    return branches, metadata


@torch.inference_mode()
def evaluate_pair(group, video_index, qp, phase, physical_gpu, device, codec, branches, metadata, quality):
    cfg = config()
    video = split()[group][video_index]
    name = Path(video["path"]).stem
    phase_dir = "candidate_pairs" if phase == "candidate" else f"{group}_pairs"
    prefix = ROOT / f"analysis/{phase_dir}/{name}_qp{qp}"
    feature_path = ROOT / f"outputs/{'candidate_validation' if phase == 'candidate' else group}/{name}_qp{qp}.npz"
    if prefix.with_suffix(".json").exists() and feature_path.exists():
        print(f"{phase} video{video_index} qp{qp}: verified-reuse", flush=True)
        return
    payload = torch_load(ROOT / f"cache/receiver/{group}/{name}_qp{qp}.pt",
                         map_location="cpu", weights_only=False)
    if payload["stream_sha256"] != sha256(payload["stream_path"]):
        raise RuntimeError("Stream identity failure")
    records = payload["records"]
    frame_count = 12 if phase == "candidate" else cfg["frames_per_video"]
    lpips_model, dists_model, inception, flow, flo_perceptual = quality
    branch_names = ["m0"] + list(branches)
    rows, transitions = [], []
    features = {branch: [] for branch in ["reference"] + branch_names}
    previous_reference, previous_outputs, previous_baseline = None, {}, None
    previous_record = None
    latency = {branch: 0.0 for branch in branch_names}
    peak_incremental_memory = {branch: 0 for branch in branch_names}
    for frame_index, target in enumerate(iter_frames(video, frame_count, device)):
        record = records[frame_index]
        reference = unit(target)
        baseline_raw = baseline_image(record, codec, device)
        outputs_raw = {"m0": baseline_raw}
        outputs = {"m0": unit(baseline_raw)}
        for branch, model in branches.items():
            method = metadata[branch]["method"]
            torch.cuda.synchronize()
            before_memory = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            rendered, seconds = render(method, model, record, previous_record, baseline_raw,
                                       previous_baseline, codec, device)
            peak_incremental_memory[branch] = max(
                peak_incremental_memory[branch],
                max(0, torch.cuda.max_memory_allocated(device) - before_memory))
            outputs_raw[branch] = rendered
            outputs[branch] = unit(rendered)
            latency[branch] += seconds
        ssim = {}
        for start in range(0, len(branch_names), 2):
            chunk_names = branch_names[start:start + 2]
            chunk = torch.cat([outputs[branch] for branch in chunk_names])
            chunk_ssim = torch_msssim_rgb(chunk, reference)
            ssim.update({branch: float(chunk_ssim[offset])
                         for offset, branch in enumerate(chunk_names)})
        for offset, branch in enumerate(branch_names):
            rows.append({"phase": phase, "video": name, "video_index": video_index,
                         "requested_qp": qp, "frame": frame_index, "branch": branch,
                         "method": metadata.get(branch, {}).get("method", branch),
                         "actual_bits": payload["actual_bits"], "visible_bpp": payload["visible_bpp"],
                         **spatial_metrics(outputs[branch], reference, lpips_model, dists_model),
                         "ssim": ssim[branch]})
        reference_feature = inception(reference).cpu().numpy()[0]
        features["reference"].append(reference_feature)
        for start in range(0, len(branch_names), 2):
            chunk_names = branch_names[start:start + 2]
            chunk_features = inception(
                torch.cat([outputs[branch] for branch in chunk_names])).cpu().numpy()
            for offset, branch in enumerate(chunk_names):
                branch_feature = chunk_features[offset]
                features[branch].append(branch_feature)
                rows[-len(branch_names) + start + offset]["paired_inception_l2"] = float(
                    np.linalg.norm(branch_feature - reference_feature))
        if (phase == "test" and video_index == 0 and qp == 2 and
                frame_index in (0, 32, 64, 95)):
            save_visual_comparison(name, qp, frame_index, reference, outputs, branch_names)
        if previous_reference is not None:
            reference_flow = flow(previous_reference, reference)
            for branch in branch_names:
                difference = reference_flow - flow(previous_outputs[branch], outputs[branch])
                transitions.append({"phase": phase, "video": name, "video_index": video_index,
                                    "requested_qp": qp, "transition": frame_index - 1,
                                    "branch": branch,
                                    "method": metadata.get(branch, {}).get("method", branch),
                                    "temporal_delta_l1": temporal_delta_l1(
                                        outputs[branch], previous_outputs[branch], reference, previous_reference),
                                    "flolpips": float(flo_perceptual(
                                        previous_reference, previous_outputs[branch], difference, normalize=True))})
        previous_reference, previous_outputs = reference, outputs
        previous_baseline, previous_record = baseline_raw, record
        print(f"{phase} video{video_index} qp{qp} frame {frame_index + 1}/{frame_count}", flush=True)
    write_csv(prefix.with_name(prefix.name + "_frames.csv"), rows)
    write_csv(prefix.with_name(prefix.name + "_transitions.csv"), transitions)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_path, **{key: np.asarray(value, np.float32) for key, value in features.items()})
    write_json(prefix.with_suffix(".json"), {
        "phase": phase, "video": name, "video_index": video_index, "requested_qp": qp,
        "physical_gpu": physical_gpu,
        "frames": frame_count, "branches": branch_names, "stream_path": payload["stream_path"],
        "stream_sha256": payload["stream_sha256"], "actual_bits": payload["actual_bits"],
        "visible_bpp": payload["visible_bpp"], "additional_transmitted_bits": 0,
        "candidate_output_enters_official_dpb": False,
        "checkpoint_sha256": {
            branch: metadata[branch]["checkpoint_sha256"] for branch in branches
        },
        "seconds": {branch: value for branch, value in latency.items()},
        "peak_incremental_memory_bytes": peak_incremental_memory})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("candidate", "validation", "test"), required=True)
    parser.add_argument("--gpu", type=int, choices=(4, 5, 6, 7), default=6)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    group = "validation" if args.phase in ("candidate", "validation") else "test"
    if args.phase == "test" and not (ROOT / "checkpoints/final/SELECTIONS_FROZEN.json").exists():
        raise RuntimeError("V9 TEST is sealed until validation selections are frozen")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    codec = load_models(device)
    branches, metadata = branches_for(args.phase, codec, device)
    quality = quality_models(device)
    stop = len(split()[group]) if args.limit is None else min(len(split()[group]), args.start + args.limit)
    for video_index in range(args.start, stop):
        for qp in config()["qps"]:
            evaluate_pair(group, video_index, qp, args.phase, args.gpu, device, codec,
                          branches, metadata, quality)


if __name__ == "__main__":
    main()
