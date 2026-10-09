#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from common import ROOT, config, fid_terms, kid_unbiased, split, write_json


BRANCHES = ("m0", "m1", "m2", "m3", "m4", "m5", "m6")


def weighted_fid(reference, candidate, counts):
    active = counts > 0
    counts, reference, candidate = counts[active], reference[active], candidate[active]
    frames = reference.shape[1]
    total = counts.sum() * frames
    weights = counts[:, None, None]
    reference_mean = (reference * weights).sum(dim=(0, 1)) / total
    candidate_mean = (candidate * weights).sum(dim=(0, 1)) / total
    root_weights = counts.sqrt()[:, None, None]
    reference_centered = ((reference - reference_mean) * root_weights).reshape(-1, reference.shape[-1])
    candidate_centered = ((candidate - candidate_mean) * root_weights).reshape(-1, candidate.shape[-1])
    cross = reference_centered @ candidate_centered.T / (total - 1)
    return (torch.square(reference_mean - candidate_mean).sum() +
            (torch.square(reference_centered).sum() + torch.square(candidate_centered).sum()) /
            (total - 1) - 2 * torch.linalg.svdvals(cross).sum())


def kernels(first, second):
    scale = first.shape[-1]
    return ((first @ first.T / scale + 1) ** 3,
            (second @ second.T / scale + 1) ** 3,
            (first @ second.T / scale + 1) ** 3)


def block_sums(matrix, videos, frames):
    return matrix.reshape(videos, frames, videos, frames).sum(axis=(1, 3))


def kernel_statistics(kernel_set, videos, frames):
    return [tuple((block_sums(matrix, videos, frames),
                   np.diag(matrix).reshape(videos, frames).sum(1))
                  for matrix in kernels_for_branch)
            for kernels_for_branch in kernel_set]


def weighted_kid(statistics, counts, frames):
    total, videos = frames * counts.sum(), len(counts)
    values = []
    for (blocks_xx, diagonal_xx), (blocks_yy, diagonal_yy), (blocks_xy, _) in statistics:
        sum_xx = counts @ blocks_xx @ counts - np.sum(counts * diagonal_xx)
        sum_yy = counts @ blocks_yy @ counts - np.sum(counts * diagonal_yy)
        sum_xy = counts @ blocks_xy @ counts
        values.append(sum_xx / (total * (total - 1)) + sum_yy / (total * (total - 1)) -
                      2 * sum_xy / (total * total))
    return values


def summary(values, seed):
    values = np.asarray(values)
    return {"procedure": "paired video-level cluster bootstrap; sample 12 videos with replacement and include all 96 frames",
            "primary_resampling_unit": "video", "frame_bootstrap_used": False,
            "repetitions": len(values), "seed": seed, "mean_delta": float(values.mean()),
            "delta_95_ci": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "probability_candidate_better": float(np.mean(values < 0))}


def load_features(qp):
    grouped = {key: [] for key in ("reference",) + BRANCHES}
    for video in split()["test"]:
        name = Path(video["path"]).stem
        with np.load(ROOT / f"outputs/test/{name}_qp{qp}.npz") as data:
            for key in grouped:
                grouped[key].append(data[key].astype(np.float64))
    return {key: np.stack(value) for key, value in grouped.items()}


def run_qp(qp, physical_gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    device = torch.device("cuda:0")
    cfg, features = config(), load_features(qp)
    flat = lambda value: value.reshape(-1, value.shape[-1])
    reference = features["reference"]
    point = {branch: {"fid": fid_terms(flat(reference), flat(features[branch]))["fid"],
                      "kid": kid_unbiased(flat(reference), flat(features[branch]))}
             for branch in BRANCHES}
    tensors = {key: torch.from_numpy(value).to(device=device, dtype=torch.float64)
               for key, value in features.items()}
    fid_deltas = {branch: [] for branch in BRANCHES[1:]}
    rng = np.random.default_rng(cfg["bootstrap_seed"] + qp)
    for repetition in range(cfg["bootstrap_repetitions"]):
        counts_np = np.bincount(rng.integers(0, 12, 12), minlength=12)
        counts = torch.as_tensor(counts_np, device=device, dtype=torch.float64)
        with torch.no_grad():
            baseline = weighted_fid(tensors["reference"], tensors["m0"], counts)
            for branch in BRANCHES[1:]:
                delta = weighted_fid(tensors["reference"], tensors[branch], counts) - baseline
                fid_deltas[branch].append(float(delta.cpu()))
        if (repetition + 1) % 25 == 0:
            print(f"QP{qp} FID bootstrap {repetition + 1}/{cfg['bootstrap_repetitions']}", flush=True)
    reference_flat = flat(reference)
    kernel_sets = {branch: kernels(reference_flat, flat(features[branch])) for branch in BRANCHES}
    kernel_stats = kernel_statistics([kernel_sets[branch] for branch in BRANCHES], 12, 96)
    del kernel_sets
    kid_deltas = {branch: [] for branch in BRANCHES[1:]}
    rng = np.random.default_rng(cfg["bootstrap_seed"] + 100 + qp)
    for _ in range(cfg["bootstrap_repetitions"]):
        counts = np.bincount(rng.integers(0, 12, 12), minlength=12)
        estimates = weighted_kid(kernel_stats, counts, 96)
        for index, branch in enumerate(BRANCHES[1:], 1):
            kid_deltas[branch].append(estimates[index] - estimates[0])
    results = []
    for branch in BRANCHES:
        row = {"requested_qp": qp, "branch": branch, **point[branch]}
        if branch != "m0":
            row.update({"fid_delta_vs_m0": point[branch]["fid"] - point["m0"]["fid"],
                        "kid_delta_vs_m0": point[branch]["kid"] - point["m0"]["kid"],
                        "fid_video_bootstrap": summary(fid_deltas[branch], cfg["bootstrap_seed"] + qp),
                        "kid_video_bootstrap": summary(kid_deltas[branch], cfg["bootstrap_seed"] + 100 + qp)})
        results.append(row)
    write_json(ROOT / f"analysis/fid_kid_bootstrap_qp{qp}.json",
               {"physical_gpu": physical_gpu, "results": results})
    print(json.dumps({"requested_qp": qp, "results": results}, indent=2))


def aggregate():
    results = []
    for qp in config()["qps"]:
        results.extend(json.loads((ROOT / f"analysis/fid_kid_bootstrap_qp{qp}.json").read_text())["results"])
    write_json(ROOT / "analysis/fid_kid_per_qp.json", {
        "protocol": "pooled per QP; 500 paired whole-video bootstrap repetitions; negative delta is better",
        "results": results})
    print(json.dumps({"rows": len(results)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qp", type=int, choices=range(4))
    parser.add_argument("--gpu", type=int, choices=(4, 5, 6, 7), default=6)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    elif args.qp is not None:
        run_qp(args.qp, args.gpu)
    else:
        raise SystemExit("provide --qp or --aggregate")


if __name__ == "__main__":
    main()
