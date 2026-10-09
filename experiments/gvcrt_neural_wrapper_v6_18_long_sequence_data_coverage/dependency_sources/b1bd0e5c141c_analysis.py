#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from common import (ROOT, config, fid_terms, kid_unbiased, read_csv, sha256, split,
                    write_csv, write_json)
from methods import candidate_specs


METRICS = ("psnr", "ssim", "lpips", "dists", "paired_inception_l2",
           "flolpips", "temporal_delta_l1", "fid", "kid")
RATE_KEYS = ("actual_bits", "visible_bpp")


def mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def collect(phase, group, branches, frame_count):
    pair_dir = "candidate_pairs" if phase == "candidate" else f"{group}_pairs"
    output_dir = "candidate_validation" if phase == "candidate" else group
    frames, transitions = [], []
    feature_sets = {qp: {key: [] for key in ["reference"] + branches} for qp in config()["qps"]}
    for video in split()[group]:
        name = Path(video["path"]).stem
        for qp in config()["qps"]:
            prefix = ROOT / f"analysis/{pair_dir}/{name}_qp{qp}"
            frames.extend(read_csv(prefix.with_name(prefix.name + "_frames.csv")))
            transitions.extend(read_csv(prefix.with_name(prefix.name + "_transitions.csv")))
            with np.load(ROOT / f"outputs/{output_dir}/{name}_qp{qp}.npz") as data:
                for key in feature_sets[qp]:
                    feature_sets[qp][key].append(data[key].copy())
    rows = []
    for qp in config()["qps"]:
        reference = np.concatenate(feature_sets[qp]["reference"])
        for branch in branches:
            selected_frames = [row for row in frames if int(row["requested_qp"]) == qp and row["branch"] == branch]
            selected_transitions = [row for row in transitions if int(row["requested_qp"]) == qp and row["branch"] == branch]
            candidate = np.concatenate(feature_sets[qp][branch])
            rows.append({"scope": "qp", "requested_qp": qp, "branch": branch,
                         "videos": len(split()[group]), "frames": len(selected_frames),
                         **{key: mean(selected_frames, key) for key in RATE_KEYS},
                         **{key: mean(selected_frames, key) for key in
                            ("psnr", "ssim", "lpips", "dists", "paired_inception_l2")},
                         **{key: mean(selected_transitions, key) for key in
                            ("flolpips", "temporal_delta_l1")},
                         "fid": fid_terms(reference, candidate)["fid"],
                         "kid": kid_unbiased(reference, candidate)})
    for branch in branches:
        selected = [row for row in rows if row["branch"] == branch]
        rows.append({"scope": "mean_qp", "requested_qp": "mean", "branch": branch,
                     "videos": len(split()[group]), "frames": sum(row["frames"] for row in selected),
                     **{key: float(np.mean([row[key] for row in selected]))
                        for key in METRICS + RATE_KEYS}})
    return rows, frames, transitions


def candidate_selection():
    specs = candidate_specs()
    branches = ["m0"] + [spec["candidate"] for spec in specs]
    rows, _, _ = collect("candidate", "validation", branches, 12)
    means = {row["branch"]: row for row in rows if row["scope"] == "mean_qp"}
    baseline = means["m0"]
    candidates = []
    for spec in specs:
        row = means[spec["candidate"]]
        gains = {"psnr_gain": row["psnr"] - baseline["psnr"],
                 **{f"{key}_reduction": (baseline[key] - row[key]) / baseline[key]
                    for key in ("lpips", "dists", "flolpips", "temporal_delta_l1")},
                 "fid_reduction": baseline["fid"] - row["fid"],
                 "kid_reduction": baseline["kid"] - row["kid"]}
        eligible = (gains["psnr_gain"] > 0 and gains["lpips_reduction"] >= 0 and
                    gains["dists_reduction"] >= 0 and gains["flolpips_reduction"] >= 0 and
                    gains["temporal_delta_l1_reduction"] >= -0.02)
        candidates.append(spec | {key: row[key] for key in METRICS} | gains | {"eligible": eligible})
    score_keys = ("psnr_gain", "lpips_reduction", "dists_reduction", "flolpips_reduction",
                  "temporal_delta_l1_reduction", "fid_reduction", "kid_reduction")
    weights = (0.20, 0.20, 0.20, 0.15, 0.10, 0.075, 0.075)
    for row in candidates:
        row["score"] = sum(weight * ((row[key] - min(item[key] for item in candidates)) /
                           max(max(item[key] for item in candidates) - min(item[key] for item in candidates), 1e-12))
                           for key, weight in zip(score_keys, weights))
    selected = {}
    for method in ("m1", "m2", "m3", "m4", "m5"):
        options = [row for row in candidates if row["method"] == method]
        eligible = [row for row in options if row["eligible"]]
        selected[method] = max(eligible or options, key=lambda row: row["score"])
        selected[method]["selected_from_eligible_set"] = bool(eligible)
    candidates_path = ROOT / "analysis/validation_candidates.csv"
    write_csv(candidates_path, candidates)
    write_json(ROOT / "analysis/validation_selection.json", {
        "protocol": "12 full-resolution contiguous frames from all 8 validation videos at QP0-3; no test access",
        "score": "min-max normalized weighted score: PSNR .20, LPIPS .20, DISTS .20, FloLPIPS .15, Temporal .10, FID .075, KID .075",
        "eligibility": "PSNR>baseline; LPIPS/DISTS/FloLPIPS non-worse; TemporalDeltaL1 <=2% worse",
        "training_integrity_sha256": sha256(ROOT / "analysis/training_integrity.json"),
        "candidate_integrity_sha256": sha256(ROOT / "analysis/candidate_integrity.json"),
        "validation_candidates_sha256": sha256(candidates_path),
        "baseline": {key: baseline[key] for key in METRICS}, "selected": selected})
    print(json.dumps({method: {key: value for key, value in row.items()
                               if key in ("candidate", "lr", "update", "eligible", "score")}
                      for method, row in selected.items()}, indent=2))


def aggregate(group):
    branches = ["m0", "m1", "m2", "m3", "m4", "m5"] + (["m6"] if group == "test" else [])
    rows, frames, transitions = collect(group, group, branches, 96)
    write_csv(ROOT / f"analysis/{group}_summary.csv", rows)
    if group == "test":
        write_csv(ROOT / "analysis/per_video_qp.csv", frames)
        thirds = []
        for third, start in zip(("early", "middle", "late"), (0, 32, 64)):
            for branch in branches:
                frame_rows = [row for row in frames if row["branch"] == branch and
                              start <= int(row["frame"]) < start + 32]
                transition_rows = [row for row in transitions if row["branch"] == branch and
                                   start <= int(row["transition"]) < start + 31]
                thirds.append({"third": third, "branch": branch,
                               **{key: mean(frame_rows, key) for key in ("psnr", "lpips", "dists")},
                               **{key: mean(transition_rows, key) for key in
                                  ("flolpips", "temporal_delta_l1")}})
        for third in ("early", "middle", "late"):
            baseline = next(row for row in thirds if row["third"] == third and row["branch"] == "m0")
            for row in (item for item in thirds if item["third"] == third):
                row["psnr_gain_db"] = row["psnr"] - baseline["psnr"]
                for key in ("lpips", "dists", "flolpips", "temporal_delta_l1"):
                    row[f"{key}_reduction_fraction"] = ((baseline[key] - row[key]) /
                                                         baseline[key])
        write_csv(ROOT / "analysis/sequence_thirds.csv", thirds)
        write_json(ROOT / "analysis/fid_kid_per_qp.json", {
            "protocol": "pooled independently at each QP over 12 videos x 96 frames",
            "results": [{key: row[key] for key in ("requested_qp", "branch", "fid", "kid")}
                        for row in rows if row["scope"] == "qp"]})
    print(json.dumps({"group": group, "rows": len(rows), "frame_rows": len(frames)}, indent=2))


def seal():
    selection = json.loads((ROOT / "analysis/validation_selection.json").read_text())
    training_integrity = ROOT / "analysis/training_integrity.json"
    if not training_integrity.exists() or json.loads(training_integrity.read_text())["status"] != "PASS":
        raise RuntimeError("Formal training integrity must pass before sealing")
    if selection["training_integrity_sha256"] != sha256(training_integrity):
        raise RuntimeError("Training integrity changed after candidate selection")
    if selection["candidate_integrity_sha256"] != sha256(ROOT / "analysis/candidate_integrity.json"):
        raise RuntimeError("Candidate integrity changed after candidate selection")
    if selection["validation_candidates_sha256"] != sha256(ROOT / "analysis/validation_candidates.csv"):
        raise RuntimeError("Validation candidate table changed after candidate selection")
    for phase in ("candidate", "validation"):
        integrity_path = ROOT / f"analysis/{phase}_integrity.json"
        if not integrity_path.exists() or json.loads(integrity_path.read_text())["status"] != "PASS":
            raise RuntimeError(f"{phase} integrity must pass before sealing")
    validation_cache_integrity = ROOT / "analysis/validation_cache_integrity.json"
    if (not validation_cache_integrity.exists() or
            json.loads(validation_cache_integrity.read_text())["status"] != "PASS"):
        raise RuntimeError("Validation receiver-cache integrity must pass before sealing")
    if not (ROOT / "analysis/validation_summary.csv").exists():
        raise RuntimeError("Full validation aggregation must finish before sealing")
    frozen = {}
    for method, selected in selection["selected"].items():
        source = Path(selected["path"])
        if sha256(source) != selected["checkpoint_sha256"]:
            raise RuntimeError(f"Selected source checkpoint changed before freeze: {method}")
        destination = ROOT / f"checkpoints/final/{method}.pt"
        shutil.copy2(source, destination)
        destination_hash = sha256(destination)
        if destination_hash != selected["checkpoint_sha256"]:
            raise RuntimeError(f"Frozen checkpoint copy mismatch: {method}")
        frozen[method] = {**selected, "selection_source_path": str(source),
                          "path": str(destination), "sha256": destination_hash}
    payload = {"status": "FROZEN_BEFORE_TEST", "selected": frozen,
               "selection_protocol": selection["protocol"],
               "validation_selection_sha256": sha256(ROOT / "analysis/validation_selection.json"),
               "validation_summary_sha256": sha256(ROOT / "analysis/validation_summary.csv"),
               "training_integrity_sha256": sha256(training_integrity),
               "validation_cache_integrity_sha256": sha256(validation_cache_integrity),
               "official_i_sha256": config()["official_i_sha256"],
               "official_p_sha256": config()["official_p_sha256"],
               "additional_transmitted_bits": 0}
    write_json(ROOT / "checkpoints/final/SELECTIONS_FROZEN.json", payload)
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-selection", action="store_true")
    parser.add_argument("--aggregate", choices=("validation", "test"))
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    if args.candidate_selection:
        candidate_selection()
    elif args.aggregate:
        aggregate(args.aggregate)
    elif args.seal:
        seal()
    else:
        raise SystemExit("select an analysis action")


if __name__ == "__main__":
    main()
