#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from common import ROOT, config, read_csv, read_json, sha256, split, write_json
from methods import candidate_specs
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("candidate", "validation", "test"), required=True)
    args = parser.parse_args()
    group = "validation" if args.phase in ("candidate", "validation") else "test"
    pair_dir = "candidate_pairs" if args.phase == "candidate" else f"{group}_pairs"
    output_dir = "candidate_validation" if args.phase == "candidate" else group
    expected_branches = 31 if args.phase == "candidate" else (7 if args.phase == "test" else 6)
    frame_count = 12 if args.phase == "candidate" else 96
    frozen = (read_json(ROOT / "checkpoints/final/SELECTIONS_FROZEN.json")
              if args.phase == "test" else None)
    if args.phase == "candidate":
        specs = candidate_specs()
        expected_names = ["m0"] + [item["candidate"] for item in specs]
        expected_checkpoint_hashes = {
            item["candidate"]: item["checkpoint_sha256"] for item in specs
        }
    elif args.phase == "validation":
        expected_names = ["m0", "m1", "m2", "m3", "m4", "m5"]
        selection = read_json(ROOT / "analysis/validation_selection.json")
        expected_checkpoint_hashes = {
            method: selected["checkpoint_sha256"]
            for method, selected in selection["selected"].items()
        }
    else:
        expected_names = ["m0", "m1", "m2", "m3", "m4", "m5", "m6"]
        expected_checkpoint_hashes = {
            method: selected["sha256"] for method, selected in frozen["selected"].items()
        }
        expected_checkpoint_hashes["m6"] = config()["historical_v8_sha256"]
    frozen_hashes_valid = (frozen is None or all(
        sha256(selected["path"]) == selected["sha256"]
        for selected in frozen["selected"].values()))
    records = []
    for video in split()[group]:
        name = Path(video["path"]).stem
        for qp in config()["qps"]:
            prefix = ROOT / f"analysis/{pair_dir}/{name}_qp{qp}"
            pair = json.loads(prefix.with_suffix(".json").read_text())
            stream_metadata = json.loads(Path(pair["stream_path"]).with_suffix(".json").read_text())
            receiver = __import__("common").torch_load(
                ROOT / f"cache/receiver/{group}/{name}_qp{qp}.pt",
                map_location="cpu", weights_only=False)
            frame_rows = read_csv(prefix.with_name(prefix.name + "_frames.csv"))
            transition_rows = read_csv(prefix.with_name(prefix.name + "_transitions.csv"))
            counts = {"frames": len(frame_rows), "transitions": len(transition_rows)}
            with np.load(ROOT / f"outputs/{output_dir}/{name}_qp{qp}.npz") as features:
                shapes = {key: list(value.shape) for key, value in features.items()}
                finite = all(np.isfinite(value).all() for value in features.values())
            checks = {"frame_rows": counts["frames"] == frame_count * expected_branches,
                      "transition_rows": counts["transitions"] == (frame_count - 1) * expected_branches,
                      "feature_keys": len(shapes) == expected_branches + 1,
                      "feature_keys_exact": set(shapes) == {"reference", *expected_names},
                      "feature_shapes": set(map(tuple, shapes.values())) == {(frame_count, 2048)},
                      "finite": finite, "stream_hash": pair["stream_sha256"] == sha256(pair["stream_path"]),
                      "branches_exact": (len(pair["branches"]) == expected_branches and
                                           len(set(pair["branches"])) == expected_branches and
                                           set(pair["branches"]) == set(expected_names)),
                      "frame_branches_exact": (set(row["branch"] for row in frame_rows) ==
                                               set(expected_names) and all(
                                                   sum(row["branch"] == branch for row in frame_rows) == frame_count
                                                   for branch in expected_names)),
                      "transition_branches_exact": (
                          set(row["branch"] for row in transition_rows) == set(expected_names) and all(
                              sum(row["branch"] == branch for row in transition_rows) == frame_count - 1
                              for branch in expected_names)),
                      "checkpoint_hashes_exact": pair["checkpoint_sha256"] == expected_checkpoint_hashes,
                      "requested_qp_exact": pair["requested_qp"] == qp,
                      "allowed_physical_gpu": pair["physical_gpu"] in config()["allowed_physical_gpus"],
                      "actual_qp_trajectory": (stream_metadata["actual_qps"] ==
                                                [item["actual_qp"] for item in receiver["records"]]),
                      "receiver_stream_identity": (receiver["stream_path"] == pair["stream_path"] and
                                                   receiver["stream_sha256"] == pair["stream_sha256"]),
                      "rate_identity": (pair["actual_bits"] == stream_metadata["actual_bits"] and
                                        pair["visible_bpp"] == stream_metadata["visible_bpp"] and
                                        all(int(row["actual_bits"]) == pair["actual_bits"] and
                                            float(row["visible_bpp"]) == pair["visible_bpp"]
                                            for row in frame_rows)),
                      "zero_bits": pair["additional_transmitted_bits"] == 0,
                      "official_dpb_isolated": pair["candidate_output_enters_official_dpb"] is False}
            records.append({"video": name, "qp": qp, "checks": checks,
                            "status": "PASS" if all(checks.values()) else "FAIL"})
    payload = {"phase": args.phase, "pairs": len(records),
               "frozen_checkpoint_hashes_valid": frozen_hashes_valid,
               "status": ("PASS" if frozen_hashes_valid and
                          all(row["status"] == "PASS" for row in records) else "FAIL"),
               "records": records}
    write_json(ROOT / f"analysis/{args.phase}_integrity.json", payload)
    print(json.dumps({"phase": args.phase, "pairs": len(records), "status": payload["status"]}, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
