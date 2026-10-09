#!/usr/bin/env python3
import argparse
import gc
import json
from pathlib import Path

from common import ROOT, config, sha256, split, torch_load, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    if args.group == "test" and not (ROOT / "checkpoints/final/SELECTIONS_FROZEN.json").exists():
        raise RuntimeError("V9 TEST cache audit is sealed until validation selections are frozen")

    records = []
    expected_cache_paths = set()
    for video in split()[args.group]:
        name = Path(video["path"]).stem
        for qp in config()["qps"]:
            cache_path = ROOT / f"cache/receiver/{args.group}/{name}_qp{qp}.pt"
            stream_path = ROOT / f"cache/bitstreams/{args.group}/{name}_qp{qp}.bin"
            metadata_path = stream_path.with_suffix(".json")
            expected_cache_paths.add(cache_path.resolve())
            checks = {"cache_exists": cache_path.exists(), "stream_exists": stream_path.exists(),
                      "metadata_exists": metadata_path.exists()}
            if all(checks.values()):
                metadata = json.loads(metadata_path.read_text())
                payload = torch_load(cache_path, map_location="cpu", weights_only=False)
                frames = payload.get("records", [])
                checks.update({
                    "video_identity": payload.get("video", {}).get("path") == video["path"],
                    "group_exact": payload.get("group") == args.group,
                    "requested_qp_exact": payload.get("requested_qp") == qp,
                    "stream_path_exact": payload.get("stream_path") == str(stream_path),
                    "stream_hash_exact": (payload.get("stream_sha256") == metadata.get("sha256") ==
                                          sha256(stream_path)),
                    "stream_size_exact": metadata.get("actual_bits") == stream_path.stat().st_size * 8,
                    "rate_exact": (payload.get("actual_bits") == metadata.get("actual_bits") and
                                   payload.get("visible_bpp") == metadata.get("visible_bpp")),
                    "zero_additional_bits": payload.get("additional_transmitted_bits") == 0,
                    "frame_count_exact": len(frames) == config()["frames_per_video"],
                    "frame_indexes_exact": [row.get("frame") for row in frames] == list(range(96)),
                    "actual_qp_trajectory_exact": ([row.get("actual_qp") for row in frames] ==
                                                   metadata.get("actual_qps")),
                    "rerun_exact": all(row.get("rerun_max_abs") == 0 for row in frames),
                    "codeword_shapes_exact": all(tuple(row["codeword"].shape) == (1, 18, 68, 120)
                                                 for row in frames),
                    "quant_shapes_exact": all(tuple(row["quant"].shape) == (1, 320, 1, 1)
                                              for row in frames),
                    "feature_shapes_exact": all(
                        (row["bridge_feature"] is None if row["frame_type"] == "I" else
                         tuple(row["bridge_feature"].shape) == (1, 256, 136, 240))
                        for row in frames),
                })
                del payload, frames
                gc.collect()
            records.append({"video": name, "qp": qp, "checks": checks,
                            "status": "PASS" if all(checks.values()) else "FAIL"})
            print(f"{args.group} {name} qp{qp}: {records[-1]['status']}", flush=True)

    actual_cache_paths = {path.resolve() for path in (ROOT / f"cache/receiver/{args.group}").glob("*.pt")}
    exact_file_set = actual_cache_paths == expected_cache_paths
    payload = {"group": args.group, "pairs": len(records), "exact_file_set": exact_file_set,
               "status": ("PASS" if exact_file_set and
                          all(row["status"] == "PASS" for row in records) else "FAIL"),
               "records": records}
    write_json(ROOT / f"analysis/{args.group}_cache_integrity.json", payload)
    print(json.dumps({"group": args.group, "pairs": len(records), "status": payload["status"]}, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
