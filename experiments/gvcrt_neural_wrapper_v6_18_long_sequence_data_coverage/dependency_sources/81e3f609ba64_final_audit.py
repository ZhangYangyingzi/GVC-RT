#!/usr/bin/env python3
import json
import subprocess

from common import REPO, ROOT, config, read_csv, read_json, sha256, write_json


def main():
    cfg = config()
    required_integrity = ("interface_audit", "training_integrity", "validation_cache_integrity",
                          "candidate_integrity", "validation_integrity", "test_cache_integrity",
                          "test_integrity")
    integrity = {}
    for name in required_integrity:
        path = ROOT / f"analysis/{name}.json"
        integrity[name] = path.exists() and read_json(path).get("status") == "PASS"

    seal_path = ROOT / "checkpoints/final/SELECTIONS_FROZEN.json"
    seal = read_json(seal_path) if seal_path.exists() else {"selected": {}}
    frozen_hashes = (len(seal["selected"]) == 5 and all(
        sha256(selected["path"]) == selected["sha256"]
        for selected in seal["selected"].values()))
    seal_dependencies_valid = (seal_path.exists() and seal.get("status") == "FROZEN_BEFORE_TEST" and
                               seal.get("validation_selection_sha256") ==
                               sha256(ROOT / "analysis/validation_selection.json") and
                               seal.get("validation_summary_sha256") ==
                               sha256(ROOT / "analysis/validation_summary.csv") and
                               seal.get("training_integrity_sha256") ==
                               sha256(ROOT / "analysis/training_integrity.json") and
                               seal.get("validation_cache_integrity_sha256") ==
                               sha256(ROOT / "analysis/validation_cache_integrity.json") and
                               seal.get("official_i_sha256") == cfg["official_i_sha256"] and
                               seal.get("official_p_sha256") == cfg["official_p_sha256"])

    bootstrap_complete = []
    for qp in cfg["qps"]:
        path = ROOT / f"analysis/fid_kid_bootstrap_qp{qp}.json"
        valid = False
        if path.exists():
            bootstrap_payload = read_json(path)
            rows = bootstrap_payload.get("results", [])
            valid = (bootstrap_payload.get("physical_gpu") in cfg["allowed_physical_gpus"] and
                     len(rows) == 7 and {row["branch"] for row in rows} ==
                     {"m0", "m1", "m2", "m3", "m4", "m5", "m6"} and
                     all(row["fid_video_bootstrap"]["repetitions"] == cfg["bootstrap_repetitions"] and
                         row["kid_video_bootstrap"]["repetitions"] == cfg["bootstrap_repetitions"]
                         for row in rows if row["branch"] != "m0"))
        bootstrap_complete.append(valid)

    required_analysis = ("validation_summary.csv", "test_summary.csv", "per_video_qp.csv",
                         "sequence_thirds.csv", "complexity.json", "fid_kid_per_qp.json")
    analysis_exists = all((ROOT / "analysis" / name).exists() for name in required_analysis)
    summaries_valid = False
    bootstrap_aggregate_valid = False
    complexity_valid = False
    if analysis_exists:
        validation_rows = read_csv(ROOT / "analysis/validation_summary.csv")
        test_rows = read_csv(ROOT / "analysis/test_summary.csv")
        thirds_rows = read_csv(ROOT / "analysis/sequence_thirds.csv")
        summaries_valid = (len(validation_rows) == 30 and len(test_rows) == 35 and
                           len(thirds_rows) == 21 and
                           {row["branch"] for row in test_rows} ==
                           {"m0", "m1", "m2", "m3", "m4", "m5", "m6"})
        bootstrap_rows = read_json(ROOT / "analysis/fid_kid_per_qp.json").get("results", [])
        bootstrap_aggregate_valid = (len(bootstrap_rows) == 28 and
                                     all((row["branch"] == "m0" or
                                          ("fid_video_bootstrap" in row and
                                           "kid_video_bootstrap" in row))
                                         for row in bootstrap_rows))
        method_complexity = read_json(ROOT / "analysis/complexity.json").get("methods", {})
        complexity_valid = (set(method_complexity) ==
                            {"m0", "m1", "m2", "m3", "m4", "m5", "m6"} and
                            all(row.get("mean_candidate_seconds_per_frame") is not None and
                                row.get("peak_incremental_memory_bytes") is not None
                                for row in method_complexity.values()))
    test_metadata = list((ROOT / "analysis/test_pairs").glob("*.json"))
    visual_dir = ROOT / "outputs/visual_comparisons"
    checks = {
        **{f"{name}_pass": value for name, value in integrity.items()},
        "official_i_hash_exact": sha256(cfg["official_i_checkpoint"]) == cfg["official_i_sha256"],
        "official_p_hash_exact": sha256(cfg["official_p_checkpoint"]) == cfg["official_p_sha256"],
        "historical_v8_hash_exact": sha256(cfg["historical_v8_checkpoint"]) == cfg["historical_v8_sha256"],
        "frozen_checkpoint_hashes_exact": frozen_hashes,
        "seal_dependencies_unchanged": seal_dependencies_valid,
        "selection_precedes_all_48_test_pairs": (seal_path.exists() and len(test_metadata) == 48 and
                                                  all(path.stat().st_mtime_ns >= seal_path.stat().st_mtime_ns
                                                      for path in test_metadata)),
        "bootstrap_all_qps_complete": all(bootstrap_complete),
        "required_analysis_exists": analysis_exists,
        "summary_structures_exact": summaries_valid,
        "bootstrap_aggregate_exact": bootstrap_aggregate_valid,
        "complexity_complete": complexity_valid,
        "four_preregistered_visuals": (visual_dir.exists() and
                                        len(list(visual_dir.glob("*_qp2_frame*.png"))) == 4),
        "official_tracked_source_unchanged": subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", "src", "checkpoints"],
            cwd=REPO).returncode == 0,
    }
    payload = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    write_json(ROOT / "analysis/final_integrity.json", payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
