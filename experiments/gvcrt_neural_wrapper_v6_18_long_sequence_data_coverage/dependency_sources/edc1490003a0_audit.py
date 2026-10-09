#!/usr/bin/env python3
import json

from bridge_adapter import FeatureAdapter
from common import REPO, ROOT, config, sha256, split, write_json
from current_only_refiner import CurrentOnlyRefiner
from dataset import historical_paths
from rgb_postprocessor import RGBPostprocessor
from temporal_refiner import TemporalRefiner


def main():
    cfg, data = config(), split()
    excluded, sources = historical_paths()
    assigned = [video["path"] for group in ("train", "validation", "test") for video in data[group]]
    counts = {"m2": sum(p.numel() for p in FeatureAdapter().parameters()),
              "m3": sum(p.numel() for p in CurrentOnlyRefiner().parameters()),
              "m4": sum(p.numel() for p in TemporalRefiner().parameters()),
              "m5": sum(p.numel() for p in RGBPostprocessor().parameters())}
    checks = {
        "repository_commit_exact": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True,
            text=True).stdout.strip() == cfg["repository_commit"],
        "official_i_hash_exact": sha256(cfg["official_i_checkpoint"]) == cfg["official_i_sha256"],
        "official_p_hash_exact": sha256(cfg["official_p_checkpoint"]) == cfg["official_p_sha256"],
        "historical_v8_hash_exact": sha256(cfg["historical_v8_checkpoint"]) == cfg["historical_v8_sha256"],
        "split_cardinality_16_8_12": [len(data[key]) for key in ("train", "validation", "test")] == [16, 8, 12],
        "split_unique": len(assigned) == len(set(assigned)) == 36,
        "no_historical_overlap": not bool(set(assigned) & excluded),
        "native_1920x1080": all(video["width"] == 1920 and video["height"] == 1080
                                  for group in ("train", "validation", "test") for video in data[group]),
        "at_least_96_frames": all(video["frame_count"] >= 96
                                    for group in ("train", "validation", "test") for video in data[group]),
        "m2_within_5_percent": abs(counts["m2"] - 19474) / 19474 <= 0.05,
        "m3_within_5_percent": abs(counts["m3"] - 19474) / 19474 <= 0.05,
        "m4_exact_19474": counts["m4"] == 19474,
        "m5_within_5_percent": abs(counts["m5"] - 19474) / 19474 <= 0.05,
        "zero_additional_bits": cfg["additional_transmitted_bits"] == 0}
    payload = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
               "bridge_modules": ["recon_generation_net.mlp.0", "recon_generation_net.mlp.1",
                                  "recon_generation_net.mlp.2", "recon_generation_net.mlp.3"],
               "official_bridge_parameters": 801124, "matched_parameter_counts": counts,
               "excluded_historical_videos": len(excluded), "exclusion_sources": sources,
               "candidate_enters_official_dpb": False, "sidecar_required": True}
    write_json(ROOT / "analysis/interface_audit.json", payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
