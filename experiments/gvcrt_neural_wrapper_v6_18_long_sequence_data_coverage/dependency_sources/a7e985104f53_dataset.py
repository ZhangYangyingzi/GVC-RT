#!/usr/bin/env python3
import json
from pathlib import Path

from common import RESEARCH, ROOT, V6, write_json


def collect_paths(value, destination):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and value["path"].endswith(".mp4"):
            destination.add(value["path"])
        for child in value.values():
            collect_paths(child, destination)
    elif isinstance(value, list):
        for child in value:
            collect_paths(child, destination)


def historical_paths():
    paths, sources = set(), []
    for stage in sorted(RESEARCH.iterdir()):
        if stage == ROOT or not stage.is_dir():
            continue
        candidates = list(stage.glob("v*_split.json")) + list(stage.glob("selected_videos.json"))
        config_path = stage / "config.json"
        if config_path.exists():
            candidates.append(config_path)
        for path in candidates:
            try:
                payload = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            before = len(paths)
            collect_paths(payload, paths)
            if len(paths) != before:
                sources.append(str(path))
    return paths, sorted(set(sources))


def main():
    manifest = json.loads((V6 / "dataset_manifest.json").read_text())
    excluded, sources = historical_paths()
    eligible = sorted((video for video in manifest["videos"] if video.get("valid") and
                       video.get("width") == 1920 and video.get("height") == 1080 and
                       video.get("frame_count", 0) >= 96 and video["path"] not in excluded),
                      key=lambda video: video["path"])
    count = 36
    if len(eligible) < count:
        raise RuntimeError(f"Need {count} historically clean videos, found {len(eligible)}")
    indexes = [round(index * (len(eligible) - 1) / (count - 1)) for index in range(count)]
    selected = [eligible[index] | {"selection_index": index, "source": "V9_FRESH"}
                for index in indexes]
    payload = {
        "selection_rule": "Exclude every path found in V1-V8 split, selection, or config artifacts; filter valid native 1920x1080 videos with >=96 frames; lexicographically sort; choose 36 evenly spaced entries; assign 16/8/12.",
        "eligible_video_count": len(eligible), "excluded_historical_video_count": len(excluded),
        "exclusion_sources": sources, "excluded_historical_paths": sorted(excluded),
        "train": selected[:16], "validation": selected[16:24],
        "test": [video | {"source": "V9_FRESH_TEST"} for video in selected[24:]],
        "frames_per_video": 96, "requested_qps": [0, 1, 2, 3], "intra_period": -1}
    assigned = [video["path"] for group in ("train", "validation", "test") for video in payload[group]]
    if len(assigned) != 36 or len(set(assigned)) != 36 or set(assigned) & excluded:
        raise RuntimeError("V9 split isolation failure")
    write_json(ROOT / "v9_split.json", payload)
    write_json(ROOT / "dataset_manifest.json", {
        "source_manifest": str(V6 / "dataset_manifest.json"),
        "source_manifest_sha256": __import__("common").sha256(V6 / "dataset_manifest.json"),
        "dataset_root": manifest["dataset_root"], "source_candidate_count": manifest["candidate_count"],
        "eligible_after_v1_v8_exclusion": len(eligible), "selected_videos": selected})
    print(json.dumps({"train": 16, "validation": 8, "test": 12,
                      "eligible": len(eligible), "excluded": len(excluded)}, indent=2))


if __name__ == "__main__":
    main()
