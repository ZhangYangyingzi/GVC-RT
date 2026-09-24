"""Freeze source-group split and consecutive native frames before model experiments."""
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration", "-of", "json", str(path)]))["streams"][0]


def main():
    cfg = json.loads((HERE / "config.json").read_text())
    dest = HERE / "data"
    dest.mkdir(exist_ok=False)
    paths = subprocess.check_output(["rg", "--files", cfg["dataset_root"], "-g", "*.mp4"], text=True).splitlines()
    # Prefer clips_long_1920, never mix different exports of the same source.
    paths = [p for p in paths if "/clips_long_1920/" in p]
    if not paths:
        raise RuntimeError("No clips_long_1920 mp4 found")
    unique = {}
    for p in sorted(paths):
        # clips_long_1920 names are UUIDs. No parent-source mapping is supplied.
        unique.setdefault(Path(p).stem, p)
    ordered = sorted(unique, key=lambda k: hashlib.sha256(f'{cfg["seed"]}:{k}'.encode()).hexdigest())
    selections = {"val": [], "train": []}
    rejected = []
    for group in ordered:
        split = "val" if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5 == 0 else "train"
        cap = 6 if split == "val" else cfg["training"]["generalization_videos"]
        if len(selections[split]) >= cap:
            continue
        p = unique[group]
        metadata = probe(p)
        if [metadata["height"], metadata["width"]] != cfg["valid_hw"] or int(metadata.get("nb_frames", 0)) < cfg["frames"]:
            rejected.append({"path": p, "metadata": metadata})
            continue
        # Read actual frame timestamps, rather than assuming the nominal frame rate.
        pts_info = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#40",
            "-show_entries", "frame=best_effort_timestamp_time,pkt_duration_time", "-of", "json", p]))["frames"]
        if len(pts_info) < cfg["frames"]+1:
            raise RuntimeError(f"Insufficient frame timestamps: {p}")
        pts = [float(f["best_effort_timestamp_time"]) for f in pts_info[:cfg["frames"]+1]]
        if not all(b > a for a,b in zip(pts, pts[1:])):
            raise RuntimeError("Non-monotone timestamps")
        entry = {"path": p, "group": group, "video_id": Path(p).stem, "split": split,
                 "metadata": metadata, "frame_start": 0, "frames": cfg["frames"],
                 "pts_seconds": pts[:-1], "duration_seconds": pts[-1]-pts[0],
                 "causal_origin": "first frame of supplied source clip; one I then consecutive P; no seek/reset",
                 "valid_hw": cfg["valid_hw"], "padded_hw": cfg["padded_hw"]}
        selections[split].append(entry)
        if len(selections["val"]) == 6 and len(selections["train"]) == cfg["training"]["generalization_videos"]:
            break
    if len(selections["val"]) != 6 or len(selections["train"]) < 2:
        raise RuntimeError("Insufficient eligible videos")
    # First two training source groups are fixed without viewing codec outputs.
    selections["train2"] = [v["video_id"] for v in selections["train"][:2]]
    manifest = {"seed": cfg["seed"], "source_root": cfg["dataset_root"],
                "split_provenance": "No applicable pre-existing UltraVideo split located; deterministic UUID-video split, no final test used. Parent-source identity unavailable; cross-clip source overlap cannot be excluded.",
                "selection": "SHA256(seed:source_group) order; SHA256(group)%5==0 validation; other groups training; native 1080p only",
                "rejected_before_experiment": rejected, **selections}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for entry in selections["train"] + selections["val"]:
        folder = dest / entry["video_id"]
        folder.mkdir()
        cap = cv2.VideoCapture(entry["path"])
        previous = None
        motion = []
        hashes = []
        for frame in range(cfg["frames"]):
            ok, bgr = cap.read()
            if not ok or list(bgr.shape[:2]) != cfg["valid_hw"]:
                raise RuntimeError("Frame read/shape failed")
            filename = folder / f"{frame:04d}.png"
            if not cv2.imwrite(str(filename), bgr):
                raise RuntimeError("PNG write failed")
            hashes.append(hashlib.sha256(filename.read_bytes()).hexdigest())
            gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (240, 135)).astype(np.float32)
            if previous is not None:
                motion.append(float(np.abs(gray-previous).mean()))
            previous = gray
        cap.release()
        entry["png_sha256"] = hashes
        entry["source_motion_mean_abs_8bit_240x135"] = float(np.mean(motion))
        print(entry["split"], entry["video_id"], entry["metadata"], "motion", np.mean(motion), flush=True)
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
