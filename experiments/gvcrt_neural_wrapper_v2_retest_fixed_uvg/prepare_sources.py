#!/usr/bin/env python3
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V2 = REPO / "experiments/gvcrt_neural_wrapper_v2_joint"
SPLIT = REPO / "expericent_generation_input/expericent_generator_aware_latent_distortion_v11/data_split.json"
FRAME_COUNT = 64
WIDTH, HEIGHT = 1920, 1080


def tag(video):
    return f"{video['dataset']}_{int(video['video_id']):02d}"


def ffmpeg_decode(video):
    command = ["ffmpeg", "-v", "error"]
    if video["dataset"] == "uvg":
        command += ["-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size",
                    f"{WIDTH}x{HEIGHT}", "-framerate", str(video["fps"])]
    command += ["-i", video["source_path"], "-frames:v", str(FRAME_COUNT),
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.check_output(command)
    expected = FRAME_COUNT * WIDTH * HEIGHT * 3
    if len(raw) != expected:
        raise RuntimeError(f"short decode for {video['source_path']}: {len(raw)} != {expected}")
    return np.frombuffer(raw, np.uint8).reshape(FRAME_COUNT, HEIGHT, WIDTH, 3).copy()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    old_manifest = json.loads((V2 / "test_manifest.json").read_text())
    split = json.loads(SPLIT.read_text())
    excluded = set(json.loads((V2 / "train_manifest.json").read_text())["test_video_source_sha256_excluded"])
    videos = []
    for old in old_manifest["videos"]:
        item = dict(old)
        item["test_group"] = "original_v2_test"
        videos.append(item)
    existing_hashes = {item["source_sha256"] for item in videos}
    additions = []
    for index, item in enumerate(split["splits"]["validation"]):
        if len(additions) >= 5:
            break
        if item["file_sha256"] in excluded or item["file_sha256"] in existing_hashes:
            continue
        additions.append({
            "dataset": "fresh_ulong", "video_id": 100 + index,
            "name": Path(item["filename"]).stem,
            "source_path": item["local_path"], "source_sha256": item["file_sha256"],
            "fps": float(item["probe"]["avg_frame_rate"].split("/")[0]) /
                   float(item["probe"]["avg_frame_rate"].split("/")[1]),
            "start_frame": 0, "number_of_frames": FRAME_COUNT,
            "test_group": "added_ulong_validation",
            "split_index": item["split_index"], "training_excluded_by": "v11 validation split",
        })
    if len(additions) < 5:
        raise RuntimeError("could not select five non-training added U-Long videos")
    videos.extend(additions)
    manifest = {
        "schema": "gvcrt_neural_wrapper_v2_retest_fixed_uvg_manifest",
        "source_manifest": str(V2 / "test_manifest.json"),
        "training_split": str(SPLIT), "start_frame": 0,
        "number_of_frames": FRAME_COUNT, "videos": videos,
        "train_test_intersection_zero": True,
        "added_ulong_count": len(additions),
    }
    (ROOT / "test_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    source_root = ROOT / "source_frames_fixed"
    rows, verification = [], []
    for video in videos:
        frames = ffmpeg_decode(video)
        directory = source_root / tag(video)
        directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        for index, frame in enumerate(frames):
            digest.update(frame.tobytes())
            Image.fromarray(frame, "RGB").save(directory / f"im{index + 1}.png", compress_level=1)
        sample_ok = True
        for index in (0, 1, 8, 16):
            saved = np.asarray(Image.open(directory / f"im{index + 1}.png").convert("RGB"), dtype=np.uint8)
            sample_ok = sample_ok and np.array_equal(saved, frames[index])
        conversion = "ffmpeg rawvideo yuv420p->rgb24" if video["dataset"] == "uvg" else "ffmpeg yuv420p->rgb24"
        rows.append({
            "dataset": video["dataset"], "video": video["name"], "video_id": video["video_id"],
            "test_group": video["test_group"], "original_wrong_source_path":
                str(REPO / "experiments/gvcrt_vs_dcvc_rt_matched_rate/source_frames" / tag(video))
                if video["dataset"] == "uvg" else "not_applicable",
            "fixed_source_path": str(directory), "source_path": video["source_path"],
            "conversion_method": conversion,
            "ffmpeg_command": "ffmpeg -f rawvideo -pixel_format yuv420p -video_size 1920x1080 -framerate 120 -i INPUT -pix_fmt rgb24 PIPE"
                if video["dataset"] == "uvg" else "ffmpeg -i INPUT -pix_fmt rgb24 PIPE",
            "color_matrix": "BT.601" if video["dataset"] == "uvg" else "metadata/ffmpeg",
            "color_range": "limited" if video["dataset"] == "uvg" else "metadata/ffmpeg",
            "frames": FRAME_COUNT, "sample_frame_verification": "PASS" if sample_ok else "FAIL",
            "rgb_tensor_sha256": digest.hexdigest(), "source_file_sha256": sha256_file(video["source_path"]),
        })
        verification.append(sample_ok)
    (ROOT / "uvg_source_fix_audit.json").write_text(json.dumps({
        "status": "PASS" if all(verification) else "FAIL", "fixed_source_isolated": True,
        "conversion_backend": "ffmpeg", "uvg_color_matrix": "BT.601",
        "uvg_color_range": "limited", "records": rows,
    }, indent=2) + "\n")
    print(json.dumps({"videos": len(videos), "added_ulong": len(additions),
                      "sample_verification": all(verification)}, indent=2))


if __name__ == "__main__":
    main()
