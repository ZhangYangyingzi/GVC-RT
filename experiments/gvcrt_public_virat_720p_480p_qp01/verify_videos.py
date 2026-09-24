#!/usr/bin/env python3
"""Verify reconstructed MP4 metadata and measured GVC-RT rate labels."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from make_original_videos import original_video_path


ROOT = Path(__file__).resolve().parent
VIDEOS = ROOT / "visualizations"
BITSTREAMS = ROOT / "bitstreams/VIRAT"


def probe(path: Path) -> dict:
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
        "-of", "json", str(path),
    ])
    return json.loads(output)["streams"][0]


def main() -> None:
    cohort = {row["sequence"]: row for row in json.loads((ROOT / "cohort.json").read_text())}
    manifest = []
    for filename in ("results.json", "results_qp23.json"):
        results = json.loads((ROOT / filename).read_text())["VIRAT"]
        for sequence, points in results.items():
            meta = cohort[sequence]
            fps = meta["fps_num"] / meta["fps_den"]
            fps_label = f"{fps:.2f}".rstrip("0").rstrip(".")
            for result in points.values():
                qp = int(result["qp_i"])
                kbps = float(result["avg_kbps"])
                name = (
                    f"{sequence}_{meta['width']}x{meta['height']}_{fps_label}fps_"
                    f"qp{qp}_{kbps:.3f}kbps.mp4"
                )
                video = VIDEOS / name
                bitstream = BITSTREAMS / f"{sequence}_q{qp}.bin"
                expected_kbps = bitstream.stat().st_size * 8 * fps / meta["frames"] / 1000
                if abs(expected_kbps - kbps) > 1e-5:
                    raise RuntimeError(f"GVC-RT bitrate mismatch: {bitstream}")
                stream = probe(video)
                expected = (meta["width"], meta["height"], meta["frames"],
                            f"{meta['fps_num']}/{meta['fps_den']}")
                actual = (stream["width"], stream["height"], int(stream["nb_frames"]),
                          stream["r_frame_rate"])
                if actual != expected:
                    raise RuntimeError(f"MP4 metadata mismatch: {video}: {actual} != {expected}")
                manifest.append({
                    "group": meta["group"], "video": meta["video_id"],
                    "resolution": f"{meta['width']}x{meta['height']}",
                    "fps": fps, "frames": meta["frames"], "qp": qp,
                    "gvcrt_bitrate_kbps": kbps, "path": str(video.resolve()),
                })
    originals = []
    for meta in cohort.values():
        video = original_video_path(meta)
        stream = probe(video)
        expected = (meta["width"], meta["height"], meta["frames"],
                    f"{meta['fps_num']}/{meta['fps_den']}")
        actual = (stream["width"], stream["height"], int(stream["nb_frames"]),
                  stream["r_frame_rate"])
        if actual != expected:
            raise RuntimeError(f"original MP4 metadata mismatch: {video}: {actual} != {expected}")
        originals.append({
            "group": meta["group"], "video": meta["video_id"],
            "resolution": f"{meta['width']}x{meta['height']}",
            "fps": meta["fps_num"] / meta["fps_den"], "frames": meta["frames"],
            "source": meta["source"], "path": str(video.resolve()),
        })
    found = set(VIDEOS.glob("*.mp4"))
    expected_paths = {Path(row["path"]) for row in manifest + originals}
    if len(manifest) != 64 or len(originals) != 16 or len(expected_paths) != 80 or found != expected_paths:
        raise RuntimeError(f"expected 80 matching MP4s, got {len(expected_paths)} records and {len(found)} files")
    (VIDEOS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    (VIDEOS / "originals_manifest.json").write_text(json.dumps(originals, indent=2) + "\n", encoding="ascii")
    print(f"Verified {len(manifest)} reconstructions and {len(originals)} originals")


if __name__ == "__main__":
    main()
