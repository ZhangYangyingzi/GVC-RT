#!/usr/bin/env python3
"""Package the exact benchmark input frames as reference viewing MP4s."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VIDEOS = ROOT / "visualizations"


def original_video_path(row: dict) -> Path:
    fps = row["fps_num"] / row["fps_den"]
    fps_label = f"{fps:.2f}".rstrip("0").rstrip(".")
    return VIDEOS / (
        f"{row['sequence']}_{row['width']}x{row['height']}_{fps_label}fps_original.mp4"
    )


def main() -> None:
    rows = json.loads((ROOT / "cohort.json").read_text())
    if len(rows) != 16:
        raise RuntimeError(f"expected 16 source sequences, got {len(rows)}")
    for row in rows:
        frame_dir = ROOT / "input_frames" / row["sequence"]
        frames = sorted(frame_dir.glob("im*.png"))
        if len(frames) != row["frames"]:
            raise RuntimeError(f"expected {row['frames']} source frames in {frame_dir}")
        output = original_video_path(row)
        command = [
            "ffmpeg", "-y", "-v", "error",
            "-framerate", f"{row['fps_num']}/{row['fps_den']}",
            "-i", str(frame_dir / "im%05d.png"),
            "-frames:v", str(row["frames"]), "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        ]
        subprocess.run(command, check=True)
        print(output, flush=True)


if __name__ == "__main__":
    main()
