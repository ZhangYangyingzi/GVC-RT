#!/usr/bin/env python3
"""Prepare the locked VIRAT frame cohort for the public GVC-RT checkpoint run."""

from __future__ import annotations

import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
WORKSPACE = REPO.parents[1]
SOURCE_DIR = WORKSPACE / "Kitware_data/videos_original"
GOP_MANIFEST = (
    WORKSPACE
    / "Projects/cosmos-predict1/upsample_LUVE/stage58_480p"
    / "stage00_protocol_data/output_v1/gop_manifest.jsonl"
)
OUTPUT = ROOT / "input_frames"

RANDOM_VIDEOS = (
    "VIRAT_S_010108_01_000570_000718",
    "VIRAT_S_010205_01_000207_000288",
    "VIRAT_S_010111_02_000198_000310",
    "VIRAT_S_010003_01_000111_000137",
    "VIRAT_S_010200_02_000349_000398",
)
SPECIAL_SAMPLES = (
    "VIRAT_S_010200_03_000470_000567_17fps_gop_000000",
    "VIRAT_S_000201_03_000640_000672_17fps_gop_000000",
    "VIRAT_S_010200_01_000254_000322_17fps_gop_000001",
)


def read_manifest_records() -> dict[str, dict]:
    wanted = set(SPECIAL_SAMPLES)
    records = {}
    with GOP_MANIFEST.open(encoding="ascii") as stream:
        for line in stream:
            row = json.loads(line)
            if row["sample_id"] in wanted:
                records[row["sample_id"]] = row
    if set(records) != wanted:
        raise RuntimeError("special GOP records are incomplete")
    return records


def probe(path: Path) -> tuple[int, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if (width, height) != (1280, 720):
        raise RuntimeError(f"expected 1280x720 source, got {width}x{height}: {path}")
    if abs(fps - 23.97) < 0.01:
        return width, height, 2397, 100
    if abs(fps - 30.0) < 0.01:
        return width, height, 30, 1
    raise RuntimeError(f"unexpected source FPS {fps}: {path}")


def decode_indices(path: Path, indices: list[int]) -> list:
    capture = cv2.VideoCapture(str(path))
    frames = []
    wanted = iter(indices)
    target = next(wanted, None)
    frame_index = 0
    while target is not None:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"source ended before frame {target}: {path}")
        if frame_index == target:
            frames.append(frame)
            target = next(wanted, None)
        frame_index += 1
    capture.release()
    return frames


def write_sequence(name: str, frames: list, width: int, height: int) -> None:
    directory = OUTPUT / name
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames, start=1):
        if (frame.shape[1], frame.shape[0]) != (width, height):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        if not cv2.imwrite(str(directory / f"im{index:05d}.png"), frame):
            raise RuntimeError(f"failed to write frame {index} for {name}")


def main() -> None:
    records = read_manifest_records()
    prepared = []

    for video_id in RANDOM_VIDEOS:
        source = SOURCE_DIR / f"{video_id}.mp4"
        _, _, fps_num, fps_den = probe(source)
        frames = decode_indices(source, list(range(96)))
        for label, width, height, out_num, out_den in (
            ("720p", 1280, 720, fps_num, fps_den),
            ("480p", 854, 480, 20, 1),
        ):
            name = f"random__{video_id}__{label}"
            write_sequence(name, frames, width, height)
            prepared.append({
                "sequence": name, "group": "random", "video_id": video_id,
                "source": str(source), "source_frame_indices": list(range(96)),
                "frames": 96, "width": width, "height": height,
                "fps_num": out_num, "fps_den": out_den,
            })

    for sample_id in SPECIAL_SAMPLES:
        record = records[sample_id]
        source = Path(record["source_path"])
        probe(source)
        indices = record["source_frame_indices"]
        frames = decode_indices(source, indices)
        for label, width, height, out_num, out_den in (
            ("720p", 1280, 720, 17, 1),
            ("480p", 854, 480, 20, 1),
        ):
            name = f"specified__{sample_id}__{label}"
            write_sequence(name, frames, width, height)
            prepared.append({
                "sequence": name, "group": "specified", "video_id": sample_id,
                "source": str(source), "source_frame_indices": indices,
                "frames": 33, "width": width, "height": height,
                "fps_num": out_num, "fps_den": out_den,
            })

    sequences = {
        row["sequence"]: {
            "width": row["width"], "height": row["height"], "frames": row["frames"],
            "fps_num": row["fps_num"], "fps_den": row["fps_den"], "intra_period": -1,
        }
        for row in prepared
    }
    config = {
        "root_path": str(ROOT),
        "test_classes": {
            "VIRAT": {"test": 1, "base_path": "input_frames", "src_type": "png", "sequences": sequences}
        },
    }
    (ROOT / "cohort.json").write_text(json.dumps(prepared, indent=2) + "\n", encoding="ascii")
    (ROOT / "test_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="ascii")
    print(f"prepared {len(prepared)} sequences in {OUTPUT}")


if __name__ == "__main__":
    main()
