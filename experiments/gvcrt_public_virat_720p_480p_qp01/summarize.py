#!/usr/bin/env python3
"""Create per-video and arithmetic-mean metric tables."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    cohort = {row["sequence"]: row for row in json.loads((ROOT / "cohort.json").read_text())}
    rows = []
    for filename in ("results.json", "results_qp23.json"):
        results = json.loads((ROOT / filename).read_text())["VIRAT"]
        for sequence, rate_results in results.items():
            meta = cohort[sequence]
            for result in rate_results.values():
                rows.append({
                    "group": meta["group"],
                    "video": meta["video_id"],
                    "resolution": f"{meta['width']}x{meta['height']}",
                    "frames": meta["frames"],
                    "fps": meta["fps_num"] / meta["fps_den"],
                    "qp": result["qp_i"],
                    "bitrate_kbps": result["avg_kbps"],
                    "psnr_db": result["avg_psnr"],
                    "lpips": result["avg_lpips"],
                    "dists": result["avg_dists"],
                })
    rows.sort(key=lambda row: (row["resolution"], row["qp"], row["group"], row["video"]))
    if len(rows) != 64:
        raise RuntimeError(f"expected 64 metric rows, got {len(rows)}")
    identities = {(row["group"], row["video"], row["resolution"], row["qp"]) for row in rows}
    if len(identities) != 64:
        raise RuntimeError("duplicate video/resolution/QP metric row")
    for row in rows:
        for key in ("bitrate_kbps", "psnr_db", "lpips", "dists"):
            if not math.isfinite(row[key]):
                raise RuntimeError(f"non-finite {key} in {row['video']}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["resolution"], row["qp"])].append(row)
    averages = []
    for (resolution, qp), group in sorted(grouped.items()):
        if len(group) != 8:
            raise RuntimeError(f"expected 8 videos for {resolution}/QP{qp}, got {len(group)}")
        averages.append({
            "resolution": resolution,
            "qp": qp,
            "video_count": len(group),
            "bitrate_kbps": mean(row["bitrate_kbps"] for row in group),
            "psnr_db": mean(row["psnr_db"] for row in group),
            "lpips": mean(row["lpips"] for row in group),
            "dists": mean(row["dists"] for row in group),
        })

    with (ROOT / "metrics_per_video.csv").open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (ROOT / "metrics_average.csv").open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=averages[0].keys())
        writer.writeheader()
        writer.writerows(averages)
    with (ROOT / "metrics_qp23_per_video.csv").open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(row for row in rows if row["qp"] in (2, 3))
    with (ROOT / "metrics_qp23_average.csv").open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=averages[0].keys())
        writer.writeheader()
        writer.writerows(row for row in averages if row["qp"] in (2, 3))
    (ROOT / "summary.json").write_text(
        json.dumps({"per_video": rows, "averages": averages}, indent=2) + "\n", encoding="ascii"
    )

    report = [
        "# Public GVC-RT VIRAT QP 0-3 benchmark",
        "",
        "## Protocol",
        "",
        "- Public checkpoints: `checkpoints/GVC-RT_I.pt` and `checkpoints/GVC-RT_P.pt`.",
        f"- I checkpoint SHA-256: `{sha256(REPO / 'checkpoints/GVC-RT_I.pt')}`.",
        f"- P checkpoint SHA-256: `{sha256(REPO / 'checkpoints/GVC-RT_P.pt')}`.",
        "- Five randomly selected 1280x720 VIRAT videos use source frames 0-95.",
        "- Three requested GOPs use their exact 33 source-frame indices from the locked Stage58 manifest.",
        "- 720p uses 1280x720 and the cohort frame rate (23.97 fps random, 17 fps requested GOPs).",
        "- 480p uses the same source images resized with Lanczos to 854x480 and a 20 fps rate clock.",
        "- Codec input is replicate-padded to a multiple of 64; metrics are evaluated only on valid pixels.",
        "- Bitrate includes the complete `.bin` stream and SPS overhead.",
        "- PSNR is RGB PSNR; LPIPS is AlexNet v0.1 with `[0,1]` normalization; DISTS uses RGB `[0,1]`.",
        "- Averages below are unweighted arithmetic means across the eight videos.",
        "",
        "## Reconstructed videos",
        "",
        "- All 64 reconstructed MP4s are in `visualizations/`; `visualizations/manifest.json` maps each file to its source, resolution, FPS, QP and measured GVC-RT bitrate.",
        "- The 16 matching original-reference MP4s are in the same folder; `visualizations/originals_manifest.json` lists their source and frame counts.",
        "- Each filename ends with `<resolution>_<fps>fps_qp<value>_<GVC-RT-kbps>kbps.mp4`.",
        "- MP4 uses H.264 CRF 12 for viewing. The kbps in its filename is the GVC-RT `.bin` bitrate, not the MP4 container bitrate.",
        "",
        "## Averages",
        "",
        "| Resolution | QP | Videos | kbps | PSNR (dB) | LPIPS | DISTS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in averages:
        report.append(
            f"| {row['resolution']} | {row['qp']} | {row['video_count']} | "
            f"{row['bitrate_kbps']:.3f} | {row['psnr_db']:.4f} | "
            f"{row['lpips']:.6f} | {row['dists']:.6f} |"
        )
    report.extend([
        "",
        "## Per-video results",
        "",
        "| Group | Video | Resolution | Frames | FPS | QP | kbps | PSNR (dB) | LPIPS | DISTS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        report.append(
            f"| {row['group']} | {row['video']} | {row['resolution']} | {row['frames']} | "
            f"{row['fps']:.2f} | {row['qp']} | {row['bitrate_kbps']:.3f} | "
            f"{row['psnr_db']:.4f} | {row['lpips']:.6f} | {row['dists']:.6f} |"
        )
    (ROOT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="ascii")
    print(json.dumps(averages, indent=2))


if __name__ == "__main__":
    main()
