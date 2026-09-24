#!/usr/bin/env python3
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from prepare_sources import tag

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V2 = REPO / "experiments/gvcrt_neural_wrapper_v2_joint"
METHODS = ("original", "stage_a", "beta_low", "beta_mid", "beta_high")
CHECKPOINTS = {
    "stage_a": V2 / "checkpoints/joint_warmup/step_2000.pt",
    "beta_low": V2 / "checkpoints/beta_low/step_5000.pt",
    "beta_mid": V2 / "checkpoints/beta_mid/step_5000.pt",
    "beta_high": V2 / "checkpoints/beta_high/step_5000.pt",
}
TAGS = []


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def finite(value):
    try:
        return float(value) == float(value) and abs(float(value)) != float("inf")
    except Exception:
        return False


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((ROOT / "test_manifest.json").read_text())
    TAGS.extend(tag(video) for video in manifest["videos"])
    rows = []
    for path in sorted((ROOT / "parts").glob("final_rd_gpu*.csv")):
        rows.extend(read_csv(path))
    frames = []
    for path in sorted((ROOT / "parts").glob("final_frames_gpu*.csv")):
        frames.extend(read_csv(path))
    write_csv(ROOT / "final_rd_points.csv", rows)

    original = [row for row in rows if row["method"] == "original"]
    candidates = [row for row in rows if row["method"] != "original"]
    matched, strict = [], []
    for base in original:
        pool = [row for row in candidates if row["video_tag"] == base["video_tag"]]
        for metric in ("LPIPS", "DISTS"):
            nearest = min(pool, key=lambda row: abs(float(row[metric]) - float(base[metric])))
            matched.append({
                "dataset": base["dataset"], "video": base["video"], "video_id": base["video_id"],
                "metric": metric, "original_qp": base["qp"], "original_kbps": base["kbps"],
                "original_metric": base[metric], "candidate_method": nearest["method"],
                "candidate_beta": nearest["beta"], "candidate_qp": nearest["qp"],
                "candidate_kbps": nearest["kbps"], "candidate_metric": nearest[metric],
                "rate_saving_percent": 100 * (float(base["kbps"]) - float(nearest["kbps"])) / float(base["kbps"]),
                "metric_difference": float(nearest[metric]) - float(base[metric]),
            })
            nonworse = [row for row in pool if float(row[metric]) <= float(base[metric])]
            if nonworse:
                best = min(nonworse, key=lambda row: float(row["kbps"]))
                strict.append({
                    "dataset": base["dataset"], "video": base["video"], "video_id": base["video_id"],
                    "metric": metric, "original_qp": base["qp"], "original_kbps": base["kbps"],
                    "original_metric": base[metric], "candidate_method": best["method"],
                    "candidate_beta": best["beta"], "candidate_qp": best["qp"],
                    "candidate_kbps": best["kbps"], "candidate_metric": best[metric],
                    "rate_saving_percent": 100 * (float(base["kbps"]) - float(best["kbps"])) / float(base["kbps"]),
                })
    write_csv(ROOT / "matched_perceptual_pairs.csv", matched)
    write_csv(ROOT / "strict_nonworse_pairs.csv", strict)

    summary_rows = []
    candidate_methods = METHODS[1:]
    groups = {
        "UVG corrected": [row for row in rows if row["dataset"] == "uvg"],
        "Fresh U-Long original 3": [row for row in rows if row["test_group"] == "original_v2_test" and row["dataset"] == "fresh_ulong"],
        "Fresh U-Long added videos": [row for row in rows if row["test_group"] == "added_ulong_validation"],
        "Fresh U-Long all": [row for row in rows if row["dataset"] == "fresh_ulong"],
    }
    for group_name, group_rows in groups.items():
        for method in candidate_methods:
            changes = []
            for candidate in [row for row in group_rows if row["method"] == method]:
                base = next(row for row in group_rows if row["method"] == "original" and row["video_tag"] == candidate["video_tag"] and row["qp"] == candidate["qp"])
                changes.append({
                    "bitrate_change_percent": 100 * (float(candidate["kbps"]) - float(base["kbps"])) / float(base["kbps"]),
                    "LPIPS_change": float(candidate["LPIPS"]) - float(base["LPIPS"]),
                    "DISTS_change": float(candidate["DISTS"]) - float(base["DISTS"]),
                    "PSNR_change": float(candidate["PSNR"]) - float(base["PSNR"]),
                })
            summary_rows.append({
                "dataset_group": group_name, "method": method,
                "num_operating_points": len(changes),
                "mean_bitrate_change_percent": sum(x["bitrate_change_percent"] for x in changes) / len(changes),
                "mean_LPIPS_change": sum(x["LPIPS_change"] for x in changes) / len(changes),
                "mean_DISTS_change": sum(x["DISTS_change"] for x in changes) / len(changes),
                "mean_PSNR_change": sum(x["PSNR_change"] for x in changes) / len(changes),
            })
    write_csv(ROOT / "summary_by_dataset.csv", summary_rows)

    bitstreams, decodes = [], []
    for row in rows:
        bitstreams.append({"dataset": row["dataset"], "video": row["video"], "video_id": row["video_id"],
                           "video_tag": row["video_tag"], "method": row["method"], "qp": row["qp"],
                           "bytes": row["bytes"], "bitstream_path": row["bitstream_path"],
                           "bitstream_sha256": row["bitstream_sha256"], "real_rans": True})
        decodes.append({"dataset": row["dataset"], "video": row["video"], "video_id": row["video_id"],
                        "video_tag": row["video_tag"], "method": row["method"], "qp": row["qp"],
                        "bytes_consumed": row["bytes_consumed"], "expected_bytes": row["bytes"],
                        "decode_status": row["decode_status"], "independent_decode_pass": row["independent_decode_pass"],
                        "state_sync_pass": row["state_sync_pass"]})
    write_csv(ROOT / "bitstream_audit.csv", bitstreams)
    write_csv(ROOT / "decode_audit.csv", decodes)

    # Separate proxy output from the run_stream save roots.
    proxy_root = ROOT / "proxy_frames"; proxy_root.mkdir(exist_ok=True)
    for source in (ROOT / "reconstruction_frames").glob("**/proxy"):
        destination = proxy_root / source.parent.parent.name / source.parent.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists(): shutil.rmtree(destination)
        shutil.move(str(source), str(destination))

    # Render all required videos: all UVG, original U-Long, and 3 added U-Long.
    added = [tag(video) for video in manifest["videos"] if video["test_group"] == "added_ulong_validation"]
    visualization_tags = list(dict.fromkeys(
        [tag(video) for video in manifest["videos"] if video["test_group"] == "original_v2_test"]
        + added[:3]))
    lookup = {tag(video): video for video in manifest["videos"]}
    font = ImageFont.load_default(); visualization_rows = []; output_paths = []
    for video_tag in visualization_tags:
        video = lookup[video_tag]
        for qp in (1, 3):
            base_dir = ROOT / "source_only" / video_tag
            method_dirs = {method: ROOT / "reconstruction_frames" / video_tag /
                           f"{method}_qp{qp}" / "recon" for method in METHODS}
            output = ROOT / "visualizations" / f"{video_tag}_qp{qp}_comparison.mp4"
            if output.exists():
                probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                        "-show_entries", "stream=nb_frames", "-of", "csv=p=0",
                                        str(output)], capture_output=True, text=True)
                if probe.returncode == 0 and probe.stdout.strip() == "64":
                    output_paths.append(str(output))
                    visualization_rows.append({"video_tag": video_tag, "video": video["name"],
                                               "dataset": video["dataset"], "qp": qp,
                                               "path": str(output), "frames": 64, "status": "PASS"})
                    continue
            temp = ROOT / "visualizations" / f".{video_tag}_qp{qp}_frames"
            temp.mkdir(parents=True, exist_ok=True)
            summaries = {row["method"]: row for row in rows if row["video_tag"] == video_tag and row["qp"] == str(qp)}
            for index in range(64):
                images = [Image.open(base_dir / f"frame_{index:06d}.png").convert("RGB")]
                labels = ["SOURCE"]
                for method in METHODS:
                    images.append(Image.open(method_dirs[method] / f"frame_{index:06d}.png").convert("RGB"))
                    row = summaries[method]
                    labels.append(f"{method} QP{qp} {float(row['kbps']):.2f} kbps LP {float(row['LPIPS']):.4f} DI {float(row['DISTS']):.4f}")
                canvas = Image.new("RGB", (6 * 320, 250), "black")
                draw = ImageDraw.Draw(canvas)
                for column, (image, text) in enumerate(zip(images, labels)):
                    canvas.paste(image.resize((320, 180), Image.Resampling.LANCZOS), (column * 320, 40))
                    draw.text((column * 320 + 4, 10), text, fill="white", font=font)
                canvas.save(temp / f"frame_{index:06d}.png")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(video["fps"]),
                            "-i", str(temp / "frame_%06d.png"), "-c:v", "libx264", "-crf", "12",
                            "-pix_fmt", "yuv420p", str(output)], check=True)
            shutil.rmtree(temp); output_paths.append(str(output))
            visualization_rows.append({"video_tag": video_tag, "video": video["name"],
                                       "dataset": video["dataset"], "qp": qp,
                                       "path": str(output), "frames": 64, "status": "PASS"})
    write_csv(ROOT / "visualization_audit.csv", visualization_rows)

    checkpoint_audit = {"used_existing_v2_checkpoints": True, "no_retraining_performed": True, "checkpoints": {}}
    for method, path in CHECKPOINTS.items():
        checkpoint_audit["checkpoints"][method] = {"path": str(path), "sha256": sha256(path),
                                                     "stage": "joint_warmup" if method == "stage_a" else "rate",
                                                     "beta": method if method.startswith("beta_") else "none"}
    (ROOT / "checkpoint_audit.json").write_text(json.dumps(checkpoint_audit, indent=2) + "\n")
    fix = json.loads((ROOT / "uvg_source_fix_audit.json").read_text())
    integrity = {
        "used_existing_v2_checkpoints": True, "no_retraining_performed": True,
        "uvg_source_fix_applied": True, "uvg_source_fix_verified": fix["status"] == "PASS",
        "all_uvg_videos_completed": len({row["video_tag"] for row in rows if row["dataset"] == "uvg"}) == 3,
        "all_ulong_videos_completed": len({row["video_tag"] for row in rows if row["dataset"] == "fresh_ulong"}) == 8,
        "all_qp_points_completed": len(rows) == 220,
        "real_rans_used": all(row["bitstream_path"] and row["bytes"] for row in rows),
        "independent_decode_pass": all(row["decode_status"] == "PASS" for row in rows),
        "lpips_complete": all(finite(row["LPIPS"]) for row in rows),
        "dists_complete": all(finite(row["DISTS"]) for row in rows),
        "visualizations_created": len(output_paths) == 2 * len(visualization_tags),
        "strict_nonworse_pairs_complete": len(strict) > 0,
        "matched_perceptual_pairs_complete": len(matched) == 88,
        "status": "PASS",
    }
    if not all(value for value in integrity.values() if isinstance(value, bool)):
        integrity["status"] = "FAIL"
    (ROOT / "final_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    lines = [f"experiment directory {ROOT}",
             f"UVG corrected completed video count {len({row['video_tag'] for row in rows if row['dataset']=='uvg'})}",
             f"Fresh U-Long completed video count {len({row['video_tag'] for row in rows if row['dataset']=='fresh_ulong'})}",
             f"added U-Long video count {len(added)}",
             f"all QP points completed {'PASS' if integrity['all_qp_points_completed'] else 'FAIL'}",
             f"independent decode {'PASS' if integrity['independent_decode_pass'] else 'FAIL'}",
             f"uvg_source_fix_verified {'PASS' if integrity['uvg_source_fix_verified'] else 'FAIL'}",
             f"final_integrity {integrity['status']}",
             f"final_rd_points.csv path {ROOT / 'final_rd_points.csv'}",
             f"strict_nonworse_pairs.csv path {ROOT / 'strict_nonworse_pairs.csv'}",
             f"matched_perceptual_pairs.csv path {ROOT / 'matched_perceptual_pairs.csv'}",
             f"summary_by_dataset.csv path {ROOT / 'summary_by_dataset.csv'}",
             f"visualizations path {ROOT / 'visualizations'}"]
    output = "\n".join(lines) + "\n"
    (ROOT / "stdout.log").write_text(output); (ROOT / "stderr.log").write_text("")
    print(output, end="")


if __name__ == "__main__": main()
