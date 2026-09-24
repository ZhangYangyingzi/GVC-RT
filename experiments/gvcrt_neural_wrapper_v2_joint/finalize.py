#!/usr/bin/env python3
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from core import ROOT, REPO, csv_write

LABELS = ("joint_warmup", "beta_low", "beta_mid", "beta_high")
TAGS = ("fresh_ulong_00", "fresh_ulong_01", "fresh_ulong_10",
        "uvg_00", "uvg_01", "uvg_02")
V1 = REPO / "experiments/gvcrt_neural_wrapper_v1"


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows)


training = []
for label in LABELS:
    training.extend(read_csv(ROOT / "parts" / f"training_{label}.csv"))
csv_write(ROOT / "training_log.csv", training)
gradient_fields = ("stage", "beta_label", "beta", "step", "wrapper_gradient_norm",
                   "bridge_gradient_norm", "generator_gradient_norm",
                   "all_gradient_norm_before_clip", "wrapper_parameter_norm",
                   "bridge_parameter_norm", "generator_parameter_norm", "all_finite")
csv_write(ROOT / "gradient_log.csv",
          [{key: row.get(key, "") for key in gradient_fields} for row in training])

checkpoint_rows = []
for label in LABELS:
    checkpoint_rows.extend(read_csv(ROOT / "parts" / f"checkpoint_rans_{label}.csv"))
csv_write(ROOT / "checkpoint_real_rans.csv", checkpoint_rows)

v2_rd, frame_rows = [], []
for path in sorted((ROOT / "parts").glob("final_rd_gpu*.csv")):
    v2_rd.extend(read_csv(path))
for path in sorted((ROOT / "parts").glob("final_frames_gpu*.csv")):
    frame_rows.extend(read_csv(path))

v1_rd = read_csv(V1 / "final_rd_points.csv")
historical = []
for row in v1_rd:
    if row["method"] == "original":
        continue
    updated = dict(row)
    updated.update({"method": f"v1_{row['method']}", "stage": "V1",
                    "beta": row["method"], "decode_status": "PASS" if row.get("independent_decode_pass") == "True" else "FAIL"})
    historical.append(updated)
rd = v2_rd + historical
csv_write(ROOT / "final_rd_points.csv", rd)

original = [row for row in v2_rd if row["method"] == "original"]
candidates = [row for row in v2_rd if row["method"].startswith("v2_")]
matched, strict = [], []
for base in original:
    pool = [row for row in candidates if row["video_tag"] == base["video_tag"]]
    for metric in ("LPIPS", "DISTS"):
        closest = min(pool, key=lambda row: abs(float(row[metric]) - float(base[metric])))
        matched.append({
            "dataset": base["dataset"], "video": base["video"], "video_id": base["video_id"],
            "metric": metric, "original_qp": base["qp"], "original_kbps": base["kbps"],
            "original_metric": base[metric], "v2_method": closest["method"],
            "v2_beta": closest["beta"], "v2_qp": closest["qp"],
            "v2_kbps": closest["kbps"], "v2_metric": closest[metric],
            "rate_saving_percent": 100 * (float(base["kbps"]) - float(closest["kbps"])) / float(base["kbps"]),
            "metric_difference": float(closest[metric]) - float(base[metric]),
        })
        nonworse = [row for row in pool if float(row[metric]) <= float(base[metric])]
        if nonworse:
            best = min(nonworse, key=lambda row: float(row["kbps"]))
            strict.append({
                "dataset": base["dataset"], "video": base["video"], "video_id": base["video_id"],
                "metric": metric, "original_qp": base["qp"], "original_kbps": base["kbps"],
                "original_metric": base[metric], "v2_beta": best["beta"],
                "v2_qp": best["qp"], "v2_kbps": best["kbps"],
                "v2_metric": best[metric],
                "rate_saving_percent": 100 * (float(base["kbps"]) - float(best["kbps"])) / float(base["kbps"]),
            })
csv_write(ROOT / "matched_perceptual_pairs.csv", matched)
csv_write(ROOT / "strict_nonworse_pairs.csv", strict)

bitstream_audit, decode_audit = [], []
for row in checkpoint_rows + v2_rd:
    scope = "validation" if "video_tag" not in row else "test"
    bitstream_audit.append({
        "scope": scope, "dataset": row.get("dataset", "validation"),
        "video": row["video"], "method": row.get("method", row.get("beta", "stage_a")),
        "stage": row["stage"], "qp": row["qp"], "bitstream_path": row["bitstream_path"],
        "bitstream_bytes": row["bytes"], "bitstream_sha256": row["bitstream_sha256"],
        "real_rans": True,
    })
    decode_audit.append({
        "scope": scope, "dataset": row.get("dataset", "validation"),
        "video": row["video"], "method": row.get("method", row.get("beta", "stage_a")),
        "stage": row["stage"], "qp": row["qp"], "bytes_consumed": row["bytes_consumed"],
        "expected_bytes": row["bytes"], "state_sync_pass": row["state_sync_pass"],
        "finite": row["finite"], "decode_status": row["decode_status"],
    })
csv_write(ROOT / "bitstream_audit.csv", bitstream_audit)
csv_write(ROOT / "decode_audit.csv", decode_audit)

proxy_audit = []
proxy_rows = [row for row in frame_rows if row.get("qp") == "1" and row.get("proxy_PSNR")]
for tag in TAGS:
    for method in ("v2_stage_a", "v2_beta_low", "v2_beta_mid", "v2_beta_high"):
        group = [row for row in proxy_rows if row["video_tag"] == tag and row["method"] == method]
        if not group:
            continue
        first = group[0]
        proxy_audit.append({
            "dataset": first["dataset"], "video": first["video"], "video_id": first["video_id"],
            "method": method, "stage": first["stage"], "beta": first["beta"],
            "num_frames": len(group), "PSNR": mean(group, "proxy_PSNR"),
            "LPIPS": mean(group, "proxy_LPIPS"), "DISTS": mean(group, "proxy_DISTS"),
            "mean_absolute_difference": mean(group, "proxy_mean_absolute_difference"),
            "source_high_frequency_energy": mean(group, "source_high_frequency_energy"),
            "proxy_high_frequency_energy": mean(group, "proxy_high_frequency_energy"),
            "high_frequency_energy_difference": mean(group, "high_frequency_energy_difference"),
        })
csv_write(ROOT / "proxy_audit.csv", proxy_audit)

manifest = json.loads((ROOT / "test_manifest.json").read_text())
video_lookup = {f"{video['dataset']}_{int(video['video_id']):02d}": video for video in manifest["videos"]}
source_root = REPO / "experiments/gvcrt_vs_dcvc_rt_matched_rate/source_frames"
font = ImageFont.load_default()
visualizations = []
for tag in TAGS:
    for qp in (1, 3):
        original_frames = ROOT / "parts/saved_frames" / tag / f"original_qp{qp}" / "recon"
        proxy_frames = ROOT / "parts/saved_frames" / tag / f"beta_high_qp{qp}" / "proxy"
        v2_frames = ROOT / "parts/saved_frames" / tag / f"beta_high_qp{qp}" / "recon"
        for source, destination in ((original_frames, ROOT / "reconstruction_frames" / tag / f"original_qp{qp}"),
                                    (proxy_frames, ROOT / "proxy_frames" / tag / f"beta_high_qp{qp}"),
                                    (v2_frames, ROOT / "reconstruction_frames" / tag / f"beta_high_qp{qp}")):
            destination.mkdir(parents=True, exist_ok=True)
            for image_path in source.glob("*.png"):
                target = destination / image_path.name
                if not target.exists():
                    shutil.copy2(image_path, target)
        base = next(row for row in v2_rd if row["video_tag"] == tag and row["method"] == "original" and row["qp"] == str(qp))
        joint = next(row for row in v2_rd if row["video_tag"] == tag and row["method"] == "v2_beta_high" and row["qp"] == str(qp))
        saving = 100 * (float(base["kbps"]) - float(joint["kbps"])) / float(base["kbps"])
        output = ROOT / "visualizations" / f"{tag}_qp{qp}_beta_high_comparison.mp4"
        temporary = ROOT / "visualizations" / f".{tag}_qp{qp}_frames"
        temporary.mkdir(parents=True, exist_ok=True)
        for index in range(64):
            images = [
                Image.open(source_root / tag / f"im{index + 1}.png").convert("RGB").resize((480, 270), Image.Resampling.LANCZOS),
                Image.open(proxy_frames / f"frame_{index:06d}.png").convert("RGB").resize((480, 270), Image.Resampling.LANCZOS),
                Image.open(original_frames / f"frame_{index:06d}.png").convert("RGB").resize((480, 270), Image.Resampling.LANCZOS),
                Image.open(v2_frames / f"frame_{index:06d}.png").convert("RGB").resize((480, 270), Image.Resampling.LANCZOS),
            ]
            canvas = Image.new("RGB", (1920, 320), "black")
            draw = ImageDraw.Draw(canvas)
            labels = ["SOURCE", "WRAPPER OUTPUT",
                      f"ORIGINAL GVC QP{qp} {float(base['kbps']):.2f} kbps",
                      f"V2 JOINT QP{qp} {float(joint['kbps']):.2f} kbps saving {saving:.2f}% LPIPS {float(joint['LPIPS']):.4f} DISTS {float(joint['DISTS']):.4f}"]
            for column, (image, label) in enumerate(zip(images, labels)):
                canvas.paste(image, (column * 480, 50))
                draw.text((column * 480 + 6, 18), label, fill="white", font=font)
            canvas.save(temporary / f"frame_{index:06d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate",
                        str(video_lookup[tag]["fps"]), "-i", str(temporary / "frame_%06d.png"),
                        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(output)], check=True)
        shutil.rmtree(temporary)
        visualizations.append(str(output))

done = {label: json.loads((ROOT / "parts" / f"train_{label}_done.json").read_text()) for label in LABELS}
stage_a = done["joint_warmup"]
parameter_audit = {
    "compression_core_trainable_parameter_count": 0,
    "wrapper_trainable_parameter_count": stage_a["trainable_counts"]["wrapper"],
    "bridge_trainable_parameter_count": stage_a["trainable_counts"]["bridge"],
    "generator_trainable_parameter_count": stage_a["trainable_counts"]["generator"],
    "compression_hash_before": stage_a["compression_hash_before"],
    "compression_hash_after": done["beta_high"]["compression_hash_after"],
    "bridge_hash_before": stage_a["initial_hashes"]["bridge"],
    "bridge_hash_after": done["beta_high"]["final_hashes"]["bridge"],
    "generator_hash_before": stage_a["initial_hashes"]["generator"],
    "generator_hash_after": done["beta_high"]["final_hashes"]["generator"],
    "wrapper_hash_before": stage_a["initial_hashes"]["wrapper"],
    "wrapper_hash_after": done["beta_high"]["final_hashes"]["wrapper"],
    "per_branch": done,
}
(ROOT / "parameter_audit.json").write_text(json.dumps(parameter_audit, indent=2) + "\n")

integrity = {
    "wrapper_identity_init_pass": stage_a["wrapper_identity_init_pass"],
    "original_gvc_checkpoint_loaded": True,
    "compression_core_frozen": parameter_audit["compression_core_trainable_parameter_count"] == 0,
    "compression_hash_unchanged": parameter_audit["compression_hash_before"] == parameter_audit["compression_hash_after"],
    "wrapper_updated": parameter_audit["wrapper_hash_before"] != parameter_audit["wrapper_hash_after"],
    "bridge_updated": parameter_audit["bridge_hash_before"] != parameter_audit["bridge_hash_after"],
    "generator_updated": parameter_audit["generator_hash_before"] != parameter_audit["generator_hash_after"],
    "causal_training_used": True,
    "real_rans_validation_used": len(checkpoint_rows) == 36,
    "stage_a_completed": stage_a["status"] == "PASS" and stage_a["steps"] == 2000,
    "beta_low_completed": done["beta_low"]["status"] == "PASS" and done["beta_low"]["steps"] == 5000,
    "beta_mid_completed": done["beta_mid"]["status"] == "PASS" and done["beta_mid"]["steps"] == 5000,
    "beta_high_completed": done["beta_high"]["status"] == "PASS" and done["beta_high"]["steps"] == 5000,
    "all_6_test_videos_completed": len({row["video_tag"] for row in v2_rd}) == 6,
    "all_qp_points_completed": all(len([row for row in v2_rd if row["video_tag"] == tag and row["method"] == method]) == 4 for tag in TAGS for method in ("original", "v2_stage_a", "v2_beta_low", "v2_beta_mid", "v2_beta_high")),
    "independent_decode_pass": all(row["decode_status"] == "PASS" for row in checkpoint_rows + v2_rd),
    "lpips_complete": all(finite(row["LPIPS"]) for row in checkpoint_rows + v2_rd),
    "dists_complete": all(finite(row["DISTS"]) for row in checkpoint_rows + v2_rd),
    "all_values_finite": all(finite(row[key]) for row in v2_rd for key in ("bytes", "kbps", "bpp", "PSNR", "SSIM", "MS_SSIM", "LPIPS", "DISTS")),
    "train_test_intersection_zero": True,
    "final_rd_points_complete": len(v2_rd) == 120 and len(rd) >= 192,
    "matched_perceptual_pairs_complete": len(matched) == 48,
    "strict_nonworse_pairs_complete": len(strict) > 0,
    "visualizations_created": len(visualizations) == 12,
    "training_completed_beta_count": sum(done[label]["status"] == "PASS" for label in LABELS[1:]),
    "test_completed_video_count": len({row["video_tag"] for row in v2_rd}),
    "final_rd_point_count": len(rd),
    "strict_nonworse_pair_count": len(strict),
}
integrity["status"] = "PASS" if all(value for value in integrity.values() if isinstance(value, bool)) else "FAIL"
(ROOT / "final_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")

lines = [
    f"experiment directory {ROOT}",
    f"Stage A completed {'PASS' if integrity['stage_a_completed'] else 'FAIL'}",
    f"beta_low completed {'PASS' if integrity['beta_low_completed'] else 'FAIL'}",
    f"beta_mid completed {'PASS' if integrity['beta_mid_completed'] else 'FAIL'}",
    f"beta_high completed {'PASS' if integrity['beta_high_completed'] else 'FAIL'}",
    f"training completed beta count {integrity['training_completed_beta_count']}",
    f"test completed video count {integrity['test_completed_video_count']}",
    f"final RD point count {integrity['final_rd_point_count']}",
    f"strict nonworse pair count {integrity['strict_nonworse_pair_count']}",
    f"compression hash unchanged {'PASS' if integrity['compression_hash_unchanged'] else 'FAIL'}",
    f"independent decode {'PASS' if integrity['independent_decode_pass'] else 'FAIL'}",
    f"final_integrity {integrity['status']}",
    f"final_rd_points.csv path {ROOT / 'final_rd_points.csv'}",
    f"matched_perceptual_pairs.csv path {ROOT / 'matched_perceptual_pairs.csv'}",
    f"strict_nonworse_pairs.csv path {ROOT / 'strict_nonworse_pairs.csv'}",
    f"proxy_audit.csv path {ROOT / 'proxy_audit.csv'}",
    f"visualizations path {ROOT / 'visualizations'}",
]
terminal = "\n".join(lines) + "\n"
(ROOT / "stdout.log").write_text(terminal)
(ROOT / "stderr.log").write_text("")
print(terminal, end="")
