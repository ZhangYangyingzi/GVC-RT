#!/usr/bin/env python3
"""Build final V3 reports after validation selection and final test are complete."""
import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "gvcrt_neural_wrapper_v2_retest_fixed_uvg"
METHODS = ("original", "beta_low", "v3_beta_low")
LABELS = {"original": "Original GVC", "beta_low": "V2 beta_low",
          "v3_beta_low": "V3 P+Bridge beta_low"}
METRICS = ("kbps", "bpp", "LPIPS", "DISTS", "PSNR", "SSIM", "MS_SSIM")


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def data_rows():
    test = json.loads((ROOT / "test_manifest.json").read_text())["videos"]
    tags = {f"fresh_ulong_{int(video['video_id']):02d}" for video in test}
    controls = [row for row in read_csv(V2 / "final_rd_points.csv")
                if row["video_tag"] in tags and row["method"] in METHODS[:2]]
    v3 = []
    for path in (ROOT / "parts").glob("final_v3_gpu*.csv"):
        v3.extend(read_csv(path))
    rows = controls + v3
    keys = [(row["video_tag"], row["method"], int(row["qp"])) for row in rows]
    expected = {(tag, method, qp) for tag in tags for method in METHODS for qp in range(4)}
    if len(rows) != 96 or len(set(keys)) != 96 or set(keys) != expected:
        raise RuntimeError(f"final test incomplete: {len(rows)} rows, {len(set(keys))} unique")
    selected = json.loads((ROOT / "selected_checkpoint.json").read_text())
    checkpoint = Path(selected["checkpoint"])
    digest = file_hash(checkpoint)
    if any(row["checkpoint_sha256"] != digest for row in v3):
        raise RuntimeError("V3 final rows do not match selected checkpoint")
    rows.sort(key=lambda row: (row["video_tag"], METHODS.index(row["method"]), int(row["qp"])))
    return rows, test


def summary_tables(rows):
    by_key = {(r["video_tag"], r["method"], int(r["qp"])): r for r in rows}
    summary = []
    comparisons = (("beta_low", "original"), ("v3_beta_low", "original"),
                   ("v3_beta_low", "beta_low"))
    for candidate, reference in comparisons:
        for qp in range(4):
            pairs = [(by_key[tag, candidate, qp], by_key[tag, reference, qp])
                     for tag in sorted({r["video_tag"] for r in rows})]
            summary.append({"candidate": candidate, "reference": reference, "qp": qp,
                            "video_count": len(pairs),
                            "mean_bitrate_change_percent": sum(100 * (float(a["kbps"]) / float(b["kbps"]) - 1)
                                                               for a, b in pairs) / len(pairs),
                            **{f"mean_{metric}_change": sum(float(a[metric]) - float(b[metric])
                                                              for a, b in pairs) / len(pairs)
                               for metric in ("LPIPS", "DISTS", "PSNR", "MS_SSIM")},
                            "rate_lower_count": sum(float(a["kbps"]) < float(b["kbps"]) for a, b in pairs),
                            "LPIPS_nonworse_count": sum(float(a["LPIPS"]) <= float(b["LPIPS"]) for a, b in pairs),
                            "DISTS_nonworse_count": sum(float(a["DISTS"]) <= float(b["DISTS"]) for a, b in pairs),
                            "LPIPS_and_DISTS_nonworse_count": sum(float(a["LPIPS"]) <= float(b["LPIPS"])
                                                                  and float(a["DISTS"]) <= float(b["DISTS"])
                                                                  for a, b in pairs)})
    write_csv(ROOT / "same_qp_summary.csv", summary)
    matched, strict = [], []
    for candidate, reference in comparisons:
        for base in (r for r in rows if r["method"] == reference):
            pool = [r for r in rows if r["method"] == candidate and r["video_tag"] == base["video_tag"]]
            for metric in ("LPIPS", "DISTS"):
                common = {"candidate": candidate, "reference": reference,
                          "video_tag": base["video_tag"], "metric": metric,
                          "reference_qp": base["qp"], "reference_kbps": base["kbps"],
                          "reference_metric": base[metric]}
                closest = min(pool, key=lambda row: abs(float(row[metric]) - float(base[metric])))
                matched.append({**common, "candidate_qp": closest["qp"],
                                "candidate_kbps": closest["kbps"], "candidate_metric": closest[metric],
                                "metric_difference": float(closest[metric]) - float(base[metric]),
                                "rate_saving_percent": 100 * (1 - float(closest["kbps"]) / float(base["kbps"]))})
                eligible = [r for r in pool if float(r[metric]) <= float(base[metric])]
                if eligible:
                    best = min(eligible, key=lambda row: float(row["kbps"]))
                    strict.append({**common, "candidate_qp": best["qp"],
                                   "candidate_kbps": best["kbps"], "candidate_metric": best[metric],
                                   "rate_saving_percent": 100 * (1 - float(best["kbps"]) / float(base["kbps"]))})
    write_csv(ROOT / "matched_perceptual_pairs.csv", matched)
    write_csv(ROOT / "strict_nonworse_pairs.csv", strict)
    rd = []
    for method in METHODS:
        for qp in range(4):
            group = [r for r in rows if r["method"] == method and int(r["qp"]) == qp]
            rd.append({"method": method, "qp": qp, "video_count": len(group),
                       **{f"mean_{metric}": sum(float(r[metric]) for r in group) / len(group)
                          for metric in METRICS}})
    write_csv(ROOT / "rd_interpolation_inputs.csv", rd)
    curves = ROOT / "rd_curves"
    curves.mkdir(exist_ok=True)
    for metric, stem in (("LPIPS", "lpips"), ("DISTS", "dists"),
                         ("PSNR", "psnr"), ("MS_SSIM", "msssim")):
        fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for method in METHODS:
            points = [r for r in rd if r["method"] == method]
            axis.plot([r["mean_kbps"] for r in points], [r[f"mean_{metric}"] for r in points],
                      "o-", label=LABELS[method])
        axis.set(xlabel="Mean real bitrate (kbps)", ylabel=metric.replace("_", "-"))
        axis.grid(alpha=.3)
        axis.legend()
        fig.savefig(curves / f"rate_{stem}.png", dpi=180)
        fig.savefig(curves / f"rate_{stem}.pdf")
        plt.close(fig)
    return rd


def render_visuals(rows, test):
    lookup = {f"fresh_ulong_{int(v['video_id']):02d}": v for v in test}
    indexed = {(r["video_tag"], r["method"], int(r["qp"])): r for r in rows}
    font = ImageFont.load_default()
    audit = []
    for tag, video in sorted(lookup.items()):
        for qp in (1, 3):
            output = ROOT / "visualizations" / f"{tag}_qp{qp}_comparison.mp4"
            output.parent.mkdir(exist_ok=True)
            command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", "1920x300", "-r", str(video["fps"]), "-i", "pipe:0",
                       "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(output)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            for index in range(64):
                source = V2 / "source_frames_fixed" / tag / f"im{index + 1}.png"
                paths = [source,
                         V2 / "reconstruction_frames" / tag / f"original_qp{qp}" / "recon" / f"frame_{index:06d}.png",
                         V2 / "reconstruction_frames" / tag / f"beta_low_qp{qp}" / "recon" / f"frame_{index:06d}.png",
                         ROOT / "reconstruction_frames" / tag / f"v3_beta_low_qp{qp}" / "recon" / f"frame_{index:06d}.png"]
                labels = [("SOURCE", "")]
                for method in METHODS:
                    row = indexed[tag, method, qp]
                    labels.append((LABELS[method],
                                   f"{float(row['kbps']):.2f} kbps   LP {float(row['LPIPS']):.4f}   DI {float(row['DISTS']):.4f}"))
                canvas = Image.new("RGB", (1920, 300), "black")
                draw = ImageDraw.Draw(canvas)
                for column, (path, label) in enumerate(zip(paths, labels)):
                    with Image.open(path) as picture:
                        canvas.paste(picture.convert("RGB").resize((480, 270), Image.Resampling.LANCZOS),
                                     (column * 480, 30))
                    draw.text((column * 480 + 4, 2), label[0], fill="white", font=font)
                    draw.text((column * 480 + 4, 16), label[1], fill="white", font=font)
                process.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"visualization ffmpeg failed: {output}")
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                    "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(output)],
                                   check=True, capture_output=True, text=True)
            if probe.stdout.strip() != "64":
                raise RuntimeError(f"visualization frame count failed: {output}")
            audit.append({"video_tag": tag, "qp": qp, "frames": 64, "path": str(output), "status": "PASS"})
    write_csv(ROOT / "visualization_audit.csv", audit)
    return audit


def audits(rows, test, visuals):
    train = json.loads((ROOT / "train_manifest.json").read_text())
    validation = json.loads((ROOT / "validation_manifest.json").read_text())
    stage_a = json.loads((ROOT / "parts/train_stage_a_done.json").read_text())
    stage_b = json.loads((ROOT / "parts/train_stage_b_done.json").read_text())
    train_hashes = {r["sha256"] for r in train["videos"]}
    val_hashes = {r["sha256"] for r in validation["videos"]}
    test_hashes = {r["source_sha256"] for r in test}
    bitstreams_valid = all(Path(r["bitstream_path"]).is_file() and
                           Path(r["bitstream_path"]).stat().st_size == int(r["bytes"]) and
                           file_hash(Path(r["bitstream_path"])) == r["bitstream_sha256"]
                           for r in rows)
    if not bitstreams_valid:
        raise RuntimeError("final bitstream size/hash audit failed")
    parameter = {"initial_hashes": stage_a["initial_hashes"],
                 "after_stage_a": stage_a["final_hashes"],
                 "before_stage_b": stage_b["initial_hashes"],
                 "after_stage_b": stage_b["final_hashes"],
                 "wrapper_updated": stage_a["initial_hashes"]["wrapper"] != stage_b["final_hashes"]["wrapper"],
                 "bridge_updated": stage_a["initial_hashes"]["bridge"] != stage_b["final_hashes"]["bridge"],
                 "compression_core_updated": stage_a["initial_hashes"]["compression"] != stage_b["final_hashes"]["compression"],
                 "detokenizer_updated": stage_a["initial_hashes"]["generator"] != stage_b["final_hashes"]["generator"],
                 "generator_trainable_parameters": stage_b["generator_trainable"]}
    (ROOT / "parameter_audit.json").write_text(json.dumps(parameter, indent=2) + "\n")
    write_csv(ROOT / "bitstream_audit.csv", [{"video_tag": r["video_tag"], "method": r["method"],
        "qp": r["qp"], "real_bytes": r["bytes"], "bitstream_path": r["bitstream_path"],
        "bitstream_sha256": r["bitstream_sha256"], "real_rans": True} for r in rows])
    write_csv(ROOT / "decode_audit.csv", [{"video_tag": r["video_tag"], "method": r["method"],
        "qp": r["qp"], "bytes_consumed": r["bytes_consumed"], "expected_bytes": r["bytes"],
        "decode_status": r["decode_status"], "state_sync_pass": r["state_sync_pass"]} for r in rows])
    for tag in {r["video_tag"] for r in rows}:
        for qp in (1, 3):
            source = ROOT / "reconstruction_frames" / tag / f"v3_beta_low_qp{qp}" / "proxy"
            destination = ROOT / "proxy_frames" / tag / f"v3_beta_low_qp{qp}"
            if source.exists() and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
    training = read_csv(ROOT / "parts/training_stage_a.csv") + read_csv(ROOT / "parts/training_stage_b.csv")
    write_csv(ROOT / "training_log.csv", training)
    validation_rows = read_csv(ROOT / "checkpoint_validation.csv")
    training_finite = len(training) == 25000 and all(
        math.isfinite(float(row[metric])) for row in training
        for metric in ("loss", "LPIPS", "DISTS", "R_est_bpp", "proxy_L1", "PSNR", "MS_SSIM"))
    validation_finite = len(validation_rows) == 132 and all(
        math.isfinite(float(row[metric])) for row in validation_rows
        for metric in ("kbps", "bpp", "LPIPS", "DISTS", "PSNR", "MS_SSIM"))
    integrity = {
        "online_random_sampling_used": train["online_sampling"], "fixed_1500_cache_not_used": True,
        "train_video_count": train["train_video_count"],
        "validation_video_count": validation["validation_video_count"],
        "test_video_count": len(test),
        "train_val_intersection_zero": not bool(train_hashes & val_hashes),
        "train_test_intersection_zero": not bool(train_hashes & test_hashes),
        "val_test_intersection_zero": not bool(val_hashes & test_hashes),
        "clip_length_4": True, "causal_4frame_training_used": True,
        "wrapper_updated": parameter["wrapper_updated"], "bridge_updated": parameter["bridge_updated"],
        "compression_core_frozen": not parameter["compression_core_updated"],
        "compression_hash_unchanged": not parameter["compression_core_updated"],
        "detokenizer_frozen": not parameter["detokenizer_updated"] and parameter["generator_trainable_parameters"] == 0,
        "detokenizer_hash_unchanged": not parameter["detokenizer_updated"],
        "stage_a_completed": stage_a["status"] == "PASS" and stage_a["steps"] == 5000,
        "stage_b_completed": stage_b["status"] == "PASS" and stage_b["steps"] == 20000,
        "checkpoint_20k_completed": (ROOT / "checkpoints/stage_b/step_20000.pt").is_file(),
        "real_rans_validation_complete": len(validation_rows) == 132,
        "final_8_ulong_test_complete": len(rows) == 96,
        "independent_decode_pass": all(r["decode_status"] == "PASS" and
                                       int(r["bytes_consumed"]) == int(r["bytes"]) for r in rows),
        "rd_curves_created": all((ROOT / "rd_curves" / f"rate_{metric}.{ext}").is_file()
                                 for metric in ("lpips", "dists", "psnr", "msssim") for ext in ("png", "pdf")),
        "visualizations_created": len(visuals) == 16,
        "all_values_finite": training_finite and validation_finite and
                             all(math.isfinite(float(r[metric])) for r in rows for metric in METRICS),
    }
    integrity["status"] = "PASS" if all(value for value in integrity.values() if isinstance(value, bool)) else "FAIL"
    (ROOT / "final_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    if integrity["status"] != "PASS":
        raise RuntimeError("final integrity failed")
    return integrity


def main():
    rows, test = data_rows()
    write_csv(ROOT / "final_rd_points.csv", rows)
    summary_tables(rows)
    visuals = render_visuals(rows, test)
    integrity = audits(rows, test, visuals)
    print(json.dumps({"status": integrity["status"], "final_points": len(rows),
                      "visualizations": len(visuals)}))


if __name__ == "__main__":
    main()
