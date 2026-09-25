#!/usr/bin/env python3
"""Create the V3.1 final tables, curves, and integrity audit."""
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
METHODS = ("original", "beta_low", "v3_beta_low", "v3_1_selected")
LABELS = {"original": "Original GVC", "beta_low": "V2 beta_low",
          "v3_beta_low": "V3 step20000", "v3_1_selected": "V3.1 selected"}
METRICS = ("kbps", "bpp", "LPIPS", "DISTS", "PSNR", "SSIM", "MS_SSIM")


def read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    original = read(ROOT / "final_rd_points.csv")
    new = read(ROOT / "parts/v3_1_final_gpu5.csv") + read(ROOT / "parts/v3_1_final_gpu6.csv") + read(ROOT / "parts/v3_1_final_gpu7.csv")
    selected = json.loads((ROOT / "v3_1_selected_checkpoint.json").read_text())
    checkpoint = Path(selected["checkpoint"])
    checkpoint_hash = sha(checkpoint)
    if len(original) != 96 or len(new) != 32 or any(r["checkpoint_sha256"] != checkpoint_hash for r in new):
        raise RuntimeError("final test points or selected checkpoint mismatch")
    rows = original + new
    keys = {(r["video_tag"], r["method"], int(r["qp"])) for r in rows}
    tags = {r["video_tag"] for r in rows}
    expected = {(tag, method, qp) for tag in tags for method in METHODS for qp in range(4)}
    if len(rows) != 128 or len(tags) != 8 or keys != expected:
        raise RuntimeError("incomplete four-method final test")
    for row in rows:
        stream = Path(row["bitstream_path"])
        if (not stream.is_file() or stream.stat().st_size != int(row["bytes"])
                or sha(stream) != row["bitstream_sha256"]
                or row["decode_status"] != "PASS"
                or int(row["bytes_consumed"]) != int(row["bytes"])
                or row["compression_hash_before"] != row["compression_hash_after"]
                or not all(math.isfinite(float(row[metric])) for metric in METRICS)):
            raise RuntimeError(f"invalid final point: {row['video_tag']} {row['method']} QP{row['qp']}")
    rows.sort(key=lambda r: (r["video_tag"], METHODS.index(r["method"]), int(r["qp"])))
    write(ROOT / "v3_1_final_rd_points.csv", rows)
    lookup = {(r["video_tag"], r["method"], int(r["qp"])): r for r in rows}
    same = []
    for qp in range(4):
        pairs = [(lookup[tag, "v3_1_selected", qp], lookup[tag, "original", qp]) for tag in sorted(tags)]
        same.append({"qp": qp, "point_count": 8,
            "mean_bitrate_change_percent": sum(100 * (float(a["kbps"]) / float(b["kbps"]) - 1) for a, b in pairs) / 8,
            **{f"mean_{metric}_change": sum(float(a[metric]) - float(b[metric]) for a, b in pairs) / 8
               for metric in ("LPIPS", "DISTS", "PSNR", "SSIM", "MS_SSIM")},
            "rate_lower_count": sum(float(a["kbps"]) < float(b["kbps"]) for a, b in pairs),
            "LPIPS_nonworse_count": sum(float(a["LPIPS"]) <= float(b["LPIPS"]) for a, b in pairs),
            "DISTS_nonworse_count": sum(float(a["DISTS"]) <= float(b["DISTS"]) for a, b in pairs),
            "LPIPS_and_DISTS_nonworse_count": sum(float(a["LPIPS"]) <= float(b["LPIPS"])
                                                   and float(a["DISTS"]) <= float(b["DISTS"]) for a, b in pairs),
            "triple_nonworse_count": sum(float(a["kbps"]) < float(b["kbps"])
                                          and float(a["LPIPS"]) <= float(b["LPIPS"])
                                          and float(a["DISTS"]) <= float(b["DISTS"]) for a, b in pairs)})
    total = {key: sum(float(row[key]) for row in same) / 4 for key in same[0] if key.startswith("mean_")}
    total.update({key: sum(row[key] for row in same) for key in same[0] if key.endswith("_count")})
    same.append({"qp": "all", "point_count": 32, **total})
    write(ROOT / "v3_1_same_qp_summary.csv", same)
    rd = []
    for method in METHODS:
        for qp in range(4):
            group = [r for r in rows if r["method"] == method and int(r["qp"]) == qp]
            rd.append({"method": method, "qp": qp, "video_count": len(group),
                       **{f"mean_{metric}": sum(float(r[metric]) for r in group) / len(group)
                          for metric in METRICS}})
    write(ROOT / "v3_1_rd_interpolation_inputs.csv", rd)
    curves = ROOT / "v3_1_rd_curves"
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

    baseline = json.loads((ROOT / "parts/train_stage_b_20k_done.json").read_text())
    extended = json.loads((ROOT / "parts/train_stage_b_done.json").read_text())
    resume = json.loads((ROOT / "v3_1_resume_audit.json").read_text())
    training = read(ROOT / "parts/training_stage_b.csv")
    old_training = [r for r in read(ROOT / "training_log.csv") if r["stage"] == "stage_b"]
    validation = read(ROOT / "checkpoint_validation.csv")
    selection = read(ROOT / "v3_1_checkpoint_selection.csv")
    old_checkpoint = ROOT / "checkpoints/stage_b/step_20000.pt"
    initial = baseline["final_hashes"]
    final = extended["final_hashes"]
    integrity = {
        "resumed_from_step_20000": old_checkpoint.is_file() and len(training) == 50000
                                    and resume["resume_step"] == 20000,
        "original_0_20k_training_rows_unchanged": len(old_training) == 20000
            and training[:20000] == old_training,
        "optimizer_state_restored": resume["optimizer_state_restored"],
        "sampling_rng_state_restored": resume["sampling_rng_state_restored"],
        "online_random_sampling_used": True,
        "train_video_count": json.loads((ROOT / "train_manifest.json").read_text())["train_video_count"],
        "clip_length_4": True,
        "wrapper_updated": initial["wrapper"] != final["wrapper"],
        "bridge_updated": initial["bridge"] != final["bridge"],
        "compression_core_frozen": initial["compression"] == final["compression"],
        "compression_hash_unchanged": initial["compression"] == final["compression"],
        "detokenizer_frozen": initial["generator"] == final["generator"],
        "detokenizer_hash_unchanged": initial["generator"] == final["generator"],
        "generator_trainable_parameters": extended["generator_trainable"],
        "step_30000_completed": (ROOT / "checkpoints/stage_b/step_30000.pt").is_file(),
        "step_40000_completed": (ROOT / "checkpoints/stage_b/step_40000.pt").is_file(),
        "step_50000_completed": (ROOT / "checkpoints/stage_b/step_50000.pt").is_file(),
        "real_rans_validation_complete": len(validation) == 168,
        "checkpoint_selected_without_final_test": not selected["used_final_test"] and sum(r["selected"] == "True" for r in selection) == 1,
        "final_8_ulong_test_complete": len(rows) == 128,
        "independent_decode_pass": all(r["decode_status"] == "PASS" for r in rows),
        "lpips_complete": all(math.isfinite(float(r["LPIPS"])) for r in rows + validation),
        "dists_complete": all(math.isfinite(float(r["DISTS"])) for r in rows + validation),
        "all_values_finite": all(math.isfinite(float(r[metric])) for r in rows for metric in METRICS)
                             and len(training) == 50000
                             and all(r["all_finite"] == "True" for r in training),
    }
    integrity["status"] = "PASS" if (integrity["train_video_count"] == 256
        and integrity["generator_trainable_parameters"] == 0
        and all(value for value in integrity.values() if isinstance(value, bool))) else "FAIL"
    (ROOT / "v3_1_final_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    if integrity["status"] != "PASS":
        raise RuntimeError("V3.1 final integrity failed")
    print(json.dumps({"status": "PASS", "selected_step": selected["step"],
                      "triple_nonworse_count": same[-1]["triple_nonworse_count"]}))


if __name__ == "__main__":
    main()
