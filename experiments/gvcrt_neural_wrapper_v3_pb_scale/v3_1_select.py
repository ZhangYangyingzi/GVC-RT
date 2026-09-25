#!/usr/bin/env python3
"""Merge frozen validation points and select the V3.1 checkpoint."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
STEPS = (20000, 30000, 40000, 50000)
METRICS = ("kbps", "LPIPS", "DISTS", "PSNR", "SSIM", "MS_SSIM", "bpp")


def read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    old = read(ROOT / "checkpoint_validation.csv")
    if len(old) != 132:
        raise RuntimeError("original V3 validation is incomplete")
    new = []
    for gpu in (5, 6):
        new.extend(read(ROOT / f"parts/v3_1_validation_gpu{gpu}.csv"))
    expected = {(step, video, qp) for step in STEPS[1:] for video in range(6) for qp in (1, 3)}
    found = {(int(r["step"]), int(r["video_index"]), int(r["qp"])) for r in new}
    if len(new) != 36 or found != expected:
        raise RuntimeError(f"V3.1 validation incomplete: {len(new)} rows")
    for row in new:
        path = Path(row["bitstream_path"])
        if (not path.is_file() or path.stat().st_size != int(row["real_bytes"])
                or int(row["bytes_consumed"]) != int(row["real_bytes"])
                or row["decode_status"] != "PASS" or row["state_sync_pass"] != "True"
                or row["compression_hash_before"] != row["compression_hash_after"]
                or not all(math.isfinite(float(row[metric])) for metric in METRICS)):
            raise RuntimeError(f"invalid V3.1 validation row: {row['step']} {row['video_index']} {row['qp']}")
    merged = old + sorted(new, key=lambda r: (int(r["step"]), int(r["video_index"]), int(r["qp"])))
    write(ROOT / "checkpoint_validation.csv", merged)

    old_convergence = read(ROOT / "training_convergence.csv")
    groups = defaultdict(list)
    for row in new:
        groups[(int(row["step"]), int(row["qp"]))].append(row)
    added_convergence = []
    for (step, qp), rows in sorted(groups.items()):
        added_convergence.append({"stage": "stage_b", "step": step, "qp": qp,
            "video_count": len(rows), **{f"mean_{metric}": sum(float(r[metric]) for r in rows) / len(rows)
                                         for metric in ("kbps", "bpp", "LPIPS", "DISTS", "PSNR", "MS_SSIM")}})
    write(ROOT / "training_convergence.csv", old_convergence + added_convergence)

    baseline = {(int(r["video_index"]), int(r["qp"])): r for r in old if r["stage"] == "original"}
    candidates = [r for r in merged if r["stage"] == "stage_b" and int(r["step"]) in STEPS]
    summary = []
    for step in STEPS:
        group = [r for r in candidates if int(r["step"]) == step]
        if len(group) != 12:
            raise RuntimeError(f"checkpoint {step} lacks 12 validation points")
        pairs = [(r, baseline[int(r["video_index"]), int(r["qp"])]) for r in group]
        mean = lambda metric: sum(float(r[metric]) for r in group) / len(group)
        baseline_mean = lambda metric: sum(float(r[metric]) for r in baseline.values()) / len(baseline)
        summary.append({"step": step, "mean_real_kbps": mean("kbps"),
            "mean_rate_change_percent_vs_original": 100 * (mean("kbps") / baseline_mean("kbps") - 1),
            "mean_LPIPS": mean("LPIPS"), "mean_LPIPS_change_vs_original": mean("LPIPS") - baseline_mean("LPIPS"),
            "mean_DISTS": mean("DISTS"), "mean_DISTS_change_vs_original": mean("DISTS") - baseline_mean("DISTS"),
            "rate_lower_count": sum(float(a["kbps"]) < float(b["kbps"]) for a, b in pairs),
            "LPIPS_nonworse_count": sum(float(a["LPIPS"]) <= float(b["LPIPS"]) for a, b in pairs),
            "DISTS_nonworse_count": sum(float(a["DISTS"]) <= float(b["DISTS"]) for a, b in pairs),
            "LPIPS_and_DISTS_nonworse_count": sum(float(a["LPIPS"]) <= float(b["LPIPS"])
                                                   and float(a["DISTS"]) <= float(b["DISTS"]) for a, b in pairs)})
    write(ROOT / "v3_1_convergence_summary.csv", summary)
    eligible = [r for r in summary if r["mean_rate_change_percent_vs_original"] < 0
                and r["mean_LPIPS_change_vs_original"] <= 0 and r["mean_DISTS_change_vs_original"] <= 0]
    if not eligible:
        raise RuntimeError("no checkpoint meets all three validation requirements")
    lowest_rate = min(r["mean_real_kbps"] for r in eligible)
    tied = [r for r in eligible if r["mean_real_kbps"] <= lowest_rate * 1.005]
    chosen = min(tied, key=lambda r: (r["mean_LPIPS"], r["mean_DISTS"], r["mean_real_kbps"]))
    selection = [{**r, "eligible": r in eligible, "within_rate_tie_0_5_percent": r in tied,
                  "selected": r is chosen} for r in summary]
    write(ROOT / "v3_1_checkpoint_selection.csv", selection)
    (ROOT / "v3_1_selected_checkpoint.json").write_text(json.dumps({
        "stage": "stage_b", "step": chosen["step"], "checkpoint": str(ROOT / "checkpoints/stage_b" / f"step_{chosen['step']:05d}.pt"),
        "used_final_test": False, "rate_tie_relative_tolerance": 0.005,
        "selection_rule": "three validation nonworse gates, lowest bitrate; LPIPS then DISTS inside 0.5% bitrate tie"}, indent=2) + "\n")
    intervals = []
    for previous, current in zip(summary, summary[1:]):
        rate = 100 * (current["mean_real_kbps"] / previous["mean_real_kbps"] - 1)
        rate_points = (current["mean_rate_change_percent_vs_original"]
                       - previous["mean_rate_change_percent_vs_original"])
        lpips = current["mean_LPIPS"] - previous["mean_LPIPS"]
        dists = current["mean_DISTS"] - previous["mean_DISTS"]
        intervals.append({"from_step": previous["step"], "to_step": current["step"],
            "rate_change_percent": rate, "rate_change_percentage_points_vs_original": rate_points,
            "LPIPS_change": lpips, "DISTS_change": dists,
            "rate_plateau": abs(rate_points) < 0.5, "LPIPS_plateau": abs(lpips) < .001,
            "DISTS_plateau": abs(dists) < .001})
    (ROOT / "v3_1_convergence_status.json").write_text(json.dumps({
        "best_step": chosen["step"], "metric_changes": intervals}, indent=2) + "\n")
    curves = ROOT / "training_curves"
    for metric in ("kbps", "LPIPS", "DISTS", "PSNR"):
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for qp in (1, 3):
            points = [r for r in read(ROOT / "training_convergence.csv")
                      if r["stage"] == "stage_b" and int(r["qp"]) == qp]
            axis.plot([int(r["step"]) for r in points], [float(r[f"mean_{metric}"]) for r in points], "o-", label=f"QP{qp}")
        axis.set(xlabel="Stage B update", ylabel="Real kbps" if metric == "kbps" else metric)
        axis.grid(alpha=.3)
        axis.legend()
        figure.savefig(curves / f"v3_1_step_vs_{metric.lower()}.png", dpi=180)
        plt.close(figure)
    print(json.dumps({"selected_step": chosen["step"], "validation_rows": len(merged)}))


if __name__ == "__main__":
    main()
