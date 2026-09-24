#!/usr/bin/env python3
"""Audit validation completeness, plot convergence, and select a Stage B checkpoint."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
STEPS = {"original": (0,), "stage_a": (0, 1000, 2000, 5000),
         "stage_b": (0, 2000, 5000, 10000, 15000, 20000)}
METRICS = ("kbps", "bpp", "LPIPS", "DISTS", "PSNR", "MS_SSIM")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest = json.loads((ROOT / "validation_manifest.json").read_text())
    count = manifest["validation_video_count"]
    all_rows = {}
    for path in (ROOT / "parts").glob("checkpoint_validation_*.csv"):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["stage"], int(row["step"]), int(row["video_index"]), int(row["qp"]))
                if key in all_rows and row != all_rows[key]:
                    raise RuntimeError(f"conflicting validation row: {key}")
                all_rows[key] = row
    expected = {(stage, step, video, qp) for stage, steps in STEPS.items()
                for step in steps for video in range(count) for qp in (1, 3)}
    missing = expected - all_rows.keys()
    if missing:
        raise RuntimeError(f"missing {len(missing)} validation points; examples: {sorted(missing)[:5]}")
    rows = [all_rows[key] for key in sorted(expected)]
    for row in rows:
        path = Path(row["bitstream_path"])
        if (not path.is_file() or path.stat().st_size != int(row["real_bytes"])
                or int(row["bytes_consumed"]) != int(row["real_bytes"])
                or row["decode_status"] != "PASS" or row["state_sync_pass"] != "True"
                or row["compression_hash_before"] != row["compression_hash_after"]
                or not all(math.isfinite(float(row[metric])) for metric in METRICS)):
            raise RuntimeError(f"invalid validation result: {row['stage']} {row['step']} {row['video_index']} {row['qp']}")
    write_csv(ROOT / "checkpoint_validation.csv", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["stage"], int(row["step"]), int(row["qp"]))].append(row)
    convergence = []
    for (stage, step, qp), group in sorted(groups.items()):
        convergence.append({"stage": stage, "step": step, "qp": qp,
                            "video_count": len(group),
                            **{f"mean_{metric}": sum(float(row[metric]) for row in group) / len(group)
                               for metric in METRICS}})
    write_csv(ROOT / "training_convergence.csv", convergence)
    curves = ROOT / "training_curves"
    curves.mkdir(exist_ok=True)
    for metric, name in (("kbps", "real_bitrate"), ("LPIPS", "LPIPS"),
                         ("DISTS", "DISTS"), ("PSNR", "PSNR")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for axis, stage in zip(axes, ("stage_a", "stage_b")):
            for qp in (1, 3):
                points = [r for r in convergence if r["stage"] == stage and r["qp"] == qp]
                axis.plot([r["step"] for r in points], [r[f"mean_{metric}"] for r in points], "o-", label=f"QP{qp}")
            axis.set(xlabel="Update", ylabel="Real kbps" if metric == "kbps" else metric,
                     title=stage.replace("_", " ").title())
            axis.grid(alpha=.3)
            axis.legend()
        fig.savefig(curves / f"step_vs_{name}.png", dpi=180)
        plt.close(fig)
    for metric in ("LPIPS", "DISTS"):
        fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for stage in ("original", "stage_a", "stage_b"):
            for step in STEPS[stage]:
                points = [r for r in convergence if r["stage"] == stage and r["step"] == step]
                axis.plot([r["mean_kbps"] for r in points], [r[f"mean_{metric}"] for r in points],
                          "o-", label=f"{stage} {step}")
        axis.set(xlabel="Real kbps", ylabel=metric)
        axis.grid(alpha=.3)
        axis.legend(fontsize=7, ncol=2)
        fig.savefig(curves / f"validation_rate_{metric.lower()}_by_step.png", dpi=180)
        plt.close(fig)

    baseline = {qp: next(r for r in convergence if r["stage"] == "original" and r["qp"] == qp)
                for qp in (1, 3)}
    selection = []
    for step in STEPS["stage_b"][1:]:
        points = [r for r in convergence if r["stage"] == "stage_b" and r["step"] == step]
        rate_change = sum((r["mean_kbps"] / baseline[r["qp"]]["mean_kbps"] - 1) for r in points) / 2
        lpips_change = sum((r["mean_LPIPS"] - baseline[r["qp"]]["mean_LPIPS"]) for r in points) / 2
        dists_change = sum((r["mean_DISTS"] - baseline[r["qp"]]["mean_DISTS"]) for r in points) / 2
        acceptable = rate_change < 0 and lpips_change <= .01
        score = rate_change + 5 * max(0, lpips_change) + 2 * max(0, dists_change)
        selection.append({"stage": "stage_b", "step": step, "mean_rate_change_fraction": rate_change,
                          "mean_LPIPS_change": lpips_change, "mean_DISTS_change": dists_change,
                          "rate_lower_LPIPS_tolerance_met": acceptable, "selection_score": score,
                          "selected": False})
    eligible = [r for r in selection if r["rate_lower_LPIPS_tolerance_met"]]
    chosen = min(eligible or selection, key=lambda row: row["selection_score"])
    chosen["selected"] = True
    write_csv(ROOT / "checkpoint_selection.csv", selection)
    (ROOT / "selected_checkpoint.json").write_text(json.dumps({
        "stage": "stage_b", "step": chosen["step"],
        "checkpoint": str(ROOT / "checkpoints/stage_b" / f"step_{chosen['step']:04d}.pt"),
        "selection_criterion": "min rate plus nonworse perceptual penalty; prioritize real-rate reduction and LPIPS delta <= 0.01",
        "used_final_test": False}, indent=2) + "\n")
    print(json.dumps({"validation_points": len(rows), "selected_step": chosen["step"]}))


if __name__ == "__main__":
    main()
