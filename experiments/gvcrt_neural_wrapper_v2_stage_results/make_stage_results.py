#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parents[0] / "gvcrt_neural_wrapper_v2_retest_fixed_uvg"
OUT = ROOT
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "rd_curves/per_video").mkdir(parents=True, exist_ok=True)

with open(V2 / "test_manifest.json") as f:
    manifest = json.load(f)
ulong_tags = {f"fresh_ulong_{int(v['video_id']):02d}" for v in manifest["videos"]}
rows = []
with open(V2 / "final_rd_points.csv", newline="") as f:
    rows = [r for r in csv.DictReader(f) if r["video_tag"] in ulong_tags]
methods = ["original", "stage_a", "beta_low", "beta_mid", "beta_high"]
labels = {"original": "Original GVC", "stage_a": "Stage A", "beta_low": "beta_low", "beta_mid": "beta_mid", "beta_high": "beta_high"}
fields = ["method", "qp", "video_count", "mean_kbps", "mean_bpp", "mean_LPIPS", "mean_DISTS", "mean_PSNR", "mean_MS_SSIM"]
agg = []
for method in methods:
    for qp in range(4):
        group = [r for r in rows if r["method"] == method and int(r["qp"]) == qp]
        agg.append({"method": method, "qp": qp, "video_count": len(group),
                    "mean_kbps": sum(float(r["kbps"]) for r in group) / len(group),
                    "mean_bpp": sum(float(r["bpp"]) for r in group) / len(group),
                    "mean_LPIPS": sum(float(r["LPIPS"]) for r in group) / len(group),
                    "mean_DISTS": sum(float(r["DISTS"]) for r in group) / len(group),
                    "mean_PSNR": sum(float(r["PSNR"]) for r in group) / len(group),
                    "mean_MS_SSIM": sum(float(r["MS_SSIM"]) for r in group) / len(group)})
with open(OUT / "v2_ulong_rd_points.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(agg)
(OUT / "source_results_manifest.json").write_text(json.dumps({"source": str(V2 / "final_rd_points.csv"), "videos": sorted(ulong_tags), "methods": methods, "qps": [0,1,2,3], "real_rans": True}, indent=2) + "\n")

plots = [("mean_LPIPS", "Rate–LPIPS", "LPIPS", "rate_lpips"), ("mean_DISTS", "Rate–DISTS", "DISTS", "rate_dists"), ("mean_PSNR", "Rate–PSNR", "PSNR", "rate_psnr"), ("mean_MS_SSIM", "Rate–MS-SSIM", "MS-SSIM", "rate_msssim")]
for metric, xlabel, ylabel, stem in plots:
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in methods:
        g = [r for r in agg if r["method"] == method]
        ax.plot([r["mean_kbps"] for r in g], [r[metric] for r in g], marker="o", label=labels[method])
    ax.set_xlabel("Mean real bitrate (kbps)"); ax.set_ylabel(ylabel); ax.grid(True, alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / f"rd_curves/{stem}.png", dpi=180); fig.savefig(OUT / f"rd_curves/{stem}.pdf"); plt.close(fig)
for tag in sorted(ulong_tags):
    for metric, ylabel, stem in [("LPIPS", "LPIPS", "lpips"), ("DISTS", "DISTS", "dists")]:
        fig, ax = plt.subplots(figsize=(6,4))
        for method in ("original", "beta_low"):
            g = [r for r in rows if r["video_tag"] == tag and r["method"] == method]
            g.sort(key=lambda r: int(r["qp"]))
            ax.plot([float(r["kbps"]) for r in g], [float(r[metric]) for r in g], marker="o", label=labels[method])
        ax.set_xlabel("Real bitrate (kbps)"); ax.set_ylabel(ylabel); ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout()
        fig.savefig(OUT / f"rd_curves/per_video/{tag}_rate_{stem}.png", dpi=180); fig.savefig(OUT / f"rd_curves/per_video/{tag}_rate_{stem}.pdf"); plt.close(fig)
print(OUT / "rd_curves")
