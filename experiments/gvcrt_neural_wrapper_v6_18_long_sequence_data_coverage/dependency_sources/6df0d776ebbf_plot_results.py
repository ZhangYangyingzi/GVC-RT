import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(ROOT/"rd_interpolation_inputs.csv")))
methods = {"original":"Original GVC","beta_low":"V2 beta_low","v3_beta_low":"V3 step20000","v4_selected":"V4 selected"}
for metric in ("LPIPS","DISTS","PSNR","MS_SSIM"):
    fig, ax = plt.subplots(figsize=(7,5), constrained_layout=True)
    for method, label in methods.items():
        points = sorted((r for r in rows if r["method"] == method), key=lambda r:int(r["qp"]))
        ax.plot([float(p["mean_kbps"]) for p in points], [float(p["mean_"+metric]) for p in points], marker="o", label=label)
    ax.set_xlabel("Mean real bitrate (kbps)"); ax.set_ylabel(metric); ax.legend(); ax.grid(alpha=.3)
    for ext in ("png","pdf"):
        fig.savefig(ROOT/"rd_curves"/f"rate_{metric.lower()}.{ext}", dpi=180)
    plt.close(fig)
