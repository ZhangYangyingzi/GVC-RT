"""Export requested V1.6 M2/M4 guarded per-video comparison."""
from bridge import *


def run(out):
    rows = []
    for split in ("Train2", "Val6"):
        summary = read_csv(V16_RESULTS / f"{split}_guard_summary.csv")
        for video in sorted({r["video"] for r in summary}):
            for alpha in (0.1, 0.25, 0.5):
                for scope in ("P", "IP"):
                    m2 = next(r for r in summary if r["video"] == video and r["alpha"] == alpha and
                              r["scope"] == scope and r["method"] == "M2_E_FRAME_GUARD")
                    m4 = next(r for r in summary if r["video"] == video and r["alpha"] == alpha and
                              r["scope"] == scope and r["method"] == "M4_O_FRAME_GUARD")
                    row = {"split": split, "video": video, "alpha": alpha, "scope": scope,
                           "comparison_type": "same budget upper bound, not same actual rate"}
                    for prefix, source in (("M2", m2), ("M4", m4)):
                        for key in ("base_bytes", "enhancement_bytes", "total_bytes", "budget_bytes",
                                    "stream_sent", "selected_label_nonzero_frames", "actual_nonzero_frames",
                                    "Z_count", "E_count", "O1_count", "O2_count", "psnr", "ms_ssim",
                                    "lpips", "psnr_minus_ZERO", "ms_ssim_minus_ZERO", "lpips_minus_ZERO"):
                            row[f"{prefix}_{key}"] = source[key]
                    for key in ("enhancement_bytes", "total_bytes", "actual_nonzero_frames", "psnr",
                                "ms_ssim", "lpips", "psnr_minus_ZERO", "ms_ssim_minus_ZERO",
                                "lpips_minus_ZERO"):
                        row[f"M4_minus_M2_{key}"] = float(m4[key]) - float(m2[key])
                    rows.append(row)
    csv_dump(out / "legacy_v16_M2_vs_M4_guard_per_video.csv", rows, replace=True)


if __name__ == "__main__":
    run(HOME / "results_v1")
