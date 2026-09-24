"""Aggregate attribution, guarded quality, matched-rate, and diagnostics."""
from collections import Counter

from bridge import *

METRICS = ("mse", "psnr", "ms_ssim", "lpips", "d")


def mean(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def frozen_candidates(split):
    return read_csv(V15_RESULTS / f"{split}_candidate_frames.csv")


def zero_summary(split):
    return [r for r in read_csv(V15_RESULTS / f"{split}_summary.csv")
            if r["method"] == "M2_E_FRAME" and r["alpha"] == 0 and r["scope"] in ("P", "IP")]


def guard_summary(out, split):
    selected = read_csv(out / f"{split}_guard_selected_frames.csv")
    allocations = read_csv(out / f"{split}_guard_allocations.csv")
    zeros = zero_summary(split)
    result = []
    for allocation in allocations:
        chosen = [r for r in selected if r["video"] == allocation["video"] and
                  r["method"] == allocation["method"] and r["alpha"] == allocation["alpha"]]
        zero_p = next(r for r in zeros if r["video"] == allocation["video"] and r["scope"] == "P")
        zero_ip = next(r for r in zeros if r["video"] == allocation["video"] and r["scope"] == "IP")
        for scope in ("P", "IP"):
            row = {**allocation, "scope": scope, "frames": 15 if scope == "P" else 16,
                   "actual_nonzero_frames": sum(r["actual_nonzero_symbols"] for r in chosen)}
            for metric in METRICS:
                p_value = mean(chosen, metric)
                if scope == "P":
                    value = p_value
                else:
                    i_value = 16 * float(zero_ip[metric]) - 15 * float(zero_p[metric])
                    value = (15 * p_value + i_value) / 16
                row[metric] = value
                row[f"{metric}_minus_ZERO"] = value - float(zero_p[metric] if scope == "P" else zero_ip[metric])
            result.append(row)
    csv_dump(out / f"{split}_guard_summary.csv", result)
    return result


def normalize_v15(split):
    rows = read_csv(V15_RESULTS / f"{split}_summary.csv")
    result = []
    for row in rows:
        base = int(row["base_bytes"]); enhancement = int(row["actual_bytes"])
        result.append({**row, "enhancement_bytes": enhancement, "total_bytes": base + enhancement,
                       "enhancement_over_base": enhancement / base,
                       "total_bpp": (base + enhancement) * 8 / (16 * PIXELS),
                       "total_kbps": (base + enhancement) * 8 / duration_seconds(split, row["video"]) / 1000,
                       "actual_nonzero_frames": int(row["nonzero_enhancement_frames"])})
    return result


def full_results(out, split, guard):
    rows = normalize_v15(split) + guard
    csv_dump(out / f"{split}_all_method_results.csv", rows)
    return rows


def aggregate(out, split, rows):
    summaries = []
    videos = [e["video_id"] for e in entries(split)]
    for method in sorted(set(r["method"] for r in rows)):
        for alpha in CONFIG["budgets"]:
            for scope in ("P", "IP"):
                chosen = [r for r in rows if r["method"] == method and r["alpha"] == alpha and r["scope"] == scope]
                assert len(chosen) == len(videos)
                item = {"split": split, "method": method, "alpha": alpha, "scope": scope,
                        "videos": len(chosen), "coverage_videos": sum(bool(r["stream_sent"]) for r in chosen),
                        "mean_video_enhancement_over_base": mean(chosen, "enhancement_over_base"),
                        "pooled_enhancement_over_base": sum(float(r["enhancement_bytes"]) for r in chosen) / sum(float(r["base_bytes"]) for r in chosen),
                        "mean_enhancement_bytes": mean(chosen, "enhancement_bytes"),
                        "mean_total_bytes": mean(chosen, "total_bytes"),
                        "mean_total_bpp": mean(chosen, "total_bpp"),
                        "mean_total_kbps": mean(chosen, "total_kbps"),
                        "mean_actual_nonzero_frames": mean(chosen, "actual_nonzero_frames")}
                for metric in METRICS:
                    item[metric] = mean(chosen, metric)
                    item[f"{metric}_minus_ZERO"] = mean(chosen, f"{metric}_minus_ZERO")
                worst_psnr = min(chosen, key=lambda r: float(r["psnr_minus_ZERO"]))
                worst_lpips = max(chosen, key=lambda r: float(r["lpips_minus_ZERO"]))
                item["worst_psnr_video"] = worst_psnr["video"]
                item["worst_psnr_minus_ZERO"] = worst_psnr["psnr_minus_ZERO"]
                item["worst_lpips_video"] = worst_lpips["video"]
                item["worst_lpips_minus_ZERO"] = worst_lpips["lpips_minus_ZERO"]
                summaries.append(item)
    csv_dump(out / f"{split}_aggregate.csv", summaries)
    return summaries


def method_differences(out, split, rows):
    pairs = [
        ("M2_MINUS_M1", "M2_E_FRAME", "M1_E_SEQUENCE"),
        ("M4_MINUS_M3", "M4_O_FRAME", "M3_O_SEQUENCE"),
        ("M4_MINUS_M2", "M4_O_FRAME", "M2_E_FRAME"),
        ("M2_GUARD_MINUS_M2", "M2_E_FRAME_GUARD", "M2_E_FRAME"),
        ("M4_GUARD_MINUS_M4", "M4_O_FRAME_GUARD", "M4_O_FRAME"),
        ("M4_GUARD_MINUS_M2_GUARD", "M4_O_FRAME_GUARD", "M2_E_FRAME_GUARD"),
    ]
    result = []
    for label, method_a, method_b in pairs:
        for alpha in CONFIG["budgets"]:
            for scope in ("P", "IP"):
                for entry in entries(split):
                    video = entry["video_id"]
                    a = next(r for r in rows if r["method"] == method_a and r["alpha"] == alpha and r["scope"] == scope and r["video"] == video)
                    b = next(r for r in rows if r["method"] == method_b and r["alpha"] == alpha and r["scope"] == scope and r["video"] == video)
                    row = {"split": split, "comparison": label, "method_a": method_a, "method_b": method_b,
                           "alpha": alpha, "scope": scope, "video": video,
                           "a_enhancement_bytes": a["enhancement_bytes"], "b_enhancement_bytes": b["enhancement_bytes"],
                           "enhancement_bytes_delta": float(a["enhancement_bytes"]) - float(b["enhancement_bytes"]),
                           "a_total_bytes": a["total_bytes"], "b_total_bytes": b["total_bytes"]}
                    for metric in METRICS:
                        row[f"{metric}_a_minus_b"] = float(a[metric]) - float(b[metric])
                    result.append(row)
    csv_dump(out / f"{split}_method_attribution.csv", result)


def original_points(split, video):
    prefix = "train2_full" if split == "Train2" else "val6"
    result = []
    for qp in range(4):
        row = next(r for r in read_csv(V12 / "results_v1" / f"{prefix}_ORIGINAL_FP32_q{qp}" / "summary.csv")
                   if r["video"] == video and r["scope"] == "IP")
        result.append(row)
    return sorted(result, key=lambda r: float(r["total_bpp"]))


def interpolate(points, bpp, metric):
    rates = np.array([float(r["total_bpp"]) for r in points])
    if bpp < rates.min() or bpp > rates.max():
        return None
    values = np.array([float(r[metric]) for r in points])
    return float(np.interp(np.log(bpp), np.log(rates), values))


def matched_rate(out, split, rows):
    result = []
    for row in (r for r in rows if r["scope"] == "IP"):
        points = original_points(split, row["video"])
        baseline = points[0]
        supported = float(points[0]["total_bpp"]) <= float(row["total_bpp"]) <= float(points[-1]["total_bpp"])
        item = {"split": split, "video": row["video"], "method": row["method"], "alpha": row["alpha"],
                "total_bytes": row["total_bytes"], "total_bpp": row["total_bpp"],
                "comparison_supported": supported, "original_min_bpp": points[0]["total_bpp"],
                "original_max_bpp": points[-1]["total_bpp"],
                "interpolation": "piecewise linear metric vs log(total_bpp); no extrapolation"}
        zero = next(r for r in rows if r["scope"] == "IP" and r["video"] == row["video"] and
                    r["method"] == row["method"] and r["alpha"] == 0)
        for metric in ("psnr", "ms_ssim", "lpips"):
            interp = interpolate(points, float(row["total_bpp"]), metric)
            item[f"zero_minus_original_qp0_{metric}"] = float(zero[metric]) - float(baseline[metric])
            item[f"enhanced_minus_zero_{metric}"] = float(row[metric]) - float(zero[metric])
            item[f"original_interpolated_{metric}"] = interp
            item[f"final_minus_original_interpolated_{metric}"] = None if interp is None else float(row[metric]) - interp
        result.append(item)
    csv_dump(out / f"{split}_matched_total_rate.csv", result)


def checks(out, split, all_rows, guard):
    failures = []
    for method in CONFIG["guard_methods"]:
        for entry in entries(split):
            video = entry["video_id"]
            sequence = [r for r in guard if r["method"] == method and r["video"] == video and r["scope"] == "P"]
            sequence.sort(key=lambda r: r["alpha"])
            for previous, current in zip(sequence, sequence[1:]):
                if float(current["d"]) > float(previous["d"]) + 1e-15:
                    failures.append(f"budget monotonicity {method} {video}")
    mapping = {"M2_E_FRAME_GUARD": "M2_E_FRAME", "M4_O_FRAME_GUARD": "M4_O_FRAME"}
    for guarded, unguarded in mapping.items():
        for row in (r for r in guard if r["method"] == guarded and r["scope"] == "P"):
            old = next(r for r in all_rows if r["method"] == unguarded and r["video"] == row["video"] and
                       r["alpha"] == row["alpha"] and r["scope"] == "P")
            if float(row["d"]) < float(old["d"]) - 1e-15:
                failures.append(f"guard beats superset {guarded} {row['video']} {row['alpha']}")
    for row in (r for r in guard if r["method"] == "M4_O_FRAME_GUARD" and r["scope"] == "P"):
        m2 = next(r for r in guard if r["method"] == "M2_E_FRAME_GUARD" and r["video"] == row["video"] and
                  r["alpha"] == row["alpha"] and r["scope"] == "P")
        if float(row["d"]) > float(m2["d"]) + 1e-15:
            failures.append(f"M4 guard worse than M2 guard {row['video']} {row['alpha']}")
    selected = read_csv(out / f"{split}_guard_selected_frames.csv")
    candidates = frozen_candidates(split)
    for row in selected:
        if float(row["mse"]) > next(float(z["mse"]) for z in candidates if z["video"] == row["video"] and z["frame"] == row["frame"] and z["candidate"] == "Z") + 1e-15:
            failures.append("selected MSE guard violation")
        if float(row["lpips"]) > next(float(z["lpips"]) for z in candidates if z["video"] == row["video"] and z["frame"] == row["frame"] and z["candidate"] == "Z") + 1e-15:
            failures.append("selected LPIPS guard violation")
    assert not failures, failures[:5]
    json_dump(out / f"{split}_sanity_checks.json", {"alpha_zero_exact_ZERO": all(float(r["enhancement_bytes"]) == 0 for r in guard if r["alpha"] == 0 and r["scope"] == "P"),
              "budget_objective_monotonic": True, "M4_guard_not_worse_than_M2_guard": True,
              "guard_not_better_than_unfiltered_superset": True,
              "all_selected_frames_meet_MSE_and_LPIPS_guard": True, "failures": []})


def run(out):
    for split in ("Train2", "Val6"):
        for suffix in ("guard_summary", "all_method_results", "aggregate", "method_attribution",
                       "matched_total_rate", "sanity_checks"):
            path = out / f"{split}_{suffix}.csv"
            if suffix == "sanity_checks":
                path = path.with_suffix(".json")
            path.unlink(missing_ok=True)
        guard = guard_summary(out, split)
        rows = full_results(out, split, guard)
        aggregate(out, split, rows)
        method_differences(out, split, rows)
        matched_rate(out, split, rows)
        checks(out, split, rows, guard)


if __name__ == "__main__":
    run(HOME / "results_v1")
