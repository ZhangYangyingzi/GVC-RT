"""V1.6 guarded allocation over the frozen V1.5 candidate measurements."""
import math
from collections import Counter
from decimal import Decimal, ROUND_FLOOR

from bridge import *


def budget_bytes(alpha, base_bytes):
    return int((Decimal(str(alpha)) * Decimal(base_bytes)).to_integral_value(rounding=ROUND_FLOOR))


def candidate_rows(split):
    rows = read_csv(V15_RESULTS / f"{split}_candidate_frames.csv")
    assert len(rows) == len(entries(split)) * 15 * 4
    return rows


def repeatability_audit(out):
    """Re-render Train2 candidates; tolerance is frozen before any Val6 decision."""
    bundle, data, metrics = Bundle(True), Data(None), None
    data.bundle = bundle
    metrics = Metrics(bundle)
    historical = candidate_rows("Train2")
    rows = []
    try:
        for entry in entries("Train2"):
            video = entry["video_id"]
            symbols = load_video_candidates("Train2", video, bundle)
            for frame in range(1, 16):
                sample = next(s for s in specs("Train2") if s["video"] == video and s["frame"] == frame)
                context = bundle.context(data.get(sample))
                source = source_frame(video, frame)
                for name in ORDER:
                    old = next(r for r in historical if r["video"] == video and int(r["frame"]) == frame and r["candidate"] == name)
                    with torch.no_grad():
                        reconstruction, _ = bundle.render(symbols[frame][name], context)
                    current = metrics.submit(source, reconstruction).result()
                    rows.append({"video": video, "frame": frame, "candidate": name,
                                 "mse_old": old["mse"], "mse_repeat": current["mse"],
                                 "mse_abs_error": abs(current["mse"] - old["mse"]),
                                 "lpips_old": old["lpips"], "lpips_repeat": current["lpips"],
                                 "lpips_abs_error": abs(current["lpips"] - old["lpips"])})
        max_mse = max(r["mse_abs_error"] for r in rows)
        max_lpips = max(r["lpips_abs_error"] for r in rows)
        # Decisions use the original unrounded V1.5 measurements. A tolerance is
        # unnecessary because those immutable values are compared to each other.
        assert CONFIG["guard_tolerance"] == 0.0
        csv_dump(out / "Train2_metric_repeatability.csv", rows)
        json_dump(out / "guard_tolerance_freeze.json", {
            "rows_recomputed": len(rows), "max_mse_abs_error": max_mse,
            "max_lpips_abs_error": max_lpips, "frozen_tolerance": 0.0,
            "decision_basis": "same immutable unrounded V1.5 candidate table values",
            "reason": "no tolerance needed; repeatability error is recorded but not used to relax Val6 guards"})
        bundle.check()
    finally:
        metrics.pool.shutdown(wait=True)


def classify(candidate, zero):
    mse_delta = candidate["mse"] - zero["mse"]
    lpips_delta = candidate["lpips"] - zero["lpips"]
    mse_bad = mse_delta > CONFIG["guard_tolerance"]
    lpips_bad = lpips_delta > CONFIG["guard_tolerance"]
    if mse_bad and lpips_bad:
        reason = "MSE_AND_LPIPS_DEGRADE"
    elif mse_bad:
        reason = "MSE_ONLY_DEGRADES"
    elif lpips_bad:
        reason = "LPIPS_ONLY_DEGRADES"
    else:
        reason = "ALLOWED"
    return mse_delta, lpips_delta, reason


def run_guard_split(out, split):
    rows = candidate_rows(split)
    bundle = Bundle(False)
    stream_dir = out / "streams" / split
    stream_dir.mkdir(parents=True, exist_ok=True)
    filters, selections, allocations = [], [], []
    symbol_cache = {}
    methods = (("M2_E_FRAME_GUARD", ["Z", "E"]),
               ("M4_O_FRAME_GUARD", ORDER))
    for entry in entries(split):
        video = entry["video_id"]
        symbols = load_video_candidates(split, video, bundle)
        symbol_cache[video] = symbols
        by_frame = {frame: {r["candidate"]: r for r in rows
                            if r["video"] == video and int(r["frame"]) == frame}
                    for frame in range(1, 16)}
        allowed = {}
        for frame in range(1, 16):
            zero = by_frame[frame]["Z"]
            allowed[frame] = {"Z"}
            for name in ORDER:
                candidate = by_frame[frame][name]
                mse_delta, lpips_delta, reason = classify(candidate, zero)
                actual_nonzero = int(candidate["nonzero_symbols"]) > 0
                if reason == "ALLOWED":
                    allowed[frame].add(name)
                filters.append({"split": split, "video": video, "frame": frame,
                                "candidate": name, "packet_kind": candidate["packet_kind"],
                                "actual_nonzero_symbols": actual_nonzero,
                                "mse": candidate["mse"], "zero_mse": zero["mse"],
                                "mse_delta_signed": mse_delta,
                                "lpips": candidate["lpips"], "zero_lpips": zero["lpips"],
                                "lpips_delta_signed": lpips_delta, "guard_result": reason,
                                "allowed": reason == "ALLOWED"})
        base_bytes = base_path(video).stat().st_size
        for method, pool in methods:
            for alpha in CONFIG["budgets"]:
                budget = budget_bytes(alpha, base_bytes)
                frame_candidates = []
                for frame in range(1, 16):
                    frame_candidates.append([{"name": name,
                                              "bytes": int(by_frame[frame][name]["packet_bytes"]),
                                              "d": by_frame[frame][name]["d"]}
                                             for name in pool if name in allowed[frame]])
                result = exact_frame_dp(frame_candidates, budget, OVERHEAD_BYTES, ORDER)
                label = str(alpha).replace(".", "p")
                path = stream_dir / f"{video}_{method}_a{label}.orc2"
                if result["stream_sent"]:
                    frames = [None] + [symbols[frame][name]
                                       for frame, name in enumerate(result["choices"], 1)]
                    coded = ORC2.encode(path, base_path(video), MODEL, 1, CONFIG["level"], DELTA,
                                        tuple(CONFIG["code_shape"][1:]), frames, bundle.net.entropy)
                    decoded = ORC2.decode(path, base_path(video), MODEL, bundle.net.entropy)
                    assert all(a is None if b is None else torch.equal(a, b)
                               for a, b in zip(decoded["frames"], frames))
                    actual_bytes = path.stat().st_size
                    assert actual_bytes == result["actual_bytes"] == coded["file_bits"] // 8
                    stream, stream_sha = str(path), sha(path)
                else:
                    actual_bytes, stream, stream_sha = 0, None, None
                assert actual_bytes <= budget
                counts = Counter(result["choices"])
                allocations.append({"split": split, "video": video, "method": method,
                                    "alpha": alpha, "base_bytes": base_bytes,
                                    "enhancement_bytes": actual_bytes,
                                    "total_bytes": base_bytes + actual_bytes,
                                    "budget_bytes": budget, "budget_slack_bytes": budget - actual_bytes,
                                    "enhancement_over_base": actual_bytes / base_bytes,
                                    "total_bpp": (base_bytes + actual_bytes) * 8 / (16 * PIXELS),
                                    "total_kbps": (base_bytes + actual_bytes) * 8 / duration_seconds(split, video) / 1000,
                                    "objective_sum_P": result["objective"],
                                    "stream_sent": result["stream_sent"], "stream": stream,
                                    "stream_sha256": stream_sha,
                                    "selected_label_nonzero_frames": 15 - counts["Z"],
                                    "Z_count": counts["Z"], "E_count": counts["E"],
                                    "O1_count": counts["O1"], "O2_count": counts["O2"]})
                for frame, name in enumerate(result["choices"], 1):
                    candidate = by_frame[frame][name]
                    selections.append({"split": split, "video": video, "method": method,
                                       "alpha": alpha, "frame": frame, "candidate": name,
                                       "actual_nonzero_symbols": int(candidate["nonzero_symbols"]) > 0,
                                       "guard_allowed": name in allowed[frame], **candidate})
    csv_dump(out / f"{split}_candidate_filter.csv", filters)
    csv_dump(out / f"{split}_guard_allocations.csv", allocations)
    csv_dump(out / f"{split}_guard_selected_frames.csv", selections)
    bundle.check()


def run(out):
    repeatability_audit(out)
    run_guard_split(out, "Train2")
    run_guard_split(out, "Val6")


if __name__ == "__main__":
    setup()
    run(HOME / "results_v1")
