"""Aggregate V1.7 candidates, freeze the guard, and allocate OA17 streams."""
from collections import Counter
from decimal import Decimal, ROUND_FLOOR

from bridge import *


ZERO = "Z_ORIGINAL"


def budget_bytes(alpha, base_bytes):
    return int((Decimal(str(alpha)) * Decimal(base_bytes)).to_integral_value(rounding=ROUND_FLOOR))


def symbols_for(candidate):
    if candidate["candidate"] == ZERO:
        return torch.zeros(CONFIG["code_shape"], dtype=torch.int16)
    return torch.load(candidate["checkpoint"], map_location="cpu", weights_only=False)["symbols"].short()


def aggregate_split(out, split):
    optimized = read_csv(out / f"{split}_optimization_checkpoints.csv")
    expected = len(entries(split)) * 15 * 2 * len(CONFIG["snapshots"])
    assert len(optimized) == expected
    bundle, metrics = Bundle(True), None
    metrics = Metrics(bundle)
    rows = []
    try:
        for entry in entries(split):
            video = entry["video_id"]
            base = base_rows(entry)
            for frame in range(1, CONFIG["gop_frames"]):
                sample = f"{video}_f{frame:04d}"
                source = source_frame(video, frame)
                zero = torch.zeros(CONFIG["code_shape"], dtype=torch.int16)
                with torch.no_grad():
                    reconstruction, residual = render_original_anchor(bundle, zero, base[frame])
                zero_metrics = metrics.submit(source, reconstruction).result()
                rows.append({"split": split, "video": video, "frame": frame, "sample": sample,
                             "candidate": ZERO, "family": "Z", "step": None, "lambda": None,
                             "checkpoint": None, "integer_symbol_sha256": tensor_hash(zero),
                             "packet_bytes": packet_bytes_for_symbols(
                                 zero, split, video, frame, bundle, out / "tmp_packets"),
                             "nonzero_symbols": 0, "residual_linf": float(residual.abs().max()),
                             **zero_metrics})
                frame_rows = [r for r in optimized if r["video"] == video and int(r["frame"]) == frame]
                assert len(frame_rows) == 2 * len(CONFIG["snapshots"])
                for row in frame_rows:
                    family, step = row["candidate"], int(row["step"])
                    candidate = f"{family}_s{step:03d}"
                    assert candidate in CONFIG["candidate_order"]
                    rows.append({**row, "family": family, "candidate": candidate, "step": step})
        expected_rows = len(entries(split)) * 15 * len(CONFIG["candidate_order"])
        assert len(rows) == expected_rows
        csv_dump(out / f"{split}_candidate_frames.csv", rows, replace=True)
        bundle.check()
    finally:
        metrics.pool.shutdown(wait=True)
    return rows


def repeatability_audit(out):
    historical = read_csv(out / "Train2_candidate_frames.csv")
    assert len(historical) == len(entries("Train2")) * 15 * len(CONFIG["candidate_order"])
    bundle, metrics = Bundle(True), None
    metrics = Metrics(bundle)
    rows = []
    try:
        by_video = {entry["video_id"]: base_rows(entry) for entry in entries("Train2")}
        for old in historical:
            video, frame = old["video"], int(old["frame"])
            source = source_frame(video, frame)
            with torch.no_grad():
                reconstruction, _ = render_original_anchor(bundle, symbols_for(old), by_video[video][frame])
            current = metrics.submit(source, reconstruction).result()
            rows.append({"video": video, "frame": frame, "candidate": old["candidate"],
                         "mse_old": old["mse"], "mse_repeat": current["mse"],
                         "mse_abs_error": abs(current["mse"] - old["mse"]),
                         "lpips_old": old["lpips"], "lpips_repeat": current["lpips"],
                         "lpips_abs_error": abs(current["lpips"] - old["lpips"])})
        max_mse = max(r["mse_abs_error"] for r in rows)
        max_lpips = max(r["lpips_abs_error"] for r in rows)
        assert CONFIG["guard_tolerance"] == 0.0
        csv_dump(out / "Train2_metric_repeatability.csv", rows, replace=True)
        json_dump(out / "guard_tolerance_freeze.json", {
            "rows_recomputed": len(rows), "max_mse_abs_error": max_mse,
            "max_lpips_abs_error": max_lpips, "frozen_tolerance": 0.0,
            "decision_basis": "same immutable unrounded V1.7 candidate table values",
            "reason": "repeatability is audited, but no tolerance relaxes Val6 guard decisions"}, replace=True)
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


def allocate(frame_candidates, budget, allowed=None):
    order = ["Z" if name == ZERO else name for name in CONFIG["candidate_order"]]
    mapped = []
    for frame, candidates in enumerate(frame_candidates, 1):
        pool = []
        for candidate in candidates.values():
            if allowed is not None and candidate["candidate"] not in allowed[frame]:
                continue
            name = "Z" if candidate["candidate"] == ZERO else candidate["candidate"]
            pool.append({"name": name, "bytes": int(candidate["packet_bytes"]), "d": candidate["d"]})
        mapped.append(pool)
    result = exact_frame_dp(mapped, budget, profiled_overhead_bytes(), order)
    result["choices"] = [ZERO if name == "Z" else name for name in result["choices"]]
    return result


def run_allocation_split(out, split):
    rows = read_csv(out / f"{split}_candidate_frames.csv")
    bundle = Bundle(False)
    filters, selections, allocations = [], [], []
    stream_dir = out / "streams" / split
    stream_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries(split):
        video = entry["video_id"]
        by_frame = {frame: {r["candidate"]: r for r in rows
                            if r["video"] == video and int(r["frame"]) == frame}
                    for frame in range(1, 16)}
        assert all(list(candidates) == CONFIG["candidate_order"] for candidates in by_frame.values())
        allowed = {}
        for frame in range(1, 16):
            zero = by_frame[frame][ZERO]
            allowed[frame] = {ZERO}
            for name in CONFIG["candidate_order"]:
                candidate = by_frame[frame][name]
                mse_delta, lpips_delta, reason = classify(candidate, zero)
                if reason == "ALLOWED":
                    allowed[frame].add(name)
                filters.append({"split": split, "video": video, "frame": frame,
                                "candidate": name, "mse": candidate["mse"], "zero_mse": zero["mse"],
                                "mse_delta_signed": mse_delta, "lpips": candidate["lpips"],
                                "zero_lpips": zero["lpips"], "lpips_delta_signed": lpips_delta,
                                "guard_result": reason, "allowed": reason == "ALLOWED"})
        base_bytes = base_path(video).stat().st_size
        for method in CONFIG["methods"]:
            for alpha in CONFIG["budgets"]:
                budget = budget_bytes(alpha, base_bytes)
                result = allocate(by_frame, budget, allowed if method == "NEW_GUARD" else None)
                path = stream_dir / f"{video}_{method}_a{alpha_label(alpha)}.oa17"
                if result["stream_sent"]:
                    frames = [None] + [symbols_for(by_frame[frame][name])
                                       for frame, name in enumerate(result["choices"], 1)]
                    inner = out / "tmp_streams" / f"{video}_{method}_a{alpha_label(alpha)}.orc2"
                    inner.parent.mkdir(parents=True, exist_ok=True)
                    inner.unlink(missing_ok=True); path.unlink(missing_ok=True)
                    coded = encode_inner(inner, frames, video, bundle)
                    actual_bytes = wrap_stream(inner, path)
                    decoded = decode_profiled(path, video, bundle, out / "tmp_decode")
                    assert all(a is None if b is None else torch.equal(a, b)
                               for a, b in zip(decoded["frames"], frames))
                    assert actual_bytes == result["actual_bytes"] == coded["file_bits"] // 8 + WRAP.size
                    inner.unlink(missing_ok=True)
                    stream, stream_sha = str(path), sha(path)
                else:
                    actual_bytes, stream, stream_sha = 0, None, None
                assert actual_bytes <= budget
                counts = Counter(result["choices"])
                allocation = {"split": split, "video": video, "method": method, "alpha": alpha,
                              "base_bytes": base_bytes, "enhancement_bytes": actual_bytes,
                              "total_bytes": base_bytes + actual_bytes, "budget_bytes": budget,
                              "budget_slack_bytes": budget - actual_bytes,
                              "enhancement_over_base": actual_bytes / base_bytes,
                              "total_bpp": (base_bytes + actual_bytes) * 8 / (16 * PIXELS),
                              "total_kbps": (base_bytes + actual_bytes) * 8 /
                                            duration_seconds(split, video) / 1000,
                              "objective_sum_P": result["objective"], "stream_sent": result["stream_sent"],
                              "stream": stream, "stream_sha256": stream_sha,
                              "selected_label_nonzero_frames": 15 - counts[ZERO]}
                allocation.update({f"{name}_count": counts[name] for name in CONFIG["candidate_order"]})
                allocations.append(allocation)
                for frame, name in enumerate(result["choices"], 1):
                    candidate = by_frame[frame][name]
                    selections.append({"split": split, "video": video, "method": method,
                                       "alpha": alpha, "frame": frame,
                                       "guard_allowed": name in allowed[frame], **candidate})
    csv_dump(out / f"{split}_candidate_filter.csv", filters, replace=True)
    csv_dump(out / f"{split}_guard_allocations.csv", allocations, replace=True)
    csv_dump(out / f"{split}_guard_selected_frames.csv", selections, replace=True)
    bundle.check()


def run(out):
    aggregate_split(out, "Train2")
    aggregate_split(out, "Val6")
    repeatability_audit(out)
    self_test()
    run_allocation_split(out, "Train2")
    run_allocation_split(out, "Val6")


if __name__ == "__main__":
    setup()
    run(HOME / "results_v1")
