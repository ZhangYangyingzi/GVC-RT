"""Candidate measurement, exact allocation, and real mixed-ORC2 repacking."""
import math
from decimal import Decimal, ROUND_FLOOR

from allocator import exact_frame_dp, exact_sequence
from bridge import *
from candidates import load_video_candidates

ORDER = CONFIG["candidate_order"]
OVERHEAD = ORC2.HEADER.size + 1


def budget_bytes(alpha, base_bytes):
    return int((Decimal(str(alpha)) * Decimal(base_bytes)).to_integral_value(rounding=ROUND_FLOOR))


def candidate_measurements(out, split, bundle, data, metrics):
    folder = out / "candidate_streams" / split
    folder.mkdir(parents=True, exist_ok=True)
    all_rows, provenance, symbols_by_video = [], [], {}
    for entry in entries(split):
        video = entry["video_id"]
        symbols, evidence = load_video_candidates(split, video, bundle, out)
        symbols_by_video[video] = symbols; provenance.extend(evidence)
        packet = {}
        for name in ORDER:
            path = folder / f"{video}_{name}.orc2"
            frames = [None] + [symbols[f][name] for f in range(1, 16)]
            coded = ORC2.encode(path, base_path(video), MODEL, 1, 3, DELTA, (8, 17, 30), frames,
                                bundle.net.entropy)
            decoded = ORC2.decode(path, base_path(video), MODEL, bundle.net.entropy)
            assert all(a is None if b is None else torch.equal(a, b) for a, b in zip(decoded["frames"], frames))
            packet[name] = coded
        futures, pending = [], []
        for frame in range(1, 16):
            spec = next(s for s in specs(split) if s["video"] == video and s["frame"] == frame)
            context = bundle.context(data.get(spec)); source = source_frame(video, frame)
            for name in ORDER:
                with torch.no_grad(): reconstruction, residual = bundle.render(symbols[frame][name], context)
                if name == "Z":
                    assert torch.count_nonzero(residual) == 0
                futures.append(metrics.submit(source, reconstruction))
                pending.append((frame, name, symbols[frame][name], packet[name]["frames"][frame]))
        for future, (frame, name, symbols_tensor, packet_row) in zip(futures, pending):
            all_rows.append({"split": split, "video": video, "frame": frame, "candidate": name,
                             "packet_bytes": packet_row["actual_bits"] // 8,
                             "packet_kind": packet_row["kind"],
                             "nonzero_symbols": int(torch.count_nonzero(symbols_tensor)), **future.result()})
    csv_dump(out / f"{split}_candidate_frames.csv", all_rows)
    csv_dump(out / f"{split}_candidate_provenance.csv", provenance)
    return all_rows, symbols_by_video


def allocate(out, split, rows, symbols_by_video, bundle):
    stream_folder = out / "mixed_streams" / split
    stream_folder.mkdir(parents=True, exist_ok=True)
    allocations, choices = [], []
    methods = (("M1_E_SEQUENCE", "sequence", ["Z", "E"]),
               ("M2_E_FRAME", "frame", ["Z", "E"]),
               ("M3_O_SEQUENCE", "sequence", ORDER),
               ("M4_O_FRAME", "frame", ORDER))
    for entry in entries(split):
        video = entry["video_id"]
        base_bytes = base_path(video).stat().st_size
        by_frame = {f: {r["candidate"]: r for r in rows if r["video"] == video and r["frame"] == f}
                    for f in range(1, 16)}
        for method, kind, allowed in methods:
            for alpha in CONFIG["budgets"]:
                budget = budget_bytes(alpha, base_bytes)
                if kind == "frame":
                    frame_candidates = [[{"name": name, "bytes": int(by_frame[f][name]["packet_bytes"]),
                                          "d": by_frame[f][name]["d"]} for name in allowed]
                                        for f in range(1, 16)]
                    selected = exact_frame_dp(frame_candidates, budget, OVERHEAD, ORDER)
                else:
                    sequence_candidates = [{"name": "Z", "bytes": 0,
                        "d": sum(by_frame[f]["Z"]["d"] for f in range(1, 16)), "choices": ["Z"] * 15}]
                    for name in allowed:
                        if name == "Z": continue
                        sequence_candidates.append({"name": name, "bytes": OVERHEAD + sum(int(by_frame[f][name]["packet_bytes"]) for f in range(1, 16)),
                            "d": sum(by_frame[f][name]["d"] for f in range(1, 16)), "choices": [name] * 15})
                    selected = exact_sequence(sequence_candidates, budget, ["Z"] + [n for n in ORDER if n != "Z"])
                label = str(alpha).replace(".", "p")
                path = stream_folder / f"{video}_{method}_a{label}.orc2"
                if selected["stream_sent"]:
                    frames = [None] + [symbols_by_video[video][f][name] for f, name in enumerate(selected["choices"], 1)]
                    coded = ORC2.encode(path, base_path(video), MODEL, 1, 3, DELTA, (8, 17, 30), frames,
                                        bundle.net.entropy)
                    decoded = ORC2.decode(path, base_path(video), MODEL, bundle.net.entropy)
                    assert all(a is None if b is None else torch.equal(a, b) for a, b in zip(decoded["frames"], frames))
                    actual = path.stat().st_size
                    assert actual == selected["actual_bytes"] == coded["file_bits"] // 8
                    stream_path, stream_hash = str(path), file_hash(path)
                else:
                    actual, stream_path, stream_hash = 0, None, None
                assert actual <= budget
                allocations.append({"split": split, "video": video, "method": method, "alpha": alpha,
                                    "base_bytes": base_bytes, "budget_bytes": budget, "actual_bytes": actual,
                                    "enhancement_fraction": actual / base_bytes, "stream_sent": selected["stream_sent"],
                                    "objective_sum_P": selected["objective"], "stream": stream_path,
                                    "stream_sha256": stream_hash})
                for frame, name in enumerate(selected["choices"], 1):
                    choices.append({"split": split, "video": video, "method": method, "alpha": alpha,
                                    "frame": frame, "candidate": name, **by_frame[frame][name]})
    csv_dump(out / f"{split}_allocations.csv", allocations)
    csv_dump(out / f"{split}_selected_frames.csv", choices)
    return allocations, choices


def run(out, split):
    bundle = Bundle(True); data = Data(bundle); metrics = Metrics(bundle)
    try:
        rows, symbols = candidate_measurements(out, split, bundle, data, metrics)
        allocations, choices = allocate(out, split, rows, symbols, bundle)
        csv_dump(out / f"{split}_evaluation_input_timings.csv", data.timings)
        bundle.check()
        return allocations, choices
    finally:
        metrics.pool.shutdown(wait=True)
