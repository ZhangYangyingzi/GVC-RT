"""Stages A/B: explain historical rate and losslessly recode identical integer grids."""
import math
import subprocess
import zlib

from bridge import *
from enhancement import arithmetic_encode
import sparse_bitstream as ORS1


FORMAL = [(1, CONFIG["historical_formal_lambdas"][0]),
          (3, CONFIG["historical_formal_lambdas"][1]),
          (5, CONFIG["historical_formal_lambdas"][2])]


def arithmetic_breakdown(raw, cdfs, per_channel):
    low, high, pending, emitted = 0, (1 << 32) - 1, 0, 0
    for index, symbol in enumerate(raw):
        cdf = cdfs[index // per_channel]
        span = high - low + 1
        high = low + span * cdf[symbol + 1] // 65536 - 1
        low += span * cdf[symbol] // 65536
        while True:
            if high < (1 << 31):
                emitted += 1 + pending
                pending = 0
            elif low >= (1 << 31):
                emitted += 1 + pending
                pending = 0
                low -= 1 << 31
                high -= 1 << 31
            elif low >= (1 << 30) and high < (3 << 30):
                pending += 1
                low -= 1 << 30
                high -= 1 << 30
            else:
                break
            low *= 2
            high = high * 2 + 1
    emitted += 1 + (pending + 1)
    before_padding = emitted + 32
    padding = (-before_padding) % 8
    return {"arithmetic_body_and_finalization_bits": emitted,
            "explicit_termination_lookahead_bits": 32, "byte_alignment_bits": padding,
            "payload_bits": before_padding + padding}


def cdf_information(symbols, cdfs):
    flat = symbols.flatten().tolist()
    per_channel = symbols.shape[-2] * symbols.shape[-1]
    zero_bits = nonzero_bits = 0.0
    for index, value in enumerate(flat):
        cdf = cdfs[index // per_channel]
        mass = (cdf[value + 129] - cdf[value + 128]) / 65536
        bits = -math.log2(mass)
        if value == 0:
            zero_bits += bits
        else:
            nonzero_bits += bits
    return zero_bits, nonzero_bits


def metric_maps():
    opt = read_csv(V13 / "results_v1/optimization_checkpoints.csv")
    opt_map = {(r["stage"], r["sample"], int(r["step"])): r for r in opt if r["quantized"]}
    enc = read_csv(OLD / "results_v1/train2_full_CODED_l3/frames.csv")
    enc_map = {(r["video"], int(r["frame"])): r for r in enc if r["type"] == "P"}
    return opt_map, enc_map


def symbol_summary(symbols):
    flat = symbols.flatten()
    nz = flat[flat != 0].double()
    return {"symbol_count": flat.numel(), "nonzero_count": int(nz.numel()),
            "nonzero_fraction": nz.numel() / flat.numel(),
            "nonzero_min": int(nz.min()) if nz.numel() else None,
            "nonzero_max": int(nz.max()) if nz.numel() else None,
            "nonzero_mean": float(nz.mean()) if nz.numel() else None,
            "nonzero_mean_abs": float(nz.abs().mean()) if nz.numel() else None,
            "sparsity_class": "ALL_ZERO" if not nz.numel() else ("SPARSE_NONZERO" if nz.numel() / flat.numel() <= .25 else "HIGH_NONZERO")}


def describe(symbols, entropy, cdfs):
    symbols = ORC2.checked_symbols(symbols)
    summary = symbol_summary(symbols)
    estimated = float(entropy.bits(symbols.cuda().float(), DELTA).item())
    zero_info, nonzero_info = cdf_information(symbols, cdfs)
    packets = ORS1.packet_candidates(symbols, entropy, DELTA)
    if "zero" in packets:
        old_packet = new_packet = packets["zero"]
        chosen_mode = "ZERO"
        breakdown = {"arithmetic_body_and_finalization_bits": 0, "explicit_termination_lookahead_bits": 0,
                     "byte_alignment_bits": 0, "payload_bits": 0}
        dense_bits = sparse_bits = 8
    else:
        old_packet = packets["dense"]
        new_packet = min((packets["dense"], packets["sparse"]), key=lambda p: (len(p), p is packets["sparse"]))
        chosen_mode = "SPARSE" if new_packet is packets["sparse"] else "DENSE"
        raw = (symbols.flatten() + 128).tolist()
        breakdown = arithmetic_breakdown(raw, cdfs, symbols.shape[-2] * symbols.shape[-1])
        assert breakdown["payload_bits"] == (len(old_packet) - 9) * 8
        dense_bits, sparse_bits = len(packets["dense"]) * 8, len(packets["sparse"]) * 8
    return {**summary, "shape": "1x8x17x30", "estimated_differentiable_bits": estimated,
            "cdf_zero_self_information_bits": zero_info, "cdf_nonzero_self_information_bits": nonzero_info,
            "cdf_total_self_information_bits": zero_info + nonzero_info,
            "old_actual_packet_bits": len(old_packet) * 8, "old_frame_header_bits": 8 if summary["nonzero_count"] == 0 else 72,
            "old_payload_bits": breakdown["payload_bits"], **breakdown,
            "actual_payload_minus_cdf_information_bits": breakdown["payload_bits"] - zero_info - nonzero_info,
            "new_dense_candidate_packet_bits": dense_bits, "new_sparse_candidate_packet_bits": sparse_bits,
            "new_chosen_mode": chosen_mode, "new_actual_packet_bits": len(new_packet) * 8,
            "packet_bits_saved": len(old_packet) * 8 - len(new_packet) * 8}


def load_historical_selected(index, spec):
    item = torch.load(historical_code_path(index, spec), map_location="cpu", weights_only=False)
    assert item["domain"] == "z = u/delta"
    assert torch.equal(item["symbols"], item["variable"].round().short())
    assert torch.equal(item["u_hat"], item["symbols"].float() * DELTA)
    return item["symbols"]


def encode_full_pair(out, name, codes, bundle):
    rows = []
    folder = out / "same_integer_streams" / name
    folder.mkdir(parents=True, exist_ok=False)
    for entry in entries("Train2"):
        video = entry["video_id"]
        frames = [None] + [codes[(video, f)] for f in range(1, 16)]
        old_path, new_path = folder / f"{video}.orc2", folder / f"{video}.ors1"
        start = time.perf_counter()
        old_stats = ORC2.encode(old_path, base_path(video), MODEL, 1, 3, DELTA, (8, 17, 30), frames, bundle.net.entropy)
        old_seconds = time.perf_counter() - start
        new_stats = ORS1.encode(new_path, base_path(video), MODEL, 3, DELTA, (8, 17, 30), frames, bundle.net.entropy)
        old_dec = ORC2.decode(old_path, base_path(video), MODEL, bundle.net.entropy)
        new_dec = ORS1.decode(new_path, base_path(video), MODEL, bundle.net.entropy)
        assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(old_dec["frames"], new_dec["frames"]))
        rows.append({"candidate": name, "video": video, "base_bits": base_path(video).stat().st_size * 8,
                     "old_bits": old_stats["file_bits"], "new_bits": new_stats["file_bits"],
                     "bits_saved": old_stats["file_bits"] - new_stats["file_bits"],
                     "fraction_saved": 1 - new_stats["file_bits"] / old_stats["file_bits"],
                     "old_enhancement_fraction": old_stats["file_bits"] / (base_path(video).stat().st_size * 8),
                     "new_enhancement_fraction": new_stats["file_bits"] / (base_path(video).stat().st_size * 8),
                     "old_encode_seconds": old_seconds,
                     "new_encode_and_mode_select_seconds": new_stats["encode_and_mode_select_seconds"],
                     "sparse_frames": sum(f["kind"] == "SPARSE" for f in new_stats["frames"]),
                     "dense_frames": sum(f["kind"] == "DENSE" for f in new_stats["frames"]),
                     "zero_frames": sum(f["kind"] == "ZERO" for f in new_stats["frames"]),
                     "integer_roundtrip_exact": True})
    return rows


def receiver_checks(out, candidates):
    rows = []
    for name in candidates:
        folder = out / "same_integer_streams" / name
        for entry in entries("Train2"):
            video = entry["video_id"]
            received = {}
            for fmt, suffix in [("old", "orc2"), ("sparse", "ors1")]:
                target = folder / f"{video}_{fmt}_receiver.pt"
                subprocess.run([sys.executable, str(HOME / "receiver.py"), "--base", str(base_path(video)),
                                "--stream", str(folder / f"{video}.{suffix}"), "--format", fmt,
                                "--output", str(target)], check=True)
                received[fmt] = torch.load(target, map_location="cpu", weights_only=False)
            assert received["old"]["symbols_hash"] == received["sparse"]["symbols_hash"]
            assert all(torch.equal(a, b) for a, b in zip(received["old"]["frames"], received["sparse"]["frames"]))
            rows.append({"candidate": name, "video": video, "integers_exact": True, "rgb_exact": True,
                         "source_access_guard": True, "base_state_unchanged": True,
                         "old_base_decode_seconds": received["old"]["base_decode_seconds"],
                         "new_base_decode_seconds": received["sparse"]["base_decode_seconds"],
                         "old_parse_seconds": received["old"]["stream_parse_seconds"],
                         "new_parse_seconds": received["sparse"]["stream_parse_seconds"],
                         "old_synthesis_seconds": sum(received["old"]["synthesis_seconds"]),
                         "new_synthesis_seconds": sum(received["sparse"]["synthesis_seconds"])})
    csv_dump(out / "receiver_checks.csv", rows)


def probes(out, bundle):
    probe_specs = [s for s in specs("Train2") if s["group"] == "network_train6"]
    rows = []
    folder = out / "rate_only_probes"
    folder.mkdir()
    for spec, count in zip(probe_specs, CONFIG["probe_nonzero_counts"]):
        symbols = torch.zeros(CONFIG["code_shape"], dtype=torch.int16)
        values = [1, -1, 2, -2]
        if count:
            positions = [math.floor(k * 4080 / count) for k in range(count)]
            for k, position in enumerate(positions):
                symbols.flatten()[position] = values[k % len(values)]
        frames = [None] + [torch.zeros_like(symbols) for _ in range(15)]
        frames[spec["frame"]] = symbols
        stem = f'{spec["sample"]}_nnz{count}'
        old = ORC2.encode(folder / f"{stem}.orc2", base_path(spec["video"]), MODEL, 1, 3, DELTA,
                          (8, 17, 30), frames, bundle.net.entropy)
        new = ORS1.encode(folder / f"{stem}.ors1", base_path(spec["video"]), MODEL, 3, DELTA,
                          (8, 17, 30), frames, bundle.net.entropy)
        a = ORC2.decode(folder / f"{stem}.orc2", base_path(spec["video"]), MODEL, bundle.net.entropy)["frames"]
        b = ORS1.decode(folder / f"{stem}.ors1", base_path(spec["video"]), MODEL, bundle.net.entropy)["frames"]
        assert all((x is None and y is None) or torch.equal(x, y) for x, y in zip(a, b))
        rows.append({"sample": spec["sample"], "nonzero_count": count, "fixed_position_rule": CONFIG["probe_rule"],
                     "old_file_bits": old["file_bits"], "new_file_bits": new["file_bits"],
                     "new_mode": new["frames"][spec["frame"]]["kind"], "roundtrip_exact": True,
                     "rate_only_not_reconstruction_RD": True})
    csv_dump(out / "rate_length_probes.csv", rows)


def malformed_tests(out, bundle):
    tests = []
    def rejected(name, fn):
        try:
            fn()
        except (ValueError, OverflowError):
            tests.append({"case": name, "rejected": True})
        else:
            raise AssertionError(f"Malformed case accepted: {name}")
    rejected("zero_gap", lambda: ORS1.decode_sparse_payload(b"\x00\x02", 1, 4080, (8, 17, 30)))
    rejected("out_of_range_position", lambda: ORS1.decode_sparse_payload(ORS1._put_varuint(4081) + b"\x02", 1, 4080, (8, 17, 30)))
    rejected("zero_sparse_value", lambda: ORS1.decode_sparse_payload(b"\x01\x00", 1, 4080, (8, 17, 30)))
    rejected("trailing_sparse_bytes", lambda: ORS1.decode_sparse_payload(b"\x01\x02\x00", 1, 4080, (8, 17, 30)))
    rejected("noncanonical_varuint", lambda: ORS1.decode_sparse_payload(b"\x81\x00\x02", 1, 4080, (8, 17, 30)))
    rejected("symbol_overflow", lambda: ORS1.sparse_payload(torch.full(CONFIG["code_shape"], 128)))
    csv_dump(out / "sparse_invalid_input_tests.csv", tests)


def plots(out, rows):
    import cv2
    plot_dir = out / "diagnostic_plots"
    plot_dir.mkdir(exist_ok=True)
    points = [r for r in rows if r["nonzero_count"] > 0]
    definitions = [
        ("actual_bits_vs_nonzero.png", "nonzero_count", "old_actual_packet_bits", "Nonzero symbols", "Old actual packet bits"),
        ("actual_bits_vs_step.png", "step", "old_actual_packet_bits", "Optimization step", "Old actual packet bits"),
        ("psnr_vs_actual_bits.png", "old_actual_packet_bits", "psnr", "Old actual packet bits", "PSNR"),
        ("lpips_vs_actual_bits.png", "old_actual_packet_bits", "lpips", "Old actual packet bits", "LPIPS"),
        ("estimated_vs_actual_bits.png", "estimated_differentiable_bits", "old_actual_packet_bits", "Differentiable estimated bits", "Old actual packet bits")]
    for filename, xkey, ykey, xlabel, ylabel in definitions:
        subset = [r for r in points if r.get(xkey) is not None and r.get(ykey) is not None]
        canvas = np.full((620, 920, 3), 250, dtype=np.uint8)
        left, right, top, bottom = 90, 890, 45, 540
        cv2.rectangle(canvas, (left, top), (right, bottom), (60, 60, 60), 1)
        xs, ys = [float(r[xkey]) for r in subset], [float(r[ykey]) for r in subset]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        if xmax == xmin: xmax += 1
        if ymax == ymin: ymax += 1
        colors = [(35, 80, 200), (45, 160, 60), (180, 80, 40), (150, 50, 160)]
        for group_index, source in enumerate(sorted({r["source"] for r in subset})):
            group = [r for r in subset if r["source"] == source]
            color = colors[group_index % len(colors)]
            for row in group:
                x = int(left + (float(row[xkey]) - xmin) / (xmax - xmin) * (right - left))
                y = int(bottom - (float(row[ykey]) - ymin) / (ymax - ymin) * (bottom - top))
                cv2.circle(canvas, (x, y), 3, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, source, (110 + 195 * group_index, 585), cv2.FONT_HERSHEY_SIMPLEX,
                        .45, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, xlabel, (left, 575), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, ylabel, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"x [{xmin:.4g}, {xmax:.4g}]  y [{ymin:.4g}, {ymax:.4g}]",
                    (left, 28), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(plot_dir / filename), canvas):
            raise RuntimeError(f"Failed to write {filename}")


def run(out, bundle):
    cdfs, fingerprint = bundle.net.entropy.coder(DELTA)
    zero_probabilities = [(cdf[129] - cdf[128]) / 65536 for cdf in cdfs]
    if not (out / "cdf_zero_probabilities.json").exists():
        json_dump(out / "cdf_zero_probabilities.json", {"cdf_fingerprint_crc32": fingerprint,
                  "per_channel": zero_probabilities, "mean": float(np.mean(zero_probabilities)),
                  "min": min(zero_probabilities), "max": max(zero_probabilities),
                  "source": "current frozen V1.3/V1.2 checkpoint integer CDF"})
    opt_map, enc_map = metric_maps()
    rows, histograms = [], []
    train_specs = specs("Train2")
    # Current enhancement encoder l3 integer codes.
    encoder_codes = {}
    for entry in entries("Train2"):
        video = entry["video_id"]
        decoded = ORC2.decode(OLD / "results_v1/train2_full_CODED_l3" / f"{video}.orc",
                              base_path(video), MODEL, bundle.net.entropy)
        for frame in range(1, 16):
            symbols = decoded["frames"][frame]
            encoder_codes[(video, frame)] = symbols
            metric = enc_map[(video, frame)]
            row = {"source": "ENCODER_l3", "lambda": None, "selected": True, "video": video, "frame": frame,
                   "sample": f"{video}_f{frame:04d}", "step": None,
                   "psnr": metric["psnr"], "ms_ssim": metric["ms_ssim"], "lpips": metric["lpips"],
                   "psnr_minus_ZERO": metric["psnr_gain_vs_ZERO_D"], "lpips_minus_ZERO": metric["lpips_gain_vs_ZERO_D"],
                   **describe(symbols, bundle.net.entropy, cdfs)}
            rows.append(row)
            histograms.append({"source": row["source"], "sample": row["sample"],
                               "counts": {str(v): int((symbols == v).sum()) for v in torch.unique(symbols).tolist()}})
    formal_codes = {}
    for index, lam in FORMAL:
        formal_codes[index] = {}
        for spec in train_specs:
            for step in CONFIG["snapshots"]:
                item = torch.load(historical_code_path(index, spec, f"step_{step:03d}.pt"), map_location="cpu", weights_only=False)
                symbols = item["symbols"]
                metric = opt_map[(f'{"B6q" if spec["group"] == "network_train6" else "Bfull"}_lambda{index}', spec["sample"], step)]
                row = {"source": f"OPT_lambda{index}", "lambda": lam, "selected": bool(metric["selected"]),
                       "video": spec["video"], "frame": spec["frame"], "sample": spec["sample"], "step": step,
                       **{k: metric[k] for k in ["psnr", "ms_ssim", "lpips", "psnr_minus_ZERO", "lpips_minus_ZERO"]},
                       **describe(symbols, bundle.net.entropy, cdfs)}
                rows.append(row)
                histograms.append({"source": row["source"], "sample": row["sample"], "step": step,
                                   "counts": {str(v): int((symbols == v).sum()) for v in torch.unique(symbols).tolist()}})
            formal_codes[index][(spec["video"], spec["frame"])] = load_historical_selected(index, spec)
    if not (out / "rate_accounting_frames.csv").exists():
        csv_dump(out / "rate_accounting_frames.csv", rows)
    else:
        rows = read_csv(out / "rate_accounting_frames.csv")
    if not (out / "nonzero_value_histograms.json").exists():
        json_dump(out / "nonzero_value_histograms.json", histograms)
    plots(out, rows)
    stream_rows = encode_full_pair(out, "ENCODER_l3", encoder_codes, bundle)
    for index, _ in FORMAL:
        stream_rows.extend(encode_full_pair(out, f"OPT_lambda{index}", formal_codes[index], bundle))
    csv_dump(out / "same_integer_recode_sequences.csv", stream_rows)
    probes(out, bundle)
    malformed_tests(out, bundle)
    receiver_checks(out, ["ENCODER_l3"] + [f"OPT_lambda{i}" for i, _ in FORMAL])
    bundle.check()
    json_dump(out / "stage_AB_status.json", {"status": "COMPLETE", "network_updates": 0,
              "frames_in_formal_candidates": 120, "historical_optimization_checkpoints_analyzed": 450,
              "encoder_frames_analyzed": 30, "invalid_v13_domain_attempts_included": False,
              "same_integer_roundtrip_and_rgb_exact": True})
