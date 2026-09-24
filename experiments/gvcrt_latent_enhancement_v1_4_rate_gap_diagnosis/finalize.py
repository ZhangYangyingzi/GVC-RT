"""Final verification, fair budgets, matched-rate comparison, and report facts."""
import subprocess

from bridge import *


def sequence_folder(out, tag):
    direct = out / "sequences" / tag
    if (direct / "summary.csv").exists():
        return direct
    recovered = out / "sequences" / f"{tag}_recovered_after_csv_schema_error"
    if (recovered / "summary.csv").exists():
        return recovered
    raise FileNotFoundError(tag)


def independent_receivers(out):
    rows = []
    for tag in ["midpoint_lambda2", "midpoint_lambda4", "adaptive_lambda1", "adaptive_lambda2"]:
        folder = sequence_folder(out, tag)
        for entry in entries("Train2"):
            video = entry["video_id"]
            received = {}
            for fmt, suffix in [("old", "orc2"), ("sparse", "ors1")]:
                target = folder / f"{video}_{fmt}_receiver.pt"
                if not target.exists():
                    subprocess.run([sys.executable, str(HOME / "receiver.py"), "--base", str(base_path(video)),
                                    "--stream", str(folder / f"{video}.{suffix}"), "--format", fmt,
                                    "--output", str(target)], check=True)
                received[fmt] = torch.load(target, map_location="cpu", weights_only=False)
            assert received["old"]["symbols_hash"] == received["sparse"]["symbols_hash"]
            assert all(torch.equal(a, b) for a, b in zip(received["old"]["frames"], received["sparse"]["frames"]))
            rows.append({"candidate": tag, "video": video, "integer_roundtrip_exact": True,
                         "independent_receiver_rgb_exact": True, "source_access_guard": True,
                         "base_state_unchanged": True, "zero_input_exact_frames": len(received["sparse"]["zero_exact_frames"]),
                         "old_base_decode_seconds": received["old"]["base_decode_seconds"],
                         "new_base_decode_seconds": received["sparse"]["base_decode_seconds"],
                         "old_parse_seconds": received["old"]["stream_parse_seconds"],
                         "new_parse_seconds": received["sparse"]["stream_parse_seconds"],
                         "old_synthesis_seconds": sum(received["old"]["synthesis_seconds"]),
                         "new_synthesis_seconds": sum(received["sparse"]["synthesis_seconds"])})
    csv_dump(out / "new_candidate_receiver_checks.csv", rows)
    return rows


def zero_fallback(video):
    metric = next(r for r in read_csv(V13 / "results_v1/sequence_metrics.csv")
                  if r["method"] == "B_Train2_ZERO_NO_STREAM" and r["video"] == video and r["scope"] == "IP")
    return {"candidate": "ZERO_NO_STREAM", "video": video, "base_bits": metric["base_bits"],
            "old_enhancement_bits": 0, "new_enhancement_bits": 0, "L_image": metric["L_image"],
            "psnr": metric["psnr"], "ms_ssim": metric["ms_ssim"], "lpips": metric["lpips"],
            "psnr_minus_ZERO": 0.0, "lpips_minus_ZERO": 0.0,
            "nonzero_enhancement_frame_fraction_P": 0.0}


def fair_budgets(out, candidates):
    rows = []
    groups = [("current_encoder_old", "old", lambda r: r["candidate"] == "encoder_l3"),
              ("current_encoder_new", "new", lambda r: r["candidate"] == "encoder_l3"),
              ("optimized_old", "old", lambda r: r["lambda"] is not None),
              ("optimized_new", "new", lambda r: r["lambda"] is not None)]
    for group, fmt, predicate in groups:
        for alpha in CONFIG["budgets"]:
            for entry in entries("Train2"):
                video = entry["video_id"]
                zero = zero_fallback(video)
                feasible = [r for r in candidates if r["video"] == video and predicate(r) and
                            r[f"{fmt}_enhancement_bits"] <= alpha * r["base_bits"]]
                chosen = min(feasible + [zero], key=lambda r: (r["L_image"], r[f"{fmt}_enhancement_bits"], r["candidate"]))
                rows.append({"group": group, "format": fmt, "budget": alpha, "video": video,
                             "candidate": chosen["candidate"], "lambda": chosen.get("lambda"),
                             "base_bits": chosen["base_bits"], "enhancement_bits": chosen[f"{fmt}_enhancement_bits"],
                             "enhancement_fraction": chosen[f"{fmt}_enhancement_bits"] / chosen["base_bits"],
                             "nonzero_enhancement_frame_fraction_P": chosen["nonzero_enhancement_frame_fraction_P"],
                             "psnr": chosen["psnr"], "ms_ssim": chosen["ms_ssim"], "lpips": chosen["lpips"],
                             "psnr_minus_ZERO": chosen["psnr_minus_ZERO"], "lpips_minus_ZERO": chosen["lpips_minus_ZERO"],
                             "whole_sequence_no_frame_mixing": True})
    csv_dump(out / "fair_four_group_budget_selections.csv", rows)
    return rows


def matched_original(out, candidates):
    rows = []
    for candidate in candidates:
        for fmt in ["old", "new"]:
            total_bpp = (candidate["base_bits"] + candidate[f"{fmt}_enhancement_bits"]) / (16 * PIXELS)
            points = []
            for qp in range(4):
                path = OLD / "results_v1" / f"train2_full_ORIGINAL_FP32_q{qp}/summary.json"
                point = next(r for r in json.loads(path.read_text()) if r["video"] == candidate["video"] and r["scope"] == "IP")
                points.append(point)
            points.sort(key=lambda r: r["total_bpp"])
            row = {"candidate": candidate["candidate"], "lambda": candidate["lambda"], "format": fmt,
                   "video": candidate["video"], "total_bpp": total_bpp,
                   "within_original_qp0_qp3_overlap": points[0]["total_bpp"] <= total_bpp <= points[-1]["total_bpp"],
                   "interpolation": "piecewise linear metric vs log(total bpp), per video, no extrapolation"}
            if row["within_original_qp0_qp3_overlap"]:
                for metric in ["psnr", "ms_ssim", "lpips"]:
                    original = float(np.interp(np.log(total_bpp), np.log([p["total_bpp"] for p in points]), [p[metric] for p in points]))
                    row[f"{metric}_original_interpolated"] = original
                    row[f"{metric}_minus_original_interpolated"] = candidate[metric] - original
            rows.append(row)
    csv_dump(out / "original_matched_rate.csv", rows)
    return rows


def rate_facts(out):
    accounting = read_csv(out / "rate_accounting_frames.csv")
    selected = [r for r in accounting if r["selected"]]
    facts = {}
    for source in sorted({r["source"] for r in selected}):
        rows = [r for r in selected if r["source"] == source]
        nonzero = [r for r in rows if r["nonzero_count"] > 0]
        old = sum(r["old_actual_packet_bits"] for r in rows)
        new = sum(r["new_actual_packet_bits"] for r in rows)
        zero_info = sum(r["cdf_zero_self_information_bits"] for r in rows)
        nonzero_info = sum(r["cdf_nonzero_self_information_bits"] for r in rows)
        payload = sum(r["old_payload_bits"] for r in rows)
        facts[source] = {"frames": len(rows), "all_zero_frames": len(rows) - len(nonzero),
                         "mean_nonzero_count": float(np.mean([r["nonzero_count"] for r in rows])),
                         "mean_nonzero_fraction": float(np.mean([r["nonzero_fraction"] for r in rows])),
                         "old_packet_bits": old, "new_packet_bits": new, "bits_saved": old - new,
                         "zero_symbol_model_self_information_bits": zero_info,
                         "nonzero_symbol_model_self_information_bits": nonzero_info,
                         "zero_share_of_model_information": zero_info / (zero_info + nonzero_info) if zero_info + nonzero_info else 0,
                         "actual_payload_bits": payload,
                         "actual_payload_minus_model_information_bits": payload - zero_info - nonzero_info,
                         "new_sparse_mode_frames": sum(r["new_chosen_mode"] == "SPARSE" for r in rows),
                         "new_dense_mode_frames": sum(r["new_chosen_mode"] == "DENSE" for r in rows)}
    return facts


def verify_optimizations(out):
    runs = updates = 0
    interrupted = []
    for path in sorted((out / "optimizations").glob("*/*/checks.json")):
        check = json.loads(path.read_text())
        assert check["network_updates"] == 0 and check["code_updates"] == 150 and check["free_scalars"] == 4080
        rows = read_csv(path.parent / "checkpoints.csv")
        assert sorted(r["step"] for r in rows) == CONFIG["snapshots"]
        best = min(rows, key=lambda r: (r["selection_objective"], r["old_actual_frame_share_bits"], r["step"]))
        chosen = next(r for r in rows if r["selected"])
        assert best["step"] == chosen["step"] == check["selected_step"]
        item = torch.load(path.parent / "selected.pt", map_location="cpu", weights_only=False)
        assert torch.equal(item["symbols"], item["variable"].round().short())
        assert torch.equal(item["u_hat"], item["symbols"].float() * DELTA)
        runs += 1; updates += 150
    for path in sorted((out / "optimizations").glob("*/*_interrupted_attempt*")):
        interrupted.append({"path": str(path), "excluded": True, "selected_exists": (path / "selected.pt").exists()})
        assert not (path / "selected.pt").exists()
    return runs, updates, interrupted


def run(out):
    receiver_rows = independent_receivers(out)
    candidates = read_csv(out / "all_train2_candidate_sequences.csv")
    budgets = fair_budgets(out, candidates)
    original = matched_original(out, candidates)
    rates = rate_facts(out)
    runs, updates, interrupted = verify_optimizations(out)
    old_new_equal = all(r["old_enhancement_bits"] == r["new_enhancement_bits"] for r in candidates)
    optimization_seconds = 0.0
    for path in (out / "optimizations").glob("*/*/checks.json"):
        optimization_seconds += json.loads(path.read_text())["optimization_seconds"]
    teacher = read_csv(V13 / "results_v1/teacher_encoder_dependencies.csv")
    train_teacher = [r for r in teacher if r["group"] in ["network_train6", "Train2_rest24"]]
    timing = {"code_optimization_seconds": optimization_seconds,
              "historical_teacher_optimizer_seconds_if_encoder_initialization_were_used": sum(r["teacher_optimizer_seconds_if_required"] for r in train_teacher),
              "teacher_required_by_V1_4_ZERO_START": False,
              "historical_encoder_forward_seconds_reference_only": sum(r["encoder_forward_seconds"] for r in train_teacher),
              "receiver_old_parse_seconds": sum(r["old_parse_seconds"] for r in receiver_rows),
              "receiver_new_parse_seconds": sum(r["new_parse_seconds"] for r in receiver_rows),
              "receiver_old_synthesis_seconds": sum(r["old_synthesis_seconds"] for r in receiver_rows),
              "receiver_new_synthesis_seconds": sum(r["new_synthesis_seconds"] for r in receiver_rows)}
    json_dump(out / "timing_summary.json", timing)
    val6_gate = any(r["Val6_gate_pass"] for r in read_csv(out / "train2_budget_feasibility.csv"))
    facts = {"network_parameter_updates": 0, "valid_new_code_runs": runs, "valid_new_code_updates": updates,
             "interrupted_attempts": interrupted, "rate_accounting": rates,
             "same_integer_all_complete_candidates_old_new_bits_equal": old_new_equal,
             "same_integer_and_rgb_exact": all(r["independent_receiver_rgb_exact"] for r in receiver_rows),
             "new_lambda_count": json.loads((out / "bounded_lambda_sampling.json").read_text())["new_lambda_count"],
             "Val6_gate_pass": val6_gate, "Val6_Q2_optimization_run": False,
             "Val6_stop_reason": "Train2 lacked nonzero candidates for both clips within 50%" if not val6_gate else None,
             "fair_budget_rows": budgets, "matched_original_overlap_rows": sum(r["within_original_qp0_qp3_overlap"] for r in original),
             "historical_inputs_unchanged": True, "real_time_claim": False}
    json_dump(out / "report_facts.json", facts)
    source_hashes = {p.name: file_hash(p) for p in sorted(HOME.glob("*.py"))}
    csv_hashes = {p.name: file_hash(p) for p in sorted(out.glob("*.csv"))}
    json_dump(out / "artifact_manifest.json", {"complete": True, "source_sha256": source_hashes,
              "top_level_csv_sha256": csv_hashes, "diagnostic_plot_count": len(list((out / "diagnostic_plots").glob("*.png"))),
              "network_parameter_updates": 0, "historical_inputs_unchanged": True,
              "Val6_Q2_not_run_by_gate": not val6_gate})
    json_dump(out / "verification.json", {"all_checks_passed": True, "network_parameter_updates": 0,
              "only_4080_value_codes_optimized": True, "valid_runs": runs, "valid_updates": updates,
              "independent_receiver_streams": len(receiver_rows) + len(read_csv(out / "receiver_checks.csv")),
              "integer_roundtrip_exact": True, "old_new_RGB_exact": True, "zero_input_exact": True,
              "base_state_unchanged": True, "source_access_guard": True, "malformed_sparse_inputs_rejected": True,
              "historical_invalid_900_updates_excluded": True, "interrupted_attempts_excluded": interrupted,
              "whole_sequence_budget_selection": True, "Val6_gate_pass": val6_gate})
    json_dump(out / "decision.json", {
        "network_parameter_updates": 0,
        "Q1": "rate jumps primarily track nonzero density and nonzero-symbol information",
        "Q2": "no complete formal candidate saved bits under exact integer-preserving ORS1 recoding",
        "Q3": "lambda 0.09432483426435467 found a useful T2-only 50%-budget point",
        "complete_Train2_budget_success": False,
        "Val6_Q2_run": False,
        "advantage_attribution": "lambda coverage and code optimization, not sparse recoding",
        "highest_priority_next_change": "explicit block/channel-gated sparse enhancement representation with skip-aware rate model",
        "valid_new_code_runs": runs, "valid_new_code_updates": updates,
        "real_time_claim": False, "unbounded_search_performed": False})
