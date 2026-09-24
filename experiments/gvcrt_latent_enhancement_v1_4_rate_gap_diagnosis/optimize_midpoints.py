"""Stage C: bounded intermediate-lambda sampling with frozen networks."""
import concurrent.futures
import math

from bridge import *
import sparse_bitstream as ORS1


class Data:
    def __init__(self, bundle):
        self.bundle = bundle
        self.base = {}
        self.contexts = {}
        self.timings = []

    def get(self, spec):
        if spec["sample"] in self.contexts:
            return self.contexts[spec["sample"]]
        start = time.perf_counter()
        if spec["video"] not in self.base:
            entry = next(e for e in entries(spec["split"]) if e["video_id"] == spec["video"])
            self.base[spec["video"]] = base_rows(entry)
        row = self.base[spec["video"]][spec["frame"]]
        with torch.no_grad():
            ec = self.bundle.fixed.ell_c(row["ell"].cuda().float()).cpu()
        result = {"ell_c": ec, "q_recon": row["q_recon"].float(), "row": row, "spec": spec}
        self.timings.append({"sample": spec["sample"], "cached_base_state_seconds": time.perf_counter() - start})
        self.contexts[spec["sample"]] = result
        return result


class Metrics:
    def __init__(self, bundle):
        self.bundle = bundle
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def submit(self, source, reconstruction):
        with torch.no_grad():
            lpips = self.bundle.lp(source, reconstruction, normalize=True).item()
        a = source.detach().cpu().numpy()[0]
        b = reconstruction.detach().cpu().numpy()[0]
        def calculate():
            mse = np.square(a.astype(np.float64) - b.astype(np.float64)).mean().item()
            return {"mse": mse, "psnr": calc_psnr(a, b, data_range=1),
                    "ms_ssim": float(calc_msssim_rgb(a, b, data_range=1)), "lpips": lpips,
                    "L_image": mse + CONFIG["lambda_p"] * lpips}
        return self.pool.submit(calculate)


def actual_snapshots(folder, spec, step, symbols, bundle):
    frames = [None] + [torch.zeros(CONFIG["code_shape"], dtype=torch.int16) for _ in range(15)]
    frames[spec["frame"]] = symbols.cpu().short()
    old_path, new_path = folder / f"step_{step:03d}.orc2", folder / f"step_{step:03d}.ors1"
    old = ORC2.encode(old_path, base_path(spec["video"]), MODEL, 1, 3, DELTA, (8, 17, 30), frames, bundle.net.entropy)
    new = ORS1.encode(new_path, base_path(spec["video"]), MODEL, 3, DELTA, (8, 17, 30), frames, bundle.net.entropy)
    a = ORC2.decode(old_path, base_path(spec["video"]), MODEL, bundle.net.entropy)["frames"][spec["frame"]]
    b = ORS1.decode(new_path, base_path(spec["video"]), MODEL, bundle.net.entropy)["frames"][spec["frame"]]
    assert torch.equal(a, symbols.cpu().short()) and torch.equal(a, b)
    old_packet = old["frames"][spec["frame"]]["actual_bits"]
    new_packet = new["frames"][spec["frame"]]["actual_bits"]
    header_share = (ORC2.HEADER.size * 8 + 8) / 15
    return {"old_diagnostic_container_bits": old["file_bits"], "new_diagnostic_container_bits": new["file_bits"],
            "old_actual_frame_packet_bits": old_packet, "new_actual_frame_packet_bits": new_packet,
            "old_actual_frame_share_bits": old_packet + header_share,
            "new_actual_frame_share_bits": new_packet + header_share,
            "old_R_actual_bpp": (old_packet + header_share) / PIXELS,
            "new_R_actual_bpp": (new_packet + header_share) / PIXELS,
            "new_mode": new["frames"][spec["frame"]]["kind"], "integer_roundtrip_exact": True}


def optimize_one(out, tag, lam, spec, bundle, data, metrics):
    folder = out / "optimizations" / tag / spec["sample"]
    if folder.exists() and not (folder / "selected.pt").exists():
        attempt = 1
        interrupted = folder.with_name(folder.name + f"_interrupted_attempt{attempt}")
        while interrupted.exists():
            attempt += 1
            interrupted = folder.with_name(folder.name + f"_interrupted_attempt{attempt}")
        folder.rename(interrupted)
    folder.mkdir(parents=True)
    context_data = data.get(spec)
    context = bundle.context(context_data)
    context_hashes = tensor_hash(context[0]), tensor_hash(context[1])
    source = source_frame(spec["video"], spec["frame"])
    variable = nn.Parameter(torch.zeros(CONFIG["code_shape"], device="cuda"))
    optimizer = torch.optim.Adam([variable], lr=CONFIG["quant_domain_z_lr"],
                                 betas=tuple(CONFIG["adam_betas"]), eps=CONFIG["adam_eps"])
    records, cache, futures, cache_checks, hits = [], {}, [], [], 0
    torch.cuda.synchronize()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def differentiate():
        rounded = variable.detach().round()
        ORC2.checked_symbols(rounded)
        ste = rounded + variable - variable.detach()
        reconstruction, _ = bundle.render(ste, context)
        mse = F.mse_loss(reconstruction, source)
        lpips = bundle.lp(reconstruction, source, normalize=True).mean()
        image_loss = mse + CONFIG["lambda_p"] * lpips
        rate = bundle.net.entropy.bits(ste, DELTA) / PIXELS
        objective = image_loss + lam * rate
        gradient = torch.autograd.grad(objective, variable)[0]
        if not torch.isfinite(gradient).all() or not torch.isfinite(objective):
            raise FloatingPointError("Nonfinite midpoint objective")
        return gradient.detach(), {"MSE_float32": mse.item(), "LPIPS_float32": lpips.item(),
                                   "L_image_float32": image_loss.item(), "R_est_bpp": rate.item(),
                                   "objective_est": objective.item()}

    with (folder / "curve.jsonl").open("x") as log:
        for step in range(CONFIG["steps"] + 1):
            optimizer.zero_grad(set_to_none=True)
            key = tensor_hash(variable.detach().round())
            cache_hit = key in cache
            if cache_hit:
                gradient, values = cache[key]
                hits += 1
                if not cache_checks:
                    fresh, fresh_values = differentiate()
                    assert torch.allclose(gradient, fresh, atol=1e-7, rtol=1e-5)
                    assert abs(values["objective_est"] - fresh_values["objective_est"]) < 1e-8
                    cache_checks.append(float((gradient - fresh).abs().max()))
            else:
                gradient, values = differentiate()
                cache[key] = gradient, values
            if step == 0:
                assert gradient.norm() > 0
            curve = {"step": step, "lambda": lam, **values, "gradient_l2": gradient.double().norm().item(),
                     "hard_state_cache_hit": cache_hit}
            if step in CONFIG["snapshots"]:
                symbols = variable.detach().round().cpu().short()
                with torch.no_grad():
                    reconstruction, residual = bundle.render(symbols, context)
                    if step == 0:
                        assert torch.count_nonzero(residual) == 0
                record = {"tag": tag, "sample": spec["sample"], "video": spec["video"], "frame": spec["frame"],
                          "group": spec["group"], "split": spec["split"], "lambda": lam, "step": step,
                          "estimated_symbol_bits": values["R_est_bpp"] * PIXELS,
                          "symbol_nonzero_count": int(torch.count_nonzero(symbols)),
                          "symbol_nonzero_fraction": float(torch.count_nonzero(symbols) / symbols.numel()),
                          "elapsed_seconds": time.perf_counter() - started,
                          **actual_snapshots(folder, spec, step, symbols, bundle)}
                save_pt(folder / f"step_{step:03d}.pt", {"symbols": symbols, "u_hat": symbols.float() * DELTA,
                         "variable": variable.detach().cpu(), "spec": spec, "domain": "z = u/delta", "step": step})
                records.append(record)
                futures.append(metrics.submit(source, reconstruction))
                print("MID", tag, spec["sample"], step, record["old_actual_frame_share_bits"], flush=True)
            log.write(json.dumps(curve, allow_nan=False) + "\n")
            log.flush()
            if step < CONFIG["steps"]:
                variable.grad = gradient.clone()
                optimizer.step()
    references = json.loads((V13 / "results_v1/references" / f'{spec["sample"]}.json').read_text())
    for record, future in zip(records, futures):
        record.update(future.result())
        record["selection_objective"] = record["L_image"] + lam * record["old_R_actual_bpp"]
        for reference in ["ZERO", "ENCODER_CONTINUOUS", "ENCODER_QUANTIZED"]:
            for metric in ["psnr", "ms_ssim", "lpips"]:
                record[f"{metric}_minus_{reference}"] = record[metric] - references[reference][metric]
    chosen = min(records, key=lambda r: (r["selection_objective"], r["old_actual_frame_share_bits"], r["step"]))
    for record in records:
        record["selected"] = record["step"] == chosen["step"]
    save_pt(folder / "selected.pt", torch.load(folder / f'step_{chosen["step"]:03d}.pt', map_location="cpu", weights_only=False))
    csv_dump(folder / "checkpoints.csv", records)
    torch.cuda.synchronize()
    json_dump(folder / "checks.json", {"network_updates": 0, "code_updates": 150, "free_scalars": 4080,
              "optimization_seconds": time.perf_counter() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "hard_state_cache_hits": hits, "cache_equivalence_max_abs": cache_checks[0] if cache_checks else None,
              "selected_step": chosen["step"], "selection_uses_historical_old_actual_rate": True})
    assert (tensor_hash(context[0]), tensor_hash(context[1])) == context_hashes
    bundle.check()
    del variable, optimizer
    torch.cuda.empty_cache()
    return chosen


def imported_six(index, lam, spec):
    folder = historical_code_path(index, spec).parent
    row = next(r for r in read_csv(folder / "checkpoints.csv") if r["selected"])
    item = torch.load(folder / "selected.pt", map_location="cpu", weights_only=False)
    assert row["lambda"] == lam and torch.equal(item["symbols"], item["variable"].round().short())
    return item["symbols"], {"selected_step": int(row["step"]), "historical_file": str(folder / "selected.pt"),
                             "sha256": file_hash(folder / "selected.pt"), "reason": "exact lambda and fixed V1.3 rules"}


def sequence_evaluate(out, tag, lam, codes, bundle, data, metrics):
    folder = out / "sequences" / tag
    if (folder / "summary.csv").exists():
        return read_csv(folder / "summary.csv")
    recovered = out / "sequences" / f"{tag}_recovered_after_csv_schema_error"
    if (recovered / "summary.csv").exists():
        return read_csv(recovered / "summary.csv")
    if folder.exists() and not (folder / "summary.csv").exists():
        if not (out / "sequence_recovery.json").exists():
            json_dump(out / "sequence_recovery.json", {"original_partial_folder": str(folder),
                      "recovered_folder": str(recovered), "reason": "first CSV writer used I-row fields only; optimization artifacts remain valid"})
        folder = recovered
    folder.mkdir(parents=True)
    frame_rows, summary = [], []
    for entry in entries("Train2"):
        video = entry["video_id"]
        frames = [None] + [codes[(video, f)] for f in range(1, 16)]
        old_path, new_path = folder / f"{video}.orc2", folder / f"{video}.ors1"
        start = time.perf_counter()
        old = ORC2.encode(old_path, base_path(video), MODEL, 1, 3, DELTA, (8, 17, 30), frames, bundle.net.entropy)
        old_seconds = time.perf_counter() - start
        new = ORS1.encode(new_path, base_path(video), MODEL, 3, DELTA, (8, 17, 30), frames, bundle.net.entropy)
        old_dec = ORC2.decode(old_path, base_path(video), MODEL, bundle.net.entropy)
        new_dec = ORS1.decode(new_path, base_path(video), MODEL, bundle.net.entropy)
        assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(old_dec["frames"], new_dec["frames"]))
        base = data.base.get(video)
        if base is None:
            data.get(next(s for s in specs("Train2") if s["video"] == video))
            base = data.base[video]
        futures, pending, outputs = [], [], []
        with torch.no_grad():
            for frame in range(16):
                source = source_frame(video, frame)
                if frame == 0:
                    reconstruction = rgb01(base[frame]["x_base"]).cuda()
                    nonzero = False
                    reference = None
                else:
                    spec = next(s for s in specs("Train2") if s["video"] == video and s["frame"] == frame)
                    context = bundle.context(data.get(spec))
                    symbols = new_dec["frames"][frame]
                    reconstruction, residual = bundle.render(symbols, context)
                    nonzero = bool(torch.count_nonzero(symbols))
                    if not nonzero:
                        assert torch.count_nonzero(residual) == 0
                    reference = json.loads((V13 / "results_v1/references" / f'{spec["sample"]}.json').read_text())
                outputs.append(reconstruction.cpu())
                futures.append(metrics.submit(source, reconstruction))
                pending.append((frame, nonzero, reference, old["frames"][frame], new["frames"][frame]))
        rows = []
        for future, (frame, nonzero, reference, old_frame, new_frame) in zip(futures, pending):
            result = future.result()
            row = {"candidate": tag, "lambda": lam, "video": video, "frame": frame,
                   "type": "I" if frame == 0 else "P", "nonzero_enhancement_frame": nonzero,
                   "old_packet_bits": old_frame["actual_bits"], "new_packet_bits": new_frame["actual_bits"],
                   "new_mode": new_frame["kind"], **result}
            if reference:
                for ref in ["ZERO", "ENCODER_QUANTIZED", "ORIGINAL_QP0"]:
                    for metric in ["psnr", "ms_ssim", "lpips"]:
                        row[f"{metric}_minus_{ref}"] = row[metric] - reference[ref][metric]
            rows.append(row)
        frame_rows.extend(rows)
        for scope in ["P", "IP"]:
            selected = [r for r in rows if scope == "IP" or r["type"] == "P"]
            item = {"candidate": tag, "lambda": lam, "video": video, "scope": scope, "frames": len(selected),
                    "base_bits": base_path(video).stat().st_size * 8,
                    "old_enhancement_bits": old["file_bits"], "new_enhancement_bits": new["file_bits"],
                    "old_enhancement_fraction": old["file_bits"] / (base_path(video).stat().st_size * 8),
                    "new_enhancement_fraction": new["file_bits"] / (base_path(video).stat().st_size * 8),
                    "nonzero_enhancement_frame_fraction_P": float(np.mean([r["nonzero_enhancement_frame"] for r in rows[1:]])),
                    "old_encode_seconds": old_seconds, "new_encode_and_mode_select_seconds": new["encode_and_mode_select_seconds"],
                    "integer_roundtrip_exact": True}
            for metric in ["psnr", "ms_ssim", "lpips", "L_image"]:
                item[metric] = float(np.mean([r[metric] for r in selected]))
            if scope == "P":
                for ref in ["ZERO", "ENCODER_QUANTIZED", "ORIGINAL_QP0"]:
                    for metric in ["psnr", "ms_ssim", "lpips"]:
                        item[f"{metric}_minus_{ref}"] = float(np.mean([r[f"{metric}_minus_{ref}"] for r in selected]))
            summary.append(item)
        write_video(folder / f"{video}.mkv", outputs, entry["metadata"]["avg_frame_rate"])
    csv_dump(folder / "frames.csv", frame_rows)
    csv_dump(folder / "summary.csv", summary)
    return summary


def run_candidate(out, tag, lam, historical_index, bundle, data, metrics):
    codes, imported, timing = {}, [], []
    for spec in specs("Train2"):
        if historical_index is not None and spec["group"] == "network_train6":
            symbols, evidence = imported_six(historical_index, lam, spec)
            codes[(spec["video"], spec["frame"])] = symbols
            imported.append({"sample": spec["sample"], **evidence})
        else:
            selected_path = out / "optimizations" / tag / spec["sample"] / "selected.pt"
            if not selected_path.exists():
                optimize_one(out, tag, lam, spec, bundle, data, metrics)
            item = torch.load(selected_path, map_location="cpu", weights_only=False)
            codes[(spec["video"], spec["frame"])] = item["symbols"]
            timing.append({"sample": spec["sample"], "optimization_seconds": json.loads((out / "optimizations" / tag / spec["sample"] / "checks.json").read_text())["optimization_seconds"]})
    if imported and not (out / "optimizations" / tag / "reused_exact_six.json").exists():
        json_dump(out / "optimizations" / tag / "reused_exact_six.json", imported)
    if not (out / "optimizations" / tag / "optimization_timings.csv").exists():
        csv_dump(out / "optimizations" / tag / "optimization_timings.csv", timing)
    summary = sequence_evaluate(out, tag, lam, codes, bundle, data, metrics)
    return summary


def existing_candidates(out):
    result = []
    v13 = read_csv(V13 / "results_v1/sequence_metrics.csv")
    recode = read_csv(out / "same_integer_recode_sequences.csv")
    for index, lam in zip(CONFIG["historical_formal_lambda_indices"], CONFIG["historical_formal_lambdas"]):
        for video in [e["video_id"] for e in entries("Train2")]:
            metric = next(r for r in v13 if r["method"] == f"B_Train2_lambda{index}" and r["video"] == video and r["scope"] == "IP")
            rate = next(r for r in recode if r["candidate"] == f"OPT_lambda{index}" and r["video"] == video)
            result.append({"candidate": f"historical_lambda{index}", "lambda": lam, "video": video,
                           "base_bits": rate["base_bits"], "old_enhancement_bits": rate["old_bits"],
                           "new_enhancement_bits": rate["new_bits"], "old_enhancement_fraction": rate["old_enhancement_fraction"],
                           "new_enhancement_fraction": rate["new_enhancement_fraction"], "psnr": metric["psnr"],
                           "ms_ssim": metric["ms_ssim"], "lpips": metric["lpips"], "L_image": metric["L_image"],
                           "psnr_minus_ZERO": metric["psnr_minus_ZERO"], "lpips_minus_ZERO": metric["lpips_minus_ZERO"],
                           "nonzero_enhancement_frame_fraction_P": metric["nonzero_enhancement_frame_fraction_P"]})
    encoder_summary = json.loads((OLD / "results_v1/train2_full_CODED_l3/summary.json").read_text())
    encoder_frames = read_csv(OLD / "results_v1/train2_full_CODED_l3/frames.csv")
    for video in [e["video_id"] for e in entries("Train2")]:
        metric = next(r for r in encoder_summary if r["video"] == video and r["scope"] == "IP")
        frames = [r for r in encoder_frames if r["video"] == video]
        image_loss = float(np.mean([10 ** (-r["psnr"] / 10) + CONFIG["lambda_p"] * r["lpips"] for r in frames]))
        rate = next(r for r in recode if r["candidate"] == "ENCODER_l3" and r["video"] == video)
        result.append({"candidate": "encoder_l3", "lambda": None, "video": video,
                       "base_bits": rate["base_bits"], "old_enhancement_bits": rate["old_bits"],
                       "new_enhancement_bits": rate["new_bits"], "old_enhancement_fraction": rate["old_enhancement_fraction"],
                       "new_enhancement_fraction": rate["new_enhancement_fraction"], "psnr": metric["psnr"],
                       "ms_ssim": metric["ms_ssim"], "lpips": metric["lpips"], "L_image": image_loss,
                       "psnr_minus_ZERO": metric["psnr_gain_vs_ZERO_D"], "lpips_minus_ZERO": metric["lpips_gain_vs_ZERO_D"],
                       "nonzero_enhancement_frame_fraction_P": 1.0})
    return result


def gate_and_crossings(rows):
    decisions = []
    for fmt in ["old", "new"]:
        for alpha in CONFIG["budgets"]:
            selected = []
            for entry in entries("Train2"):
                video = entry["video_id"]
                feasible = [r for r in rows if r["video"] == video and r[f"{fmt}_enhancement_bits"] <= alpha * r["base_bits"]]
                zero_metric = next(r for r in read_csv(V13 / "results_v1/sequence_metrics.csv")
                                   if r["method"] == "B_Train2_ZERO_NO_STREAM" and r["video"] == video and r["scope"] == "IP")
                zero = {"candidate": "ZERO_NO_STREAM", "lambda": None, "video": video, "base_bits": base_path(video).stat().st_size * 8,
                        "old_enhancement_bits": 0, "new_enhancement_bits": 0, "L_image": float("inf"),
                        "psnr_minus_ZERO": 0.0, "lpips_minus_ZERO": 0.0, "nonzero_enhancement_frame_fraction_P": 0.0}
                zero["L_image"] = zero_metric["L_image"]
                chosen = min(feasible + [zero], key=lambda r: (r["L_image"], r[f"{fmt}_enhancement_bits"], r["candidate"]))
                selected.append(chosen)
            gain = float(np.mean([r["psnr_minus_ZERO"] for r in selected]))
            lpips = float(np.mean([r["lpips_minus_ZERO"] for r in selected]))
            passes = (alpha == .5 and all(r["candidate"] != "ZERO_NO_STREAM" for r in selected) and
                      gain >= .1 and lpips <= CONFIG["train2_gate"]["mean_ip_lpips_worsening_max"])
            decisions.append({"format": fmt, "budget": alpha, "video0_candidate": selected[0]["candidate"],
                              "video1_candidate": selected[1]["candidate"], "both_nonzero": all(r["candidate"] != "ZERO_NO_STREAM" for r in selected),
                              "mean_ip_psnr_gain": gain, "mean_ip_lpips_change": lpips, "Val6_gate_pass": passes})
    crossings = []
    for entry in entries("Train2"):
        video = entry["video_id"]
        ordered = sorted([r for r in rows if r["video"] == video and r["lambda"] is not None], key=lambda r: r["lambda"])
        for low, high in zip(ordered, ordered[1:]):
            if (low["new_enhancement_fraction"] - .5) * (high["new_enhancement_fraction"] - .5) < 0:
                crossings.append({"video": video, "low_lambda": low["lambda"], "high_lambda": high["lambda"]})
    return decisions, crossings


def normalize_new_summary(rows):
    normalized = []
    for row in rows:
        if row["scope"] != "IP":
            continue
        p = next(r for r in rows if r["video"] == row["video"] and r["scope"] == "P")
        normalized.append({**row, "psnr_minus_ZERO": p["psnr_minus_ZERO"] * 15 / 16,
                           "lpips_minus_ZERO": p["lpips_minus_ZERO"] * 15 / 16})
    return normalized


def run(out):
    bundle = Bundle(perceptual=True)
    data, metrics = Data(bundle), Metrics(bundle)
    all_rows = existing_candidates(out)
    sampled = []
    try:
        for index, lam in [(2, CONFIG["first_midpoint_lambdas"][0]), (4, CONFIG["first_midpoint_lambdas"][1])]:
            tag = f"midpoint_lambda{index}"
            rows = run_candidate(out, tag, lam, index, bundle, data, metrics)
            normalized = normalize_new_summary(rows)
            all_rows.extend(normalized); sampled.append({"tag": tag, "lambda": lam, "reason": "required first midpoint"})
        decisions, crossings = gate_and_crossings(all_rows)
        adaptive = 0
        while not any(r["Val6_gate_pass"] for r in decisions) and crossings and adaptive < 2:
            interval = min(crossings, key=lambda r: (r["high_lambda"] / r["low_lambda"], r["video"]))
            lam = math.sqrt(interval["low_lambda"] * interval["high_lambda"])
            if any(r["lambda"] is not None and math.isclose(lam, r["lambda"], rel_tol=1e-14) for r in all_rows):
                break
            adaptive += 1
            tag = f"adaptive_lambda{adaptive}"
            rows = run_candidate(out, tag, lam, None, bundle, data, metrics)
            all_rows.extend(normalize_new_summary(rows))
            sampled.append({"tag": tag, "lambda": lam, "reason": "measured adjacent candidates crossed 50% new-format budget",
                            "trigger_interval": interval})
            decisions, crossings = gate_and_crossings(all_rows)
        csv_dump(out / "all_train2_candidate_sequences.csv", all_rows)
        csv_dump(out / "train2_budget_feasibility.csv", decisions)
        json_dump(out / "bounded_lambda_sampling.json", {"sampled": sampled, "new_lambda_count": len(sampled),
                  "maximum": CONFIG["maximum_new_lambdas"], "remaining_crossings": crossings,
                  "stopped_because": "Train2 gate passed" if any(r["Val6_gate_pass"] for r in decisions) else "bounded candidates exhausted or no measured adjacent crossing",
                  "Val6_expand": any(r["Val6_gate_pass"] for r in decisions)})
        csv_dump(out / "stage_C_input_timings.csv", data.timings)
        bundle.check()
    finally:
        metrics.pool.shutdown(wait=True)
