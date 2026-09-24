"""Optimize original-anchor enhancement code grids for Train2 and Val6."""
from bridge import *


def metric_sync(metrics, source, reconstruction):
    return metrics.submit(source, reconstruction).result()


def evaluate_symbols(out, split, name, lam, spec, symbols, bundle, metrics, step, elapsed, source, init_source):
    row = base_rows(next(e for e in entries(split) if e["video_id"] == spec["video"]))[int(spec["frame"])]
    with torch.no_grad():
        reconstruction, residual = render_original_anchor(bundle, symbols, row)
    packet = packet_bytes_for_symbols(symbols, split, spec["video"], int(spec["frame"]), bundle, out / "tmp_packets")
    result = {"split": split, "candidate": name, "lambda": lam, "sample": spec["sample"],
              "video": spec["video"], "frame": int(spec["frame"]), "step": step,
              "init_source": init_source, "nonzero_symbols": int(torch.count_nonzero(symbols)),
              "packet_bytes": packet, "frame_share_bytes": packet + profiled_overhead_bytes() / 15,
              "elapsed_seconds": elapsed, "integer_symbol_sha256": tensor_hash(symbols.cpu().short()),
              "residual_linf": float(residual.detach().abs().max())}
    result.update(metric_sync(metrics, source, reconstruction))
    result["d"] = result["mse"] + CONFIG["lambda_p"] * result["lpips"]
    result["selection_objective"] = result["d"] + lam * (result["frame_share_bytes"] * 8 / PIXELS)
    return result


def init_choice(split, candidate, spec, lam, bundle, metrics, source):
    zero = torch.zeros(CONFIG["code_shape"], dtype=torch.int16)
    legacy, path, status = load_legacy_symbols(split, candidate, spec["sample"])
    row = base_rows(next(e for e in entries(split) if e["video_id"] == spec["video"]))[int(spec["frame"])]
    choices = []
    for label, symbols in (("zero", zero), ("legacy", legacy if legacy is not None else zero)):
        with torch.no_grad():
            reconstruction, _ = render_original_anchor(bundle, symbols, row)
        m = metric_sync(metrics, source, reconstruction)
        rate = bundle.net.entropy.bits(symbols.cuda().float(), DELTA).item() / PIXELS
        d = m["mse"] + CONFIG["lambda_p"] * m["lpips"]
        choices.append({"label": label, "symbols": symbols.clone(), "mse": m["mse"], "lpips": m["lpips"],
                        "d": d, "estimated_rate": rate, "objective": d + lam * rate})
    chosen = min(choices, key=lambda c: (c["objective"], c["d"], int(torch.count_nonzero(c["symbols"]))))
    return chosen["symbols"], {"legacy_path": path, "legacy_status": status,
                               "chosen_initialization": chosen["label"],
                               "zero_initial_objective": choices[0]["objective"],
                               "legacy_initial_objective": choices[1]["objective"]}


def optimize_one(out, split, candidate, lam, spec, bundle, metrics):
    folder = out / "optimizations" / split / candidate / spec["sample"]
    if (folder / "checkpoints.csv").exists():
        return
    if folder.exists():
        attempt = 1
        moved = folder.with_name(folder.name + f"_interrupted_attempt{attempt}")
        while moved.exists():
            attempt += 1
            moved = folder.with_name(folder.name + f"_interrupted_attempt{attempt}")
        folder.rename(moved)
    folder.mkdir(parents=True, exist_ok=True)
    base = base_rows(next(e for e in entries(split) if e["video_id"] == spec["video"]))
    row = base[int(spec["frame"])]
    source = source_frame(spec["video"], int(spec["frame"]))
    initial_symbols, init_record = init_choice(split, candidate, spec, lam, bundle, metrics, source)
    variable = nn.Parameter(initial_symbols.cuda().float())
    optimizer = torch.optim.Adam([variable], lr=CONFIG["quant_domain_z_lr"],
                                 betas=tuple(CONFIG["adam_betas"]), eps=CONFIG["adam_eps"])
    records, cache, hits, cache_check = [], {}, 0, None
    torch.cuda.synchronize(); started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    ell_b, ell_c, q = original_context(bundle, row)

    def differentiate():
        rounded = variable.detach().round()
        ORC2.checked_symbols(rounded)
        ste = rounded + variable - variable.detach()
        residual = bundle.net.synthesize(ste * DELTA, ell_c)
        reconstruction = rgb01(bundle.fixed.g(ell_b + residual, q))
        mse = F.mse_loss(reconstruction, source)
        perceptual = bundle.lp(reconstruction, source, normalize=True).mean()
        image_loss = mse + CONFIG["lambda_p"] * perceptual
        rate = bundle.net.entropy.bits(ste, DELTA) / PIXELS
        objective = image_loss + lam * rate
        gradient = torch.autograd.grad(objective, variable)[0]
        if not torch.isfinite(gradient).all() or not torch.isfinite(objective):
            raise FloatingPointError("Nonfinite optimization objective")
        return gradient.detach(), {"mse_float32": mse.item(), "lpips_float32": perceptual.item(),
                                   "d_float32": image_loss.item(), "estimated_bpp": rate.item(),
                                   "estimated_objective": objective.item()}

    with (folder / "curve.jsonl").open("x") as log:
        for step in range(CONFIG["steps"] + 1):
            optimizer.zero_grad(set_to_none=True)
            key = tensor_hash(variable.detach().round().cpu())
            if key in cache:
                gradient, values = cache[key]; hits += 1
                if cache_check is None:
                    fresh, fresh_values = differentiate()
                    cache_check = float((gradient - fresh).abs().max())
            else:
                gradient, values = differentiate(); cache[key] = gradient, values
            if step == 0:
                assert gradient.norm() > 0
            log.write(json.dumps({"step": step, "lambda": lam, **values,
                                  "gradient_l2": gradient.double().norm().item(),
                                  "hard_state_cache_hit": key in cache and hits > 0}, allow_nan=False) + "\n")
            log.flush()
            if step in CONFIG["snapshots"]:
                symbols = variable.detach().round().cpu().short()
                save_path = folder / f"step_{step:03d}.pt"
                torch.save({"symbols": symbols, "variable": variable.detach().cpu(), "spec": spec,
                            "domain": "z = u/delta", "step": step,
                            "original_anchor": True, **init_record}, save_path)
                row_record = evaluate_symbols(out, split, candidate, lam, spec, symbols, bundle, metrics,
                                              step, time.perf_counter() - started, source,
                                              init_record["chosen_initialization"])
                row_record["checkpoint"] = str(save_path)
                records.append(row_record)
                print("V17OPT", split, candidate, spec["sample"], step, row_record["packet_bytes"], flush=True)
            if step < CONFIG["steps"]:
                variable.grad = gradient.clone(); optimizer.step()
    unique = []
    seen = set()
    for row_record in records:
        if row_record["integer_symbol_sha256"] not in seen:
            unique.append(row_record); seen.add(row_record["integer_symbol_sha256"])
    chosen = min(unique, key=lambda r: (r["selection_objective"], r["packet_bytes"], r["step"]))
    for row_record in records:
        row_record["selected_by_lambda"] = row_record["integer_symbol_sha256"] == chosen["integer_symbol_sha256"]
    csv_dump(folder / "checkpoints.csv", records)
    json_dump(folder / "checks.json", {"network_updates": 0, "code_updates": CONFIG["steps"],
              "free_scalars": CONFIG["code_scalar_count"], "selected_step": chosen["step"],
              "unique_integer_checkpoints": len(unique), "optimization_seconds": time.perf_counter() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "hard_state_cache_hits": hits,
              "cache_equivalence_max_abs": cache_check, **init_record})
    bundle.check(); del variable, optimizer; torch.cuda.empty_cache()


def run_split(out, split):
    bundle = Bundle(True); metrics = Metrics(bundle)
    try:
        all_specs = sorted(specs(split), key=lambda s: (s["video"], int(s["frame"])))
        for candidate, lam in (("O1", CONFIG["O1_lambda"]), ("O2", CONFIG["O2_lambda"])):
            for spec in all_specs:
                optimize_one(out, split, candidate, lam, spec, bundle, metrics)
        rows = []
        for path in sorted((out / "optimizations" / split).glob("*/**/checkpoints.csv")):
            rows.extend(read_csv(path))
        csv_dump(out / f"{split}_optimization_checkpoints.csv", rows, replace=True)
        json_dump(out / f"{split}_optimization_complete.json", {"frames": len(all_specs),
                  "candidate_optimizations": len(all_specs) * 2, "code_updates": len(all_specs) * 2 * CONFIG["steps"],
                  "network_updates": 0}, replace=True)
    finally:
        metrics.pool.shutdown(wait=True)


def run(out):
    setup()
    run_split(out, "Train2")
    run_split(out, "Val6")


if __name__ == "__main__":
    run(HOME / "results_v1")
