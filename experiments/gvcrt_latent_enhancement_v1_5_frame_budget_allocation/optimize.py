"""Frozen-network Val6 O1/O2 code optimization."""
from bridge import *


def snapshot(folder, step, symbols, spec, bundle):
    frames = [None] + [torch.zeros(CONFIG["code_shape"], dtype=torch.int16) for _ in range(15)]
    frames[spec["frame"]] = symbols.cpu().short()
    path = folder / f"step_{step:03d}.orc2"
    coded = ORC2.encode(path, base_path(spec["video"]), MODEL, 1, 3, DELTA, (8, 17, 30), frames,
                        bundle.net.entropy)
    decoded = ORC2.decode(path, base_path(spec["video"]), MODEL, bundle.net.entropy)
    assert torch.equal(decoded["frames"][spec["frame"]], symbols.cpu().short())
    packet_bytes = coded["frames"][spec["frame"]]["actual_bits"] // 8
    return {"diagnostic_container_bytes": coded["file_bits"] // 8, "packet_bytes": packet_bytes,
            "frame_share_bytes": packet_bytes + (ORC2.HEADER.size + 1) / 15,
            "integer_roundtrip_exact": True}


def optimize_one(out, name, lam, spec, bundle, data, metrics):
    folder = out / "optimizations" / name / spec["sample"]
    if (folder / "selected.pt").exists():
        return
    if folder.exists():
        attempt = 1
        moved = folder.with_name(folder.name + f"_interrupted_attempt{attempt}")
        while moved.exists():
            attempt += 1
            moved = folder.with_name(folder.name + f"_interrupted_attempt{attempt}")
        folder.rename(moved)
    folder.mkdir(parents=True)
    item = data.get(spec)
    context = bundle.context(item)
    context_hashes = tensor_hash(context[0]), tensor_hash(context[1])
    source = source_frame(spec["video"], spec["frame"])
    variable = nn.Parameter(torch.zeros(CONFIG["code_shape"], device="cuda"))
    optimizer = torch.optim.Adam([variable], lr=CONFIG["quant_domain_z_lr"],
                                 betas=tuple(CONFIG["adam_betas"]), eps=CONFIG["adam_eps"])
    records, futures, cache, cache_check, hits = [], [], {}, None, 0
    torch.cuda.synchronize(); started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()

    def differentiate():
        rounded = variable.detach().round()
        ORC2.checked_symbols(rounded)
        ste = rounded + variable - variable.detach()
        reconstruction, _ = bundle.render(ste, context)
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
            key = tensor_hash(variable.detach().round())
            if key in cache:
                gradient, values = cache[key]
                hits += 1
                if cache_check is None:
                    fresh, fresh_values = differentiate()
                    assert torch.allclose(gradient, fresh, atol=1e-7, rtol=1e-5)
                    assert abs(values["estimated_objective"] - fresh_values["estimated_objective"]) < 1e-8
                    cache_check = float((gradient - fresh).abs().max())
            else:
                gradient, values = differentiate()
                cache[key] = gradient, values
            if step == 0:
                assert gradient.norm() > 0
            log.write(json.dumps({"step": step, "lambda": lam, **values,
                                  "gradient_l2": gradient.double().norm().item(),
                                  "hard_state_cache_hit": key in cache and hits > 0}, allow_nan=False) + "\n")
            log.flush()
            if step in CONFIG["snapshots"]:
                symbols = variable.detach().round().cpu().short()
                with torch.no_grad():
                    reconstruction, residual = bundle.render(symbols, context)
                    if step == 0:
                        assert torch.count_nonzero(residual) == 0
                row = {"candidate": name, "lambda": lam, "sample": spec["sample"],
                       "video": spec["video"], "frame": spec["frame"], "step": step,
                       "nonzero_symbols": int(torch.count_nonzero(symbols)),
                       "elapsed_seconds": time.perf_counter() - started,
                       **snapshot(folder, step, symbols, spec, bundle)}
                save_pt(folder / f"step_{step:03d}.pt", {"symbols": symbols, "variable": variable.detach().cpu(),
                        "spec": spec, "domain": "z = u/delta", "step": step})
                records.append(row); futures.append(metrics.submit(source, reconstruction))
                print("OPT", name, spec["sample"], step, row["packet_bytes"], flush=True)
            if step < CONFIG["steps"]:
                variable.grad = gradient.clone(); optimizer.step()
    for row, future in zip(records, futures):
        row.update(future.result())
        row["selection_objective"] = row["d"] + lam * (row["frame_share_bytes"] * 8 / PIXELS)
    chosen = min(records, key=lambda r: (r["selection_objective"], r["packet_bytes"], r["step"]))
    for row in records:
        row["selected"] = row["step"] == chosen["step"]
    save_pt(folder / "selected.pt", torch.load(folder / f"step_{chosen['step']:03d}.pt",
                                                 map_location="cpu", weights_only=False))
    csv_dump(folder / "checkpoints.csv", records)
    torch.cuda.synchronize()
    json_dump(folder / "checks.json", {"network_updates": 0, "code_updates": CONFIG["steps"],
              "free_scalars": CONFIG["code_scalar_count"], "selected_step": chosen["step"],
              "optimization_seconds": time.perf_counter() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "hard_state_cache_hits": hits,
              "cache_equivalence_max_abs": cache_check})
    assert (tensor_hash(context[0]), tensor_hash(context[1])) == context_hashes
    bundle.check()
    del variable, optimizer
    torch.cuda.empty_cache()


def run(out):
    bundle, data = Bundle(True), Data(None)
    data.bundle = bundle
    metrics = Metrics(bundle)
    try:
        for name, lam in (("O1", CONFIG["O1_lambda"]), ("O2", CONFIG["O2_lambda"])):
            for spec in specs("Val6"):
                optimize_one(out, name, lam, spec, bundle, data, metrics)
        json_dump(out / "optimization_complete.json", {"Val6_frames": len(specs("Val6")),
                  "candidate_optimizations": 2 * len(specs("Val6")), "network_updates": 0,
                  "code_updates": 2 * len(specs("Val6")) * CONFIG["steps"]})
        csv_dump(out / "optimization_input_timings.csv", data.timings)
        bundle.check()
    finally:
        metrics.pool.shutdown(wait=True)
