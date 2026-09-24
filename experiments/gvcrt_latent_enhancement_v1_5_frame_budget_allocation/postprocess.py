"""Final metrics, wrong-source attribution, temporal-error diagnostics, and videos."""
import collections

from bridge import *
from candidates import load_video_candidates


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def aggregate_metrics(out, split, bundle, metrics):
    selected = read_csv(out / f"{split}_selected_frames.csv")
    allocations = read_csv(out / f"{split}_allocations.csv")
    i_rows = {}
    for entry in entries(split):
        video = entry["video_id"]
        base = base_rows(entry)
        source = source_frame(video, 0)
        i_rows[video] = metrics.submit(source, rgb01(base[0]["x_base"]).cuda()).result()
    summary = []
    keys = ["mse", "psnr", "ms_ssim", "lpips", "d"]
    for allocation in allocations:
        p = [r for r in selected if r["video"] == allocation["video"] and
             r["method"] == allocation["method"] and r["alpha"] == allocation["alpha"]]
        assert len(p) == 15
        zero = [r for r in selected if r["video"] == allocation["video"] and
                r["method"] == allocation["method"] and r["alpha"] == 0]
        row = {**allocation, "scope": "P", "frames": 15,
               "nonzero_enhancement_frames": sum(r["candidate"] != "Z" for r in p)}
        for key in keys:
            row[key] = mean(p, key)
            row[f"{key}_minus_ZERO"] = row[key] - mean(zero, key)
        summary.append(row)
        ip = {**allocation, "scope": "IP", "frames": 16,
              "nonzero_enhancement_frames": row["nonzero_enhancement_frames"]}
        for key in keys:
            ip[key] = (sum(r[key] for r in p) + i_rows[allocation["video"]][key]) / 16
            ip[f"{key}_minus_ZERO"] = (row[f"{key}_minus_ZERO"] * 15) / 16
        summary.append(ip)
    csv_dump(out / f"{split}_summary.csv", summary)
    return summary


def principal_reconstructions(out, bundle, data):
    split, method, alpha = "Val6", "M4_O_FRAME", .5
    selected = [r for r in read_csv(out / "Val6_selected_frames.csv")
                if r["method"] == method and r["alpha"] == alpha]
    videos = [e["video_id"] for e in entries(split)]
    outputs, symbols = {}, {}
    (out / "videos").mkdir(parents=True, exist_ok=True)
    for entry in entries(split):
        video = entry["video_id"]
        stream = next(r["stream"] for r in read_csv(out / "Val6_allocations.csv")
                      if r["video"] == video and r["method"] == method and r["alpha"] == alpha)
        decoded = ORC2.decode(stream, base_path(video), MODEL, bundle.net.entropy)
        base = base_rows(entry); frames = []
        for frame in range(16):
            if frame == 0:
                reconstruction = rgb01(base[frame]["x_base"]).cuda()
            else:
                spec = next(s for s in specs(split) if s["video"] == video and s["frame"] == frame)
                reconstruction, _ = bundle.render(decoded["frames"][frame], bundle.context(data.get(spec)))
                symbols[(video, frame)] = decoded["frames"][frame]
            frames.append(reconstruction.detach().cpu())
        outputs[video] = frames
        write_video(out / "videos" / f"{video}_{method}_a0p5.mkv", frames, entry["metadata"]["avg_frame_rate"])
    return outputs, symbols, selected, videos


def temporal_diagnostics(out, outputs, bundle, data):
    rows = []
    for entry in entries("Val6"):
        video = entry["video_id"]
        source = [source_frame(video, frame).cpu() for frame in range(16)]
        enhanced = outputs[video]
        base = base_rows(entry)
        zero = [rgb01(row["x_base"]).cpu() if frame == 0 else None for frame, row in enumerate(base)]
        with torch.no_grad():
            for frame in range(1, 16):
                spec = next(s for s in specs("Val6") if s["video"] == video and s["frame"] == frame)
                zero[frame], _ = bundle.render(torch.zeros(CONFIG["code_shape"], dtype=torch.int16),
                                                bundle.context(data.get(spec)))
                zero[frame] = zero[frame].cpu()
        for method, reconstructions in (("ZERO", zero), ("M4_O_FRAME_a0p5", enhanced)):
            errors = [reconstruction - target for reconstruction, target in zip(reconstructions, source)]
            transitions = [float((errors[t] - errors[t - 1]).abs().mean()) for t in range(1, 16)]
            rows.append({"video": video, "method": method, "transition": "I_to_P",
                         "count": 1, "T_error": transitions[0],
                         "definition": "mean(abs(e_t-e_(t-1))); no optical-flow compensation"})
            rows.append({"video": video, "method": method, "transition": "P_to_P",
                         "count": 14, "T_error": float(np.mean(transitions[1:])),
                         "definition": "mean(abs(e_t-e_(t-1))); no optical-flow compensation"})
    csv_dump(out / "Val6_temporal_error.csv", rows)


def wrong_source(out, bundle, data, metrics, symbols, selected, videos):
    rows, futures = [], []
    selected_map = {(r["video"], int(r["frame"])): r for r in selected}
    all_candidates = {}
    for video in videos:
        all_candidates[video], _ = load_video_candidates("Val6", video, bundle, out)
    for index, video in enumerate(videos):
        donor = videos[(index + 1) % len(videos)]
        for frame in range(1, 16):
            target = selected_map[(video, frame)]
            name = target["candidate"]
            if name == "Z":
                continue
            donor_symbols = all_candidates[donor][frame][name]
            spec = next(s for s in specs("Val6") if s["video"] == video and s["frame"] == frame)
            reconstruction, _ = bundle.render(donor_symbols, bundle.context(data.get(spec)))
            futures.append(metrics.submit(source_frame(video, frame), reconstruction))
            rows.append({"target_video": video, "donor_video": donor, "frame": frame,
                         "target_selection_preserved": name, "target_was_nonzero": True,
                         "donor_all_zero": bool(torch.count_nonzero(donor_symbols) == 0),
                         "target_psnr": target["psnr"], "target_lpips": target["lpips"],
                         "target_d": target["d"]})
    for row, future in zip(rows, futures):
        result = future.result(); row.update({f"wrong_source_{k}": v for k, v in result.items()})
        row["wrong_minus_target_psnr"] = result["psnr"] - row["target_psnr"]
        row["wrong_minus_target_lpips"] = result["lpips"] - row["target_lpips"]
        row["wrong_minus_target_d"] = result["d"] - row["target_d"]
    csv_dump(out / "Val6_wrong_source_principal.csv", rows)
    json_dump(out / "Val6_wrong_source_summary.json", {
        "scope": "principal nonzero selections only: M4_O_FRAME alpha=0.5",
        "pairs": len(rows), "donor_all_zero_pairs": sum(r["donor_all_zero"] for r in rows),
        "mean_wrong_minus_target_psnr": mean(rows, "wrong_minus_target_psnr"),
        "mean_wrong_minus_target_lpips": mean(rows, "wrong_minus_target_lpips"),
        "mean_wrong_minus_target_d": mean(rows, "wrong_minus_target_d")})


def run(out):
    bundle = Bundle(True); data = Data(bundle); metrics = Metrics(bundle)
    try:
        train = aggregate_metrics(out, "Train2", bundle, metrics)
        val = aggregate_metrics(out, "Val6", bundle, metrics)
        outputs, symbols, selected, videos = principal_reconstructions(out, bundle, data)
        temporal_diagnostics(out, outputs, bundle, data)
        wrong_source(out, bundle, data, metrics, symbols, selected, videos)
        bundle.check()
    finally:
        metrics.pool.shutdown(wait=True)


if __name__ == "__main__":
    setup()
    run(Path("results_v1").resolve())
