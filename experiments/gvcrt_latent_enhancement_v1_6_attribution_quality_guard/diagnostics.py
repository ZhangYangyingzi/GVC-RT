"""Principal-point videos, source attribution, and temporal diagnostics."""
from bridge import *


def average(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def run(out):
    setup()
    bundle = Bundle(True)
    data = Data(bundle)
    metrics = Metrics(bundle)
    method, alpha = "M4_O_FRAME_GUARD", 0.5
    selected = [r for r in read_csv(out / "Val6_guard_selected_frames.csv")
                if r["method"] == method and r["alpha"] == alpha]
    allocations = read_csv(out / "Val6_guard_allocations.csv")
    videos = [entry["video_id"] for entry in entries("Val6")]
    selected_map = {(r["video"], int(r["frame"])): r for r in selected}
    outputs, zeros, source_frames = {}, {}, {}
    timeline, temporal_summary = [], []
    video_rows = []
    (out / "videos").mkdir(parents=True, exist_ok=True)
    try:
        for entry in entries("Val6"):
            video = entry["video_id"]
            allocation = next(r for r in allocations if r["video"] == video and
                              r["method"] == method and r["alpha"] == alpha)
            decoded = ORC2.decode(allocation["stream"], base_path(video), MODEL, bundle.net.entropy)
            base = base_rows(entry)
            enhanced, zero, source = [], [], []
            with torch.no_grad():
                for frame in range(16):
                    source.append(source_frame(video, frame).cpu())
                    if frame == 0:
                        enhanced_frame = zero_frame = rgb01(base[frame]["x_base"]).cpu()
                    else:
                        spec = next(s for s in specs("Val6") if s["video"] == video and s["frame"] == frame)
                        context = bundle.context(data.get(spec))
                        enhanced_frame, _ = bundle.render(decoded["frames"][frame], context)
                        zero_frame, _ = bundle.render(torch.zeros(CONFIG["code_shape"], dtype=torch.int16), context)
                        enhanced_frame, zero_frame = enhanced_frame.cpu(), zero_frame.cpu()
                    enhanced.append(enhanced_frame); zero.append(zero_frame)
            outputs[video], zeros[video], source_frames[video] = enhanced, zero, source
            output_path = out / "videos" / f"{video}_{method}_a0p5.mkv"
            write_video(output_path, enhanced, entry["metadata"]["avg_frame_rate"])
            video_rows.append({"video": video, "method": method, "alpha": alpha,
                               "path": str(output_path), "sha256": sha(output_path), "frames": 16})

            errors = {
                "ZERO": [reconstruction - target for reconstruction, target in zip(zero, source)],
                method: [reconstruction - target for reconstruction, target in zip(enhanced, source)],
            }
            transition_values = {name: [float((values[t] - values[t - 1]).abs().mean())
                                        for t in range(1, 16)] for name, values in errors.items()}
            for name, values in transition_values.items():
                temporal_summary.append({"video": video, "method": name, "transition": "I_to_P",
                                         "count": 1, "T_error": values[0],
                                         "definition": "mean(abs(e_t-e_(t-1))); no motion compensation"})
                temporal_summary.append({"video": video, "method": name, "transition": "P_to_P",
                                         "count": 14, "T_error": float(np.mean(values[1:])),
                                         "definition": "mean(abs(e_t-e_(t-1))); no motion compensation"})
            previous = "I"
            for frame in range(1, 16):
                candidate = selected_map[(video, frame)]["candidate"]
                timeline.append({"video": video, "transition_to_frame": frame,
                                 "previous_candidate": previous, "candidate": candidate,
                                 "switch": previous != candidate,
                                 "transition_class": "I_to_P" if frame == 1 else "P_to_P",
                                 "zero_T_error": transition_values["ZERO"][frame - 1],
                                 "guard_T_error": transition_values[method][frame - 1],
                                 "guard_minus_zero_T_error": transition_values[method][frame - 1] - transition_values["ZERO"][frame - 1]})
                previous = candidate

        csv_dump(out / "Val6_principal_videos.csv", video_rows)
        csv_dump(out / "Val6_temporal_timeline.csv", timeline)
        csv_dump(out / "Val6_temporal_error.csv", temporal_summary)
        neighborhoods = []
        for row in timeline:
            frame = int(row["transition_to_frame"])
            near_switch = any(other["video"] == row["video"] and other["switch"] and
                              abs(int(other["transition_to_frame"]) - frame) <= 1 for other in timeline)
            neighborhoods.append({**row, "within_one_transition_of_switch": near_switch})
        csv_dump(out / "Val6_switch_neighborhood_temporal.csv", neighborhoods)

        candidates = {video: load_video_candidates("Val6", video, bundle) for video in videos}
        wrong_rows, futures = [], []
        for index, video in enumerate(videos):
            donor = videos[(index + 1) % len(videos)]
            for frame in range(1, 16):
                target = selected_map[(video, frame)]
                name = target["candidate"]
                if name == "Z":
                    continue
                donor_symbols = candidates[donor][frame][name]
                spec = next(s for s in specs("Val6") if s["video"] == video and s["frame"] == frame)
                reconstruction, _ = bundle.render(donor_symbols, bundle.context(data.get(spec)))
                futures.append(metrics.submit(source_frame(video, frame), reconstruction))
                wrong_rows.append({"target_video": video, "donor_video": donor, "frame": frame,
                                   "fixed_cyclic_donor": True, "target_selection_preserved": name,
                                   "target_was_nonzero": True,
                                   "donor_all_zero": bool(torch.count_nonzero(donor_symbols) == 0),
                                   "target_psnr": target["psnr"], "target_lpips": target["lpips"],
                                   "target_d": target["d"]})
        for row, future in zip(wrong_rows, futures):
            result = future.result()
            row.update({f"wrong_source_{key}": value for key, value in result.items()})
            row["wrong_minus_target_psnr"] = result["psnr"] - float(row["target_psnr"])
            row["wrong_minus_target_lpips"] = result["lpips"] - float(row["target_lpips"])
            row["wrong_minus_target_d"] = result["d"] - float(row["target_d"])
        csv_dump(out / "Val6_wrong_source_guard_principal.csv", wrong_rows)
        json_dump(out / "Val6_wrong_source_guard_summary.json", {
            "scope": "diagnostic only; M4_O_FRAME_GUARD alpha=0.5 nonzero selections",
            "donor_rule": "fixed next-video cyclic permutation; never quality-selected",
            "pairs": len(wrong_rows), "donor_all_zero_pairs": sum(r["donor_all_zero"] for r in wrong_rows),
            "mean_wrong_minus_target_psnr": average(wrong_rows, "wrong_minus_target_psnr"),
            "mean_wrong_minus_target_lpips": average(wrong_rows, "wrong_minus_target_lpips"),
            "mean_wrong_minus_target_d": average(wrong_rows, "wrong_minus_target_d")})
        bundle.check()
    finally:
        metrics.pool.shutdown(wait=True)


if __name__ == "__main__":
    run(HOME / "results_v1")
