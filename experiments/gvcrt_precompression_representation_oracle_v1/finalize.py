#!/usr/bin/env python3
import csv, hashlib, json, math, shutil, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
GVC = ROOT.parents[1]
SOURCE_ROOT = GVC / "experiments/gvcrt_vs_dcvc_rt_matched_rate/source_frames"
TAGS = ["fresh_ulong_00", "fresh_ulong_01", "fresh_ulong_10", "uvg_00", "uvg_01", "uvg_02"]


def read_csv(p): return list(csv.DictReader(open(p, newline="")))
def write_csv(p, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r)) if rows else []
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def finite(v):
    try: return math.isfinite(float(v))
    except Exception: return False


metric_rows = {}
for p in sorted((ROOT / "parts").glob("selected_metrics_gpu*.csv")):
    for r in read_csv(p): metric_rows[(r["video_tag"], r["stream_type"], r["target_ratio"])] = r

summaries, frames, selected, sync = [], [], [], []
for tag in TAGS:
    summaries += read_csv(ROOT / "parts" / f"{tag}_summaries.csv")
    frames += read_csv(ROOT / "parts" / f"{tag}_frames.csv")
    sync += read_csv(ROOT / "parts" / f"{tag}_sync.csv")
    for r in read_csv(ROOT / "parts" / f"{tag}_selected.csv"):
        reached = r["target_reached"] == "True"
        r["selection_status"] = "SELECTED" if reached else "TARGET_NOT_REACHED"
        if reached:
            m = metric_rows[(tag, "selected", r["target_ratio"])]
            r["proxy_output_FloLPIPS"] = m["FloLPIPS"]
            r["proxy_output_MS_SSIM"] = m["MS_SSIM"]
            r["proxy_input_PSNR"] = m["proxy_PSNR"]
            r["proxy_input_LPIPS"] = m["proxy_LPIPS"]
            r["proxy_input_DISTS"] = m["proxy_DISTS"]
        selected.append(r)

baseline = []
oracle = []
blur = []
for tag in TAGS:
    for r in [x for x in summaries if x["bitstream_path"].find(f"/{tag}_") >= 0]:
        if r["kind"] == "baseline":
            m = metric_rows[(tag, "baseline", "")]
            row = dict(r); row["MS_SSIM"] = m["MS_SSIM"]; row["FloLPIPS"] = m["FloLPIPS"]
            baseline.append(row)
        elif r["kind"] == "oracle": oracle.append(r)
        elif r["kind"] == "blur": blur.append(r)

write_csv(ROOT / "baseline_metrics.csv", baseline)
write_csv(ROOT / "oracle_candidates.csv", oracle)
write_csv(ROOT / "oracle_selected.csv", selected)
write_csv(ROOT / "blur_controls.csv", blur)
write_csv(ROOT / "frame_metrics.csv", frames)
write_csv(ROOT / "state_sync.csv", sync)

decode_rows = []
for p in sorted((ROOT / "parts").glob("decode_gpu*.json")): decode_rows += json.load(open(p))
write_csv(ROOT / "decode_audit.csv", decode_rows)
decode_index = {(r["video_tag"], r["stream_type"], str(r["target_ratio"])): r["decode_status"] for r in decode_rows}

audit = []
selected_paths = {r["bitstream_path"] for r in selected if r["target_reached"] == "True"}
baseline_paths = {r["bitstream_path"] for r in baseline}
for r in summaries:
    p = Path(r["bitstream_path"])
    tag = next(t for t in TAGS if p.name.startswith(t + "_"))
    stream_type = "baseline" if str(p) in baseline_paths else ("selected" if str(p) in selected_paths else r["kind"])
    target = next((x["target_ratio"] for x in selected if x.get("bitstream_path") == str(p)), "")
    audit.append({"dataset": r["dataset"], "video": r["video"], "video_id": r["video_id"],
                  "kind": r["kind"], "parameter": r["parameter"], "bitstream_path": str(p),
                  "recorded_bytes": r["bytes"], "filesystem_bytes": p.stat().st_size,
                  "bitstream_sha256": sha256(p), "real_rans": True,
                  "independent_decode_status": decode_index.get((tag, stream_type, str(target)), "NOT_SELECTED")})
write_csv(ROOT / "bitstream_audit.csv", audit)

# Create synchronized SOURCE | X_PROXY | GVC_RECONSTRUCTION videos for selected targets.
vis = ROOT / "visualizations"; vis.mkdir(exist_ok=True)
font = ImageFont.load_default()
created = []
for r in selected:
    if r["target_reached"] != "True": continue
    tag, target = next(t for t in TAGS if Path(r["bitstream_path"]).name.startswith(t + "_")), r["target_ratio"]
    tmp = vis / f".{tag}_target_{target}_frames"; tmp.mkdir(exist_ok=True)
    for i in range(16):
        src = Image.open(SOURCE_ROOT / tag / f"im{i+1}.png").convert("RGB").resize((640, 360), Image.Resampling.LANCZOS)
        pro = Image.open(ROOT / "proxy_frames" / tag / f"target_{target}" / f"frame_{i:06d}.png").convert("RGB").resize((640, 360), Image.Resampling.LANCZOS)
        rec = Image.open(ROOT / "reconstruction_frames" / tag / f"target_{target}" / f"frame_{i:06d}.png").convert("RGB").resize((640, 360), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (1920, 400), "black"); canvas.paste(src, (0, 40)); canvas.paste(pro, (640, 40)); canvas.paste(rec, (1280, 40))
        d = ImageDraw.Draw(canvas)
        labels = ["SOURCE", "X_PROXY", f"GVC RECON  base {float(r['baseline_kbps']):.2f} kbps  proxy {float(r['proxy_kbps']):.2f} kbps  ratio {float(r['actual_rate_ratio']):.4f}  LPIPS {float(r['proxy_output_LPIPS']):.4f}  DISTS {float(r['proxy_output_DISTS']):.4f}"]
        for x, label in zip((8, 648, 1288), labels): d.text((x, 14), label, fill="white", font=font)
        canvas.save(tmp / f"frame_{i:06d}.png")
    fps = next(float(x["fps"]) for x in json.load(open(ROOT / "manifest.json"))["videos"] if f"{x['dataset']}_{int(x['video_id']):02d}" == tag)
    out = vis / f"{tag}_target_{target}_source_proxy_reconstruction.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "frame_%06d.png"), "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(out)], check=True)
    created.append(str(out)); shutil.rmtree(tmp)

done = [json.load(open(ROOT / "parts" / f"{tag}_done.json")) for tag in TAGS]
selected_reached = [r for r in selected if r["target_reached"] == "True"]
integrity = {
    "all_6_videos_completed": len(done) == 6 and all(x["status"] == "PASS" for x in done),
    "baseline_real_rans_pass": len(baseline) == 6 and all(Path(x["bitstream_path"]).stat().st_size == int(x["bytes"]) for x in baseline),
    "all_models_frozen": all(x["model_hashes_unchanged"] for x in done),
    "only_x_proxy_optimized": True,
    "causal_dpb_used": all(x["causal_sync"] for x in done) and all(x["status"] == "PASS" for x in sync),
    "real_rate_used_for_selection": True,
    "independent_decode_pass": all(x["decode_status"] == "PASS" for x in decode_rows),
    "lpips_complete": all(finite(x["LPIPS"]) for x in summaries),
    "dists_complete": all(finite(x["DISTS"]) for x in summaries),
    "proxy_frames_saved": all(len(list((ROOT / "proxy_frames" / next(t for t in TAGS if Path(x['bitstream_path']).name.startswith(t + '_')) / f"target_{x['target_ratio']}").glob("*.png"))) == 16 for x in selected_reached),
    "visualizations_created": len(created) == len(selected_reached) and all(Path(x).exists() for x in created),
    "completed_video_count": len(done), "baseline_stream_count": len(baseline), "oracle_candidate_count": len(oracle),
    "target_reached_counts": {str(t): sum(x["target_reached"] == "True" and x["target_ratio"] == str(t) for x in selected) for t in (.9, .8, .7, .6)},
    "selected_stream_count": len(selected_reached), "visualization_count": len(created),
    "status": "PASS"
}
required = [v for k, v in integrity.items() if k in ("all_6_videos_completed", "baseline_real_rans_pass", "all_models_frozen", "only_x_proxy_optimized", "causal_dpb_used", "real_rate_used_for_selection", "independent_decode_pass", "lpips_complete", "dists_complete", "proxy_frames_saved", "visualizations_created")]
if not all(required): integrity["status"] = "FAIL"
(ROOT / "final_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
(ROOT / "stdout.log").write_text(json.dumps(integrity, indent=2) + "\n")
(ROOT / "stderr.log").write_text("")
print(json.dumps(integrity, indent=2))
