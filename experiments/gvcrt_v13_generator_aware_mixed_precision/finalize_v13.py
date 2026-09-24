#!/usr/bin/env python3
import csv
import hashlib
import json
import platform
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from run_v13 import MAN, CFG, read_csv, write_csv


def combine(suffix, output):
    rows = []
    for path in sorted((ROOT / "parts").glob(f"*_{suffix}.csv")): rows.extend(read_csv(path))
    write_csv(ROOT / output, rows); return rows


def main():
    expected = [(v["dataset"], int(v["video_id"]), q) for v in MAN["videos"] for q in CFG["requested_qps"]]
    done = []
    for dataset, video_id, qp in expected:
        path = ROOT / "parts" / f"{dataset}_{video_id:02d}_qp{qp}_done.json"
        if path.exists() and json.loads(path.read_text()).get("status") == "PASS": done.append(path)
    baseline = combine("baseline_cells", "baseline_cells.csv")
    combine("block_probe", "block_probe.csv"); combine("frame_candidate_maps", "frame_candidate_maps.csv")
    combine("beam_states", "beam_states.csv"); combine("trajectory_steps", "trajectory_steps.csv")
    combine("final_trajectories", "final_trajectories.csv"); final = combine("final_streams", "final_streams.csv")
    uniform = combine("uniform_controls", "uniform_controls.csv")
    frame_metrics = combine("frame_metrics", "frame_metrics.csv")
    sequence_metrics = combine("sequence_metrics", "sequence_metrics.csv")
    audits = combine("bitstream_audit", "bitstream_audit.csv"); coverage = combine("search_coverage", "search_coverage.csv")
    if not audits:
        write_csv(ROOT / "bitstream_audit.csv", [], [
            "dataset", "video", "video_id", "qp", "target_rate_ratio", "bitstream_path",
            "bitstream_sha256", "frames", "decoded_quantized_symbols_match",
            "decoded_precision_map_match", "decoded_latent_match", "decoder_causal_state_match",
            "final_RGB_match", "total_consumed_bytes", "file_bytes", "decode_audit_pass"])
    multi = []
    for path in sorted((ROOT / "parts").glob("*_multi_qp.csv")): multi.extend(read_csv(path))
    # QP1/QP3 are the freshly rerun identity baselines from this experiment.
    for row in baseline:
        multi.append({"dataset": row["dataset"], "video": row["video"], "video_id": row["video_id"], "QP": row["qp"],
                      "actual_total_bytes": row["baseline_bytes"], "actual_bits": row["baseline_bits"], "kbps": row["kbps"],
                      "bpp": row["bpp"], "PSNR": row["PSNR"], "MS_SSIM": row["MS_SSIM"], "LPIPS": row["LPIPS"],
                      "DISTS": row["DISTS"], "bitstream_path": row["bitstream_path"],
                      "bitstream_sha256": row["bitstream_sha256"], "decode_pass": True})
    multi.sort(key=lambda r: (r["dataset"], int(r["video_id"]), int(r["QP"])))
    write_csv(ROOT / "multi_qp_curve.csv", multi)
    logs = []
    for path in sorted((ROOT / "logs").glob("*.log")):
        logs.append(f"===== {path.name} =====\n{path.read_text(errors='replace')}")
    (ROOT / "run.log").write_text("\n".join(logs))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True).stdout.strip()
    checkpoints = []
    for path in sorted((REPO / "checkpoints").glob("*.pt")):
        h = hashlib.sha256(path.read_bytes()).hexdigest(); checkpoints.append({"path": str(path), "sha256": h})
    runtime = {"python": sys.version, "platform": platform.platform(), "git_commit": commit,
               "manifest_sha256": hashlib.sha256((ROOT / "manifest.json").read_bytes()).hexdigest(),
               "random_seed": CFG["random_seed"], "checkpoints": checkpoints}
    (ROOT / "config.runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    identity_count = sum(str(r.get("identity_gate_pass")).lower() == "true" for r in baseline)
    v124_hash_matches = sum(r.get("bitstream_sha256") == r.get("v12_4_sha256") for r in baseline)
    model_hash_matches = sum(r.get("model_hash_before") == r.get("model_hash_after") for r in baseline)
    uniform_decode_pass = sum(str(r.get("decode_audit_pass")).lower() == "true" for r in uniform)
    multi_qp_decode_pass = sum(str(r.get("decode_pass")).lower() == "true" for r in multi)
    target_not_reached = sum(r.get("selection_type") == "TARGET_NOT_REACHED" for r in final)
    selected_mixed = sum(r.get("selection_type") == "MIXED_PRECISION" for r in final)
    selected_decode_pass = sum(str(r.get("decode_audit_pass")).lower() == "true" for r in audits)
    integrity = {"expected_cells": 12, "completed_cells": len(done), "identity_gate_pass": identity_count,
                 "v12_4_baseline_sha256_match": v124_hash_matches, "model_parameter_hash_unchanged": model_hash_matches,
                 "target_searches_complete": len(final), "target_not_reached": target_not_reached,
                 "selected_mixed_streams": selected_mixed, "selected_mixed_decode_audit_pass": selected_decode_pass,
                 "selected_mixed_decode_audit_total": len(audits), "uniform_controls": len(uniform),
                 "uniform_decode_pass": uniform_decode_pass, "multi_qp_rows": len(multi),
                 "multi_qp_decode_pass": multi_qp_decode_pass,
                 "decode_failures": sum(int(r.get("decode_failures", 0)) for r in coverage),
                 "numerical_failures": sum(int(r.get("numerical_failures", 0)) for r in coverage),
                 "status": "PASS" if len(done) == 12 and identity_count == 12 and v124_hash_matches == 12 and model_hash_matches == 12 else "INCOMPLETE"}
    (ROOT / "final_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    artifacts = ["baseline_cells.csv", "block_probe.csv", "frame_candidate_maps.csv", "beam_states.csv", "trajectory_steps.csv",
                 "final_trajectories.csv", "final_streams.csv", "uniform_controls.csv", "multi_qp_curve.csv", "frame_metrics.csv",
                 "sequence_metrics.csv", "bitstream_audit.csv", "search_coverage.csv", "run.log"]
    artifact_manifest = []
    for name in artifacts + ["config.json", "manifest.json", "implementation_audit.txt", "final_integrity.json", "config.runtime.json"]:
        path = ROOT / name
        if path.exists(): artifact_manifest.append({"path": name, "bytes": path.stat().st_size,
                                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (ROOT / "artifact_manifest.json").write_text(json.dumps(artifact_manifest, indent=2) + "\n")
    print(str(ROOT))
    print(f"12 cells complete: {len(done) == 12}")
    print(f"identity gate pass: {integrity['identity_gate_pass']}/12")
    print(f"target searches complete: {integrity['target_searches_complete']}/36")
    print(f"selected mixed decode audit pass: {selected_decode_pass}/{len(audits)}")
    print("artifacts: " + ", ".join(artifacts) + ", bitstreams/")


if __name__ == "__main__": main()
