#!/usr/bin/env python3
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows(name):
    with open(ROOT / name, newline="") as handle:
        return list(csv.DictReader(handle))


selected = json.loads((ROOT / "v3_1_selected_checkpoint.json").read_text())
resume = json.loads((ROOT / "v3_1_resume_audit.json").read_text())
integrity = json.loads((ROOT / "v3_1_final_integrity.json").read_text())
validation = next(r for r in rows("v3_1_convergence_summary.csv") if int(r["step"]) == selected["step"])
final = next(r for r in rows("v3_1_same_qp_summary.csv") if r["qp"] == "all")
lines = [
    f"experiment directory {ROOT}",
    f"resume checkpoint path {resume['resume_checkpoint']}",
    f"resume optimizer state {'PASS' if resume['optimizer_state_restored'] else 'FAIL'}",
    *[f"{step // 1000}k completed {'PASS' if integrity[f'step_{step}_completed'] else 'FAIL'}"
      for step in (30000, 40000, 50000)],
    f"selected checkpoint step {selected['step']}",
    f"validation mean rate change vs Original {validation['mean_rate_change_percent_vs_original']}%",
    f"validation mean LPIPS change vs Original {validation['mean_LPIPS_change_vs_original']}",
    f"validation mean DISTS change vs Original {validation['mean_DISTS_change_vs_original']}",
    f"final-test mean rate change vs Original {final['mean_bitrate_change_percent']}%",
    f"final-test mean LPIPS change vs Original {final['mean_LPIPS_change']}",
    f"final-test mean DISTS change vs Original {final['mean_DISTS_change']}",
    f"final-test triple-nonworse count {final['triple_nonworse_count']} / 32",
    f"compression frozen {'PASS' if integrity['compression_core_frozen'] else 'FAIL'}",
    f"detokenizer frozen {'PASS' if integrity['detokenizer_frozen'] else 'FAIL'}",
    f"independent decode {'PASS' if integrity['independent_decode_pass'] else 'FAIL'}",
    f"v3_1_final_integrity {integrity['status']}",
    f"v3_1_convergence_summary.csv path {ROOT / 'v3_1_convergence_summary.csv'}",
    f"v3_1_checkpoint_selection.csv path {ROOT / 'v3_1_checkpoint_selection.csv'}",
    f"v3_1_same_qp_summary.csv path {ROOT / 'v3_1_same_qp_summary.csv'}",
    f"v3_1_rd_curves path {ROOT / 'v3_1_rd_curves'}",
]
print("\n".join(lines))
