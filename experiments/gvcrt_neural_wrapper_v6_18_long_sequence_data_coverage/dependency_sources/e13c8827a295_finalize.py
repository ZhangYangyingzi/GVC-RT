#!/usr/bin/env python3
import json

import numpy as np

from common import ROOT, read_csv, write_csv, write_json


LOWER = ("lpips", "dists", "flolpips", "temporal_delta_l1")


def classify_retention(psnr, perceptual, temporal_regression, distributional_regression):
    regression = temporal_regression or distributional_regression
    if psnr >= 0.8 and perceptual >= 0.8 and not regression:
        return "SUPPORTED"
    if psnr < 0.6 or perceptual < 0.6 or regression:
        return "NOT SUPPORTED"
    return "PARTIAL"


def main():
    final_audit = ROOT / "analysis/final_integrity.json"
    if not final_audit.exists() or json.loads(final_audit.read_text())["status"] != "PASS":
        raise RuntimeError("Final integrity audit must pass before report generation")
    rows = read_csv(ROOT / "analysis/test_summary.csv")
    means = {row["branch"]: {key: float(value) for key, value in row.items()
                              if key not in ("scope", "requested_qp", "branch")}
             for row in rows if row["scope"] == "mean_qp"}
    baseline, proposed = means["m0"], means["m4"]
    gains = {}
    for branch, values in means.items():
        gains[branch] = {"psnr_gain_db": values["psnr"] - baseline["psnr"],
                         "ssim_gain": values["ssim"] - baseline["ssim"],
                         "paired_inception_l2_reduction_fraction":
                             ((baseline["paired_inception_l2"] - values["paired_inception_l2"]) /
                              baseline["paired_inception_l2"]),
                         **{f"{key}_reduction_fraction": (baseline[key] - values[key]) / baseline[key]
                            for key in LOWER},
                         "fid_change": values["fid"] - baseline["fid"],
                         "kid_change": values["kid"] - baseline["kid"]}
    retention_rows = []
    for branch in ("m1", "m2", "m3", "m5", "m6"):
        row = {"branch": branch,
               "psnr": gains[branch]["psnr_gain_db"] / gains["m4"]["psnr_gain_db"]
                       if gains["m4"]["psnr_gain_db"] != 0 else None}
        for key in LOWER:
            denominator = gains["m4"][f"{key}_reduction_fraction"]
            row[key] = gains[branch][f"{key}_reduction_fraction"] / denominator if denominator != 0 else None
        row.update({"fid_absolute_change": gains[branch]["fid_change"],
                    "fid_relative_change": gains[branch]["fid_change"] / baseline["fid"],
                    "kid_absolute_change": gains[branch]["kid_change"],
                    "kid_relative_change": gains[branch]["kid_change"] / abs(baseline["kid"]),
                    "average_perceptual_retention": float(np.mean([row[key] for key in
                                                                     ("lpips", "dists", "flolpips")]))})
        retention_rows.append(row)
    write_csv(ROOT / "analysis/gain_retention.csv", retention_rows)
    retention = {row["branch"]: row for row in retention_rows}
    temporal_regression = lambda branch: gains[branch]["temporal_delta_l1_reduction_fraction"] < -0.02
    distributional_regression = lambda branch: gains[branch]["fid_change"] > 0 or gains[branch]["kid_change"] > 0
    training = classify_retention(retention["m1"]["psnr"], retention["m1"]["average_perceptual_retention"],
                                  temporal_regression("m1"), distributional_regression("m1"))
    capacity = classify_retention(retention["m2"]["psnr"], retention["m2"]["average_perceptual_retention"],
                                  temporal_regression("m2"), distributional_regression("m2"))
    additional = {}
    for control in ("m3", "m5"):
        additional[control] = {"psnr_db": gains["m4"]["psnr_gain_db"] - gains[control]["psnr_gain_db"],
                               **{key: 100 * (gains["m4"][f"{key}_reduction_fraction"] -
                                              gains[control][f"{key}_reduction_fraction"])
                                  for key in LOWER},
                               "m4_fid_no_worse": means["m4"]["fid"] <= means[control]["fid"],
                               "m4_kid_no_worse": means["m4"]["kid"] <= means[control]["kid"]}
    def material(control):
        metric_count = sum(additional[control][key] >= 2 for key in LOWER)
        return ((additional[control]["psnr_db"] >= 0.30 or metric_count >= 2) and
                additional[control]["m4_fid_no_worse"] and additional[control]["m4_kid_no_worse"])
    temporal = "SUPPORTED" if material("m3") else "NOT SUPPORTED"
    complexity = json.loads((ROOT / "analysis/complexity.json").read_text())
    m4_latency = complexity["methods"]["m4"]["mean_candidate_seconds_per_frame"]
    m5_latency = complexity["methods"]["m5"]["mean_candidate_seconds_per_frame"]
    position = "SUPPORTED" if material("m5") and m4_latency <= m5_latency else "NOT SUPPORTED"
    thirds = read_csv(ROOT / "analysis/sequence_thirds.csv")
    no_collapse = True
    for third in ("early", "middle", "late"):
        base = next(row for row in thirds if row["third"] == third and row["branch"] == "m0")
        m4 = next(row for row in thirds if row["third"] == third and row["branch"] == "m4")
        no_collapse &= (float(m4["psnr"]) > float(base["psnr"]) and
                        all(float(m4[key]) <= float(base[key]) for key in LOWER))
    payload = {"gains_vs_m0": gains, "retention_vs_m4": retention,
               "m4_additional_over_controls": additional,
               "interpretations": {"training_insufficiency": training,
                                   "extra_capacity": capacity,
                                   "temporal_evidence": temporal,
                                   "interface_position": position},
               "no_m4_long_sequence_collapse": bool(no_collapse),
               "same_stream_zero_bit": True,
               "thresholds": {"supported_retention": 0.8, "insufficient_retention": 0.6,
                              "material_psnr_db": 0.30, "material_percentage_points": 2.0}}
    write_json(ROOT / "analysis/causal_interpretation.json", payload)
    write_reports(means, gains, payload, complexity)
    print(json.dumps(payload, indent=2))
    print_terminal_summary(means, gains, payload, complexity)


def percent(value):
    return f"{100 * value:.2f}%"


def write_reports(means, gains, interpretation, complexity):
    names = {"m0": "Official baseline", "m1": "Bridge fine-tune",
             "m2": "Bridge + matched adapter", "m3": "Current-only codeword",
             "m4": "Temporal codeword (ours)", "m5": "Temporal RGB postprocessor",
             "m6": "Historical V8 reference"}
    frozen = json.loads((ROOT / "checkpoints/final/SELECTIONS_FROZEN.json").read_text())
    lines = ["# Stage V9 Report", "", "## Protocol", "",
             "All results use the same real entropy-coded streams and untouched official decoder trajectory. "
             "M1-M5 were trained for 80 updates under the same V9 plan with three learning rates; checkpoint "
             "selection used validation only. M6 is an external historical reference. Additional transmitted bits: 0.",
             "", "## Frozen Validation Selections", "",
             "| Method | LR | Update | Eligible | SHA256 |", "|---|---:|---:|---|---|"]
    for branch, selected in frozen["selected"].items():
        lines.append(f"| {branch.upper()} | {selected['lr']:.0e} | {selected['update']} | "
                     f"{selected['selected_from_eligible_set']} | `{selected['sha256']}` |")
    lines.extend(["", "## Mean-QP Fresh-Test Results", "",
             "| Method | PSNR | SSIM | LPIPS | DISTS | FloLPIPS | TemporalDeltaL1 | Paired Inception L2 | FID | KID |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for branch in names:
        row = means[branch]
        lines.append(f"| {branch.upper()} {names[branch]} | {row['psnr']:.4f} | {row['ssim']:.6f} | "
                     f"{row['lpips']:.6f} | {row['dists']:.6f} | {row['flolpips']:.6f} | "
                     f"{row['temporal_delta_l1']:.6f} | {row['paired_inception_l2']:.6f} | "
                     f"{row['fid']:.6f} | {row['kid']:.8f} |")
    lines.extend(["", "## Gains Versus M0", "",
                  "| Method | PSNR gain | LPIPS reduction | DISTS reduction | FloLPIPS reduction | Temporal reduction | FID change | KID change |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for branch in ("m1", "m2", "m3", "m4", "m5", "m6"):
        row = gains[branch]
        lines.append(f"| {branch.upper()} | {row['psnr_gain_db']:+.4f} dB | "
                     f"{percent(row['lpips_reduction_fraction'])} | {percent(row['dists_reduction_fraction'])} | "
                     f"{percent(row['flolpips_reduction_fraction'])} | "
                     f"{percent(row['temporal_delta_l1_reduction_fraction'])} | "
                     f"{row['fid_change']:+.6f} | {row['kid_change']:+.8f} |")
    qp_rows = [row for row in read_csv(ROOT / "analysis/test_summary.csv")
               if row["scope"] == "qp" and row["branch"] == "m0"]
    lines.extend(["", "## Communication", "",
                  "All branches reuse these exact official streams; additional transmitted bits are zero.", "",
                  "| Requested QP | Mean bits/sequence | Visible BPP |",
                  "|---:|---:|---:|"])
    for row in qp_rows:
        lines.append(f"| {row['requested_qp']} | {float(row['actual_bits']):.2f} | "
                     f"{float(row['visible_bpp']):.8f} |")
    labels = interpretation["interpretations"]
    lines.extend(["", "## Causal Questions", "",
                  f"- RQ1, better original-bridge training: **{labels['training_insufficiency']}**.",
                  f"- RQ2, matched extra bridge capacity: **{labels['extra_capacity']}**.",
                  f"- RQ3, temporal evidence: **{labels['temporal_evidence']}**.",
                  f"- RQ4, pre-generator interface position: **{labels['interface_position']}**.",
                  f"- RQ5: the supported explanation is the combination indicated by those four preregistered outcomes.",
                  f"- RQ6: M4 remains same-stream and zero-bit: **{interpretation['same_stream_zero_bit']}**.",
                  f"- RQ7: M4 fresh-test changes are listed above, including FID and KID without suppressing regressions.",
                  f"- RQ8: the strongest defensible claim is stated in `FINAL_DECISION.md`.",
                  "", "## Robustness And Complexity", "",
                  f"M4 avoids early/middle/late collapse: **{interpretation['no_m4_long_sequence_collapse']}**. "
                  "Per-QP bootstrap intervals are in `analysis/fid_kid_per_qp.json`; segment results are in "
                  "`analysis/sequence_thirds.csv`.", "",
                  "| Method | Trainable params | Stored params | Added-module MACs | Mean seconds/frame | Peak incremental memory |",
                  "|---|---:|---:|---:|---:|---:|"])
    for branch in names:
        row = complexity["methods"][branch]
        latency = row.get("mean_candidate_seconds_per_frame")
        memory = row.get("peak_incremental_memory_bytes")
        lines.append(f"| {branch.upper()} | {row['trainable_parameters']:,} | {row['stored_parameters']:,} | "
                     f"{row.get('added_structural_module_macs', 0):,} | "
                     f"{latency:.6f} | {memory:,} |")
    lines.extend(["", "MACs cover only each added structural module, not shared generator execution or the "
                  "copied M1 bridge. Full branch cost is represented by measured latency and peak incremental memory.",
                  "", "## Integrity", "",
                  "Official checkpoint hashes, frozen selection hashes, stream hashes, actual-QP trajectories, "
                  "zero-bit status, DPB isolation, and artifact dimensions passed the recorded integrity audits. "
                  "The four preregistered visual comparisons are in `outputs/visual_comparisons/`."])
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")

    supported = [key.replace("_", " ") for key, value in labels.items() if value == "SUPPORTED"]
    partial = [key.replace("_", " ") for key, value in labels.items() if value == "PARTIAL"]
    claim = ("The matched V9 study supports " + ", ".join(supported) + "." if supported else
             "The matched V9 study does not support a single dominant causal explanation.")
    if partial:
        claim += " Evidence is partial for " + ", ".join(partial) + "."
    decision = ["# Stage V9 Final Decision", "", claim, "",
                "This conclusion is limited to the fresh V9 split, the fixed model family, and the "
                "predeclared 80-update selection budget. M6 is not part of the matched causal comparison.", "",
                "## Strongest Defensible Claim", "", claim, "",
                "## Main Remaining Threat To Novelty", "",
                "The study isolates the tested bridge, capacity, temporal, and RGB controls, but does not establish "
                "that the same causal ordering holds under larger training budgets or alternative feature-space adapters.",
                "", "## Recommended V10", "",
                "Repeat the strongest unresolved comparison with multiple training seeds and a matched temporal "
                "feature-space adapter, while retaining the same-stream and frozen-DPB protocol."]
    (ROOT / "FINAL_DECISION.md").write_text("\n".join(decision) + "\n")


def print_terminal_summary(means, gains, interpretation, complexity):
    frozen = json.loads((ROOT / "checkpoints/final/SELECTIONS_FROZEN.json").read_text())
    retention = interpretation["retention_vs_m4"]
    labels = interpretation["interpretations"]
    audit = json.loads((ROOT / "analysis/interface_audit.json").read_text())
    print("=" * 60)
    print("STAGE V9: INTERFACE CAUSAL CONTROL STUDY")
    print("=" * 60)
    print(f"REPOSITORY: {ROOT.parents[1]}")
    print(f"EXPERIMENT: {ROOT}")
    print("GPU: physical 4,5,6,7 only")
    print("SPLIT: TRAIN 16 / VALIDATION 8 / FRESH TEST 12")
    print("Frames/video: 96")
    print("QPs: 0,1,2,3")
    print("OFFICIAL BRIDGE: recon_generation_net.mlp.0-3")
    print(f"Original parameter count: {audit['official_bridge_parameters']:,}")
    print("Does it affect official DPB?: yes in-path; candidates are isolated side-cars")
    print("TRAINING: 80 updates/method; LR 1e-4,3e-4,1e-3; latent 16x16 / RGB 256x256")
    print("=" * 60)
    print("FINAL TEST")
    print("=" * 60)
    for branch in ("m0", "m1", "m2", "m3", "m4", "m5", "m6"):
        row, gain = means[branch], gains[branch]
        params = complexity["methods"][branch]["trainable_parameters"]
        print(f"{branch.upper()}: params={params:,} PSNR={row['psnr']:.4f} "
              f"gain={gain['psnr_gain_db']:+.4f}dB LPIPS_reduction={percent(gain['lpips_reduction_fraction'])} "
              f"DISTS_reduction={percent(gain['dists_reduction_fraction'])} "
              f"FloLPIPS_reduction={percent(gain['flolpips_reduction_fraction'])} "
              f"Temporal_reduction={percent(gain['temporal_delta_l1_reduction_fraction'])} "
              f"FID={row['fid']:.6f} KID={row['kid']:.8f}")
        if branch in retention:
            print(f"  Retention vs M4: PSNR={percent(retention[branch]['psnr'])}, "
                  f"perceptual={percent(retention[branch]['average_perceptual_retention'])}")
    print("=" * 60)
    print("CAUSAL QUESTIONS")
    print("=" * 60)
    print(f"TRAINING INSUFFICIENCY: {labels['training_insufficiency']}")
    print(f"EXTRA CAPACITY: {labels['extra_capacity']}")
    print(f"TEMPORAL EVIDENCE: {labels['temporal_evidence']}")
    print(f"INTERFACE POSITION: {labels['interface_position']}")
    print("COMMUNICATION: bitstream identical; extra transmitted bits=0")
    selections = ", ".join(
        f"{key}@{value['update']}" for key, value in frozen["selected"].items())
    print(f"FROZEN CHECKPOINTS: {selections}")


if __name__ == "__main__":
    main()
