"""Evidence-only aggregation, budget deviations, matched-precision overlap, and timing."""
import argparse
from collections import defaultdict
from support import *


def cohort(name):
    if name.startswith("val6_"):
        return "Val6"
    if name.startswith("train2_full_"):
        return "Train2_full"
    if name.startswith("train6_Badapt"):
        return "Train6_adaptation_sparse"
    return "Train6_sparse"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results_v1")
    args=parser.parse_args()
    out=HERE/args.output
    allrows=[]
    for path in sorted(out.glob("*/summary.json")):
        for r in json.loads(path.read_text()):
            r["cohort"]=cohort(r["method"])
            allrows.append(r)
    write_csv(out/"all_metrics.csv",allrows)
    summary=[r for r in allrows if r["video"]=="ALL"]
    write_csv(out/"aggregate_metrics.csv",summary)
    budgets=[]
    for r in allrows:
        if r["video"]=="ALL" or r["scope"]!="IP" or r["kind"] not in ["DIRECT","LEARNED"] or r["diagnostic"]:
            continue
        for fraction in CFG["evaluation"]["budget_fractions"]:
            budgets.append({"cohort":r["cohort"],"video":r["video"],"method":r["method"],"kind":r["kind"],"delta":r["delta"],
                            "base_bits":r["base_bits"],"actual_enhancement_bits":r["enhancement_bits"],
                            "budget_fraction":fraction,"budget_bits":r["base_bits"]*fraction,
                            "actual_fraction":r["actual_enhancement_fraction"],
                            "actual_minus_budget_bits":r["enhancement_bits"]-r["base_bits"]*fraction,
                            "within_budget":r["enhancement_bits"]<=r["base_bits"]*fraction,
                            "IP_PSNR_gain_vs_ZERO_D":r["psnr_gain_vs_ZERO_D"],"IP_LPIPS_gain_vs_ZERO_D":r["lpips_gain_vs_ZERO_D"]})
    write_csv(out/"budget_deviations.csv",budgets)
    overlaps=[]
    for r in allrows:
        if r["video"]=="ALL" or r["scope"]!="IP" or r["kind"] not in ["DIRECT","LEARNED"]:
            continue
        reference_cohort=r["cohort"] if r["cohort"]!="Train6_adaptation_sparse" else "Train6_sparse"
        points=sorted([p for p in allrows if p["cohort"]==reference_cohort and p["video"]==r["video"] and p["scope"]=="IP" and p["kind"]=="ORIGINAL_FP32"],key=lambda p:p["total_bpp"])
        item={"cohort":r["cohort"],"video":r["video"],"method":r["method"],"total_bpp":r["total_bpp"],
              "reference":"original GVC-RT actual QP 0-3 streams, matched FP32 P generator / unchanged FP16 I",
              "interpolation":"per-video piecewise linear metric vs log(total bpp), no extrapolation"}
        if len(points)>=2 and points[0]["total_bpp"]<=r["total_bpp"]<=points[-1]["total_bpp"]:
            item["within_sampled_overlap"]=True
            item["reference_min_bpp"]=points[0]["total_bpp"]
            item["reference_max_bpp"]=points[-1]["total_bpp"]
            for metric in ["psnr","ms_ssim","lpips"]:
                value=float(np.interp(np.log(r["total_bpp"]),np.log([p["total_bpp"] for p in points]),[p[metric] for p in points]))
                item[metric+"_original_interpolated"]=value
                item[metric+"_minus_original_interpolated"]=r[metric]-value
        else:
            item["within_sampled_overlap"]=False
        overlaps.append(item)
    write_csv(out/"matched_rate_comparisons.csv",overlaps)
    timings=[]
    for path in sorted(out.glob("*/*_coding.json")):
        data=json.loads(path.read_text())
        method=path.parent.name
        video=path.stem.removesuffix("_coding")
        recv=data["receiver"]
        timings.append({"cohort":cohort(method),"method":method,"video":video,
                        **data["sender"],"receiver_entropy_decode_seconds":recv["entropy_decode_seconds"],
                        "receiver_base_decode_seconds":recv["base_decode_wall_seconds"],
                        "receiver_synthesis_seconds":sum(recv["synthesis_seconds"]),
                        "receiver_peak_allocated_bytes":recv["peak_allocated_bytes"],
                        "optimization_time_excluded_from_this_table":"separately recorded in sender_optimization_timing and historic teacher checks; must not claim realtime from these timings"})
    write_csv(out/"coding_timing.csv",timings)
    historical=[]
    for s in json.loads((out/"teacher_selection.json").read_text()):
        if s["eligible"]:
            historical.append({"video":s["video"],"frame":s["frame"],"historic_optimizer_seconds":s["historic_full_optimizer_run_seconds"],
                               "current_restore_and_verification_seconds":s["teacher_lookup_restore_evaluate_seconds"],"optimizer_reused":True})
    write_csv(out/"historic_teacher_optimizer_timing.csv",historical)
    # Preserve the final V1 base-only model as an extra unambiguous historical control.
    extra=read_csv(V1/"run_v1/base_only_d_val6/metrics.csv")
    write_csv(out/"historical_Val6_final_BaseOnly_D.csv",extra)
    evidence={"stage_A":json.loads((out/"stage_A_gate.json").read_text()) if (out/"stage_A_gate.json").exists() else None,
              "train6_coded":json.loads((out/"train6_coded_gate.json").read_text()) if (out/"train6_coded_gate.json").exists() else None,
              "train2_full":json.loads((out/"train2_full_gate.json").read_text()) if (out/"train2_full_gate.json").exists() else None,
              "validation_shared_training":False,"source_information_and_rate_efficiency_are_separate":True}
    for path in [out/"status_train.json",out/"status_all.json"]:
        if path.exists():
            evidence["execution_status"]=json.loads(path.read_text())
    val=[r for r in summary if r["cohort"]=="Val6" and r["kind"]=="LEARNED" and r["scope"]=="IP"]
    evidence["Val6_actual_points"]=val
    evidence["learned_budget_coverage"]={}
    evidence["learned_useful_budget_coverage"]={}
    for c in ["Train6_sparse","Train2_full","Val6"]:
        evidence["learned_budget_coverage"][c]={str(f):len({b["video"] for b in budgets if b["cohort"]==c and b["kind"]=="LEARNED" and b["budget_fraction"]==f and b["within_budget"]}) for f in CFG["evaluation"]["budget_fractions"]}
        useful={str(f):set() for f in CFG["evaluation"]["budget_fractions"]}
        for b in budgets:
            if b["cohort"]!=c or b["kind"]!="LEARNED" or not b["within_budget"]:
                continue
            scope="TARGET_P" if c=="Train6_sparse" else "P"
            real=next(r for r in allrows if r["method"]==b["method"] and r["video"]==b["video"] and r["scope"]==scope)
            wrong_name=b["method"].replace("train6_CODED_l","train6_CODED_WRONG_l") if c=="Train6_sparse" else b["method"].replace("_CODED_l","_WRONG_l")
            wrong=next((r for r in allrows if r["method"]==wrong_name and r["video"]==b["video"] and r["scope"]==scope),None)
            if wrong and real["psnr_gain_vs_ZERO_D"]>=CFG["training"]["quantized_gate_min_PSNR_gain_db"] and real["lpips_gain_vs_ZERO_D"]<=CFG["training"]["gate_LPIPS_max_increase"] and real["psnr"]-wrong["psnr"]>=CFG["training"]["gate_REAL_minus_WRONG_PSNR_db"] and real["lpips"]-wrong["lpips"]<=CFG["training"]["gate_LPIPS_max_increase"]:
                useful[str(b["budget_fraction"])].add(b["video"])
        evidence["learned_useful_budget_coverage"][c]={k:len(v) for k,v in useful.items()}
    save_json(out/"decision_evidence.json",evidence)
    lines=["# V1.2 measured complete I/P tables","","Rates include all actual headers. Sparse Train6 enhances only 3/15 P frames per sequence.",""]
    for c in ["Train6_sparse","Train2_full","Val6"]:
        lines += [f"## {c}","","| Method | base kbps | enhancement kbps | total kbps | PSNR | MS-SSIM | LPIPS | PSNR gain vs ZERO_D | LPIPS change |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in summary:
            if r["cohort"]==c and r["scope"]=="IP":
                enh=f'{r["enhancement_kbps"]:.3f}' if r["enhancement_kbps"] is not None else "diagnostic"
                total=f'{r["total_kbps"]:.3f}' if r["total_kbps"] is not None else "—"
                lines.append(f'| {r["method"]} | {r["base_kbps"]:.3f} | {enh} | {total} | {r["psnr"]:.6f} | {r["ms_ssim"]:.6f} | {r["lpips"]:.6f} | {r["psnr_gain_vs_ZERO_D"]:+.6f} | {r["lpips_gain_vs_ZERO_D"]:+.6f} |')
        lines.append("")
    with (out/"main_tables.md").open("x") as f:
        f.write("\n".join(lines)+"\n")
    print(json.dumps(evidence,indent=2))


if __name__=="__main__":
    main()
