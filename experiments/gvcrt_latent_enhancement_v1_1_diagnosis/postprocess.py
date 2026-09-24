"""Aggregate complete diagnostics, retain full Pareto set, and create matched visualizations."""
import argparse
import csv
import json
from collections import defaultdict
from common import HERE,CFG,save_json,write_csv,np


def read_csv(path):
    with path.open() as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        for key,value in list(row.items()):
            if value in ["True","False"]:
                row[key]=value=="True"
            elif value=="":
                row[key]=None
            else:
                try:
                    row[key]=float(value)
                except ValueError:
                    pass
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results_v1")
    parser.add_argument("--visuals-only",action="store_true",help="Preserve previously completed CSV aggregation")
    args=parser.parse_args()
    out=HERE/args.output
    status=json.loads((out/"status.json").read_text())
    assert status["status"]=="COMPLETE"
    ablation=read_csv(out/"source_ablation.csv")
    oracle=read_csv(out/"oracle_metrics.csv")
    references=read_csv(out/"oracle_references.csv")
    pareto=read_csv(out/"oracle_pareto.csv")
    samples=json.loads((HERE/"fixed_samples.json").read_text())
    ablation_comparisons=[]
    for level in range(3):
        for scope in ["P","IP"]:
            rows=[r for r in ablation if r["level"]==level and r["scope"]==scope and r["video"]!="ALL_VAL6"]
            indexed={(r["video"],r["mode"]):r for r in rows}
            for mode in ["REAL","CONTINUOUS"]:
                target=[r for r in rows if r["mode"]==mode]
                zero=[indexed[(r["video"],"ZERO")] for r in target]
                wrong=[indexed[(r["video"],"WRONG_SOURCE")] for r in target]
                ablation_comparisons.append({"level":level,"scope":scope,"mode":mode,"videos":len(target),
                    "psnr_gt_zero_videos":sum(a["psnr"]>b["psnr"] for a,b in zip(target,zero)),
                    "lpips_lt_zero_videos":sum(a["lpips"]<b["lpips"] for a,b in zip(target,zero)),
                    "joint_psnr_lpips_better_than_zero_videos":sum(a["psnr"]>b["psnr"] and a["lpips"]<b["lpips"] for a,b in zip(target,zero)),
                    "joint_better_than_zero_and_wrong_videos":sum(a["psnr"]>b["psnr"] and a["lpips"]<b["lpips"] and a["psnr"]>c["psnr"] and a["lpips"]<c["lpips"] for a,b,c in zip(target,zero,wrong))})
    if not args.visuals_only:
        write_csv(out/"ablation_consistency.csv",ablation_comparisons)
    oracle_summary=[]
    for initialization in CFG["oracle"]["initializations"]:
        for objective in CFG["oracle"]["objectives"]:
            for step in CFG["oracle"]["snapshots"]:
                rows=[r for r in oracle if r["initialization"]==initialization and r["objective"]==objective and r["step"]==step]
                oracle_summary.append({"initialization":initialization,"objective":objective,"step":step,"samples":len(rows),
                                       **{k:float(np.mean([r[k] for r in rows])) for k in ["psnr","ms_ssim","lpips","delta_latent_l2","delta_latent_max_abs","relative_delta_l2"]}})
    if not args.visuals_only:
        write_csv(out/"oracle_summary.csv",oracle_summary)
    improvements=[]
    for sample in samples:
        sid=sample["sample"]
        ref={r["mode"]:r for r in references if r["sample"]==sid}
        candidates=[r for r in oracle if r["sample"]==sid]
        for r in candidates:
            a={k:r[k] for k in ["sample","initialization","objective","step","psnr","lpips","ms_ssim","relative_delta_l2"]}
            for mode in ["BASE_ONLY_C","REFERENCE_FP32","ORIGINAL_FP16"]:
                b=ref[mode]
                a[f"psnr_minus_{mode}"]=r["psnr"]-b["psnr"]
                a[f"lpips_minus_{mode}"]=r["lpips"]-b["lpips"]
                a[f"dominates_{mode}"]=r["psnr"]>=b["psnr"] and r["lpips"]<=b["lpips"] and (r["psnr"]>b["psnr"] or r["lpips"]<b["lpips"])
            improvements.append(a)
    if not args.visuals_only:
        write_csv(out/"oracle_reference_comparisons.csv",improvements)
        write_csv(out/"oracle_nondominated.csv",[r for r in pareto if r["nondominated_psnr_lpips"]])
    checks=json.loads((out/"oracle_checks.json").read_text())
    evidence={"oracle_samples_with_any_point_dominating_base_only":len({r["sample"] for r in improvements if r["dominates_BASE_ONLY_C"]}),
              "oracle_samples_with_step150_point_dominating_base_only":len({r["sample"] for r in improvements if r["step"]==150 and r["dominates_BASE_ONLY_C"]}),
              "oracle_samples_with_point_dominating_original_fp16":len({r["sample"] for r in improvements if r["dominates_ORIGINAL_FP16"]}),
              "latent_optimization_runs":len(checks["runs"]),"failures":checks["failures"],
              "total_latent_updates":checks["total_latent_optimizer_updates"],"network_updates":0}
    if not args.visuals_only:
        save_json(out/"decision_evidence.json",evidence)
    from plot_cv import curves,pareto as plot_pareto
    import cv2
    plots=out/"plots"
    plots.mkdir()
    for sample in samples:
        sid=sample["sample"]
        rows=[r for r in oracle if r["sample"]==sid]
        curves(plots/f"{sid}_curves.png",rows,sid)
        prs=[r for r in pareto if r["sample"]==sid]
        plot_pareto(plots/f"{sid}_pareto.png",prs,sid)
        folder=out/"oracle"/sid
        paths=[("SOURCE",folder/"source.png"),("ORIGINAL",folder/"ORIGINAL_FP16.png"),
               ("BASE_ONLY_C",folder/"BASE_ONLY_C.png"),("ZERO_START",folder/"ZERO_START_C_DELTA2.png")]
        paths += [(f"{i}/{o} 150",folder/f"{i}_{o}"/"step_150.png") for i in CFG["oracle"]["initializations"] for o in CFG["oracle"]["objectives"]]
        full,crops=[],[]
        x1,y1,x2,y2=sample["crop_xyxy"]
        for label,path in paths:
            im=cv2.imread(str(path))
            if im is None:
                continue
            small=cv2.resize(im,(480,270),interpolation=cv2.INTER_AREA)
            crop=im[y1:y2,x1:x2].copy()
            for panel in [small,crop]:
                cv2.putText(panel,label,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),3)
                cv2.putText(panel,label,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
            full.append(small)
            crops.append(crop)
        cv2.imwrite(str(plots/f"{sid}_full_comparison.png"),np.concatenate([np.concatenate(full[:4],1),np.concatenate(full[4:],1)],0))
        cv2.imwrite(str(plots/f"{sid}_crop_comparison.png"),np.concatenate([np.concatenate(crops[:4],1),np.concatenate(crops[4:],1)],0))
    # Machine-readable main tables, leaving interpretation to report.md.
    text=["# V1.1 measured main tables","","## Table 1: Val6 P frames","",
          "| delta | mode | PSNR | MS-SSIM | LPIPS | PSNR-ZERO | MS-SSIM-ZERO | LPIPS-ZERO |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for scope in ["P","IP"]:
        if scope=="IP":
            text += ["","## Table 1b: Full I/P","","| delta | mode | PSNR | MS-SSIM | LPIPS | PSNR-ZERO | MS-SSIM-ZERO | LPIPS-ZERO |","|---|---|---:|---:|---:|---:|---:|---:|"]
        for level in range(3):
            for mode in CFG["ablation"]["modes"]:
                r=next(r for r in ablation if r["video"]=="ALL_VAL6" and r["scope"]==scope and r["level"]==level and r["mode"]==mode)
                text.append(f'| {r["delta"]} | {mode} | {r["psnr"]:.6f} | {r["ms_ssim"]:.6f} | {r["lpips"]:.6f} | {r["psnr_minus_zero"]:+.6f} | {r["ms_ssim_minus_zero"]:+.6f} | {r["lpips_minus_zero"]:+.6f} |')
    text += ["","## Table 2: All individual step-150 runs","","| sample | initialization | objective | step | PSNR | MS-SSIM | LPIPS | relative L2 | delta max abs |","|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in oracle:
        if r["step"]==150:
            text.append(f'| {r["sample"]} | {r["initialization"]} | {r["objective"]} | 150 | {r["psnr"]:.6f} | {r["ms_ssim"]:.6f} | {r["lpips"]:.6f} | {r["relative_delta_l2"]:.6f} | {r["delta_latent_max_abs"]:.6f} |')
    with (out/"main_tables.md").open("x") as f:
        f.write("\n".join(text)+"\n")
    print(json.dumps(evidence,indent=2))


if __name__=="__main__":
    main()
