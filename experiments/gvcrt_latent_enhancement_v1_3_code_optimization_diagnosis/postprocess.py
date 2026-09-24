"""Aggregate measured code optimization, partitioned quality, actual budgets and matched-rate comparisons."""
import argparse
from bridge import *


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results_v1")
    args=p.parse_args()
    out=HOME/args.output
    checkpoints=[]
    timings=[]
    for path in sorted((out/"optimizations").glob("*/*/checkpoints.csv")):
        stage=path.parent.parent.name
        if not stage.startswith(("A6_","Aext_","B6q_","Bfull_")):
            continue  # explicitly excluded invalid dispatch attempt
        checkpoints.extend(read_csv(path))
        check=json.loads((path.parent/"checks.json").read_text())
        timings.append({"stage":stage,"sample":path.parent.name,**{k:v for k,v in check.items() if not isinstance(v,dict)}})
    write_csv(out/"optimization_checkpoints.csv",checkpoints)
    write_csv(out/"optimization_timings.csv",timings)
    write_csv(out/"selected_and_step150.csv",[r for r in checkpoints if r["selected"] or r["step"]==150])
    pareto=[]
    for sid in sorted({r["sample"] for r in checkpoints}):
        for quantized in [False,True]:
            rows=[r for r in checkpoints if r["sample"]==sid and r["quantized"]==quantized]
            for r in rows:
                dominated=any(o["psnr"]>=r["psnr"] and o["lpips"]<=r["lpips"] and
                              (not quantized or o["actual_frame_share_bits"]<=r["actual_frame_share_bits"]) and
                              (o["psnr"]>r["psnr"] or o["lpips"]<r["lpips"] or (quantized and o["actual_frame_share_bits"]<r["actual_frame_share_bits"])) for o in rows)
                if not dominated:
                    pareto.append(r)
    write_csv(out/"nondominated_checkpoints.csv",pareto)
    frame_rows=[]
    sequences=[]
    groups=[]
    coding=[]
    for path in sorted((out/"sequences").glob("*/summary.csv")):
        sequences.extend(read_csv(path))
        frame_rows.extend(read_csv(path.parent/"frames.csv"))
        groups.extend(read_csv(path.parent/"groups.csv"))
        for check_path in path.parent.glob("*_checks.json"):
            check=json.loads(check_path.read_text())
            coding.append({"method":path.parent.name,"video":check_path.stem.removesuffix("_checks"),
                           "entropy_encode_seconds":check["entropy_encode_seconds"],
                           "base_decode_seconds":check["receiver"]["base_decode_seconds"],
                           "entropy_decode_seconds":check["receiver"]["entropy_decode_seconds"],
                           "synthesis_seconds":sum(check["receiver"]["synthesis_seconds"]),
                           "independent_RGB_exact":check["independent_RGB_exact"],
                           "no_enhancement_file_sent":check["no_enhancement_file_sent"]})
    write_csv(out/"frame_metrics.csv",frame_rows)
    write_csv(out/"sequence_metrics.csv",sequences)
    write_csv(out/"group_metrics.csv",groups)
    write_csv(out/"coding_timings.csv",coding)
    aggregate=[]
    for method in sorted({r["method"] for r in sequences}):
        for scope in ["P","IP"]:
            rows=[r for r in sequences if r["method"]==method and r["scope"]==scope]
            if not rows:
                continue
            duration=sum(r["duration_seconds"] for r in rows)
            base=sum(r["base_bits"] for r in rows)
            enh=sum(r["enhancement_bits"] for r in rows) if rows[0]["enhancement_bits"] is not None else None
            frames=sum(r["frames"] for r in rows)
            item={"method":method,"split":rows[0]["split"],"scope":scope,"frames":frames,"lambda":rows[0]["lambda"],
                  "base_bits":base,"enhancement_bits":enh,"base_kbps":base/duration/1000,
                  "enhancement_kbps":enh/duration/1000 if enh is not None else None,
                  "total_kbps":(base+enh)/duration/1000 if enh is not None else None,
                  "enhancement_fraction":enh/base if enh is not None else None,"diagnostic":enh is None}
            for key in ["psnr","ms_ssim","lpips","L_image"]+[f"{m}_minus_{ref}" for ref in ["ZERO","ENCODER_CONTINUOUS","ENCODER_QUANTIZED","ORIGINAL_QP0"] for m in ["psnr","ms_ssim","lpips"]]:
                item[key]=sum(r[key]*r["frames"] for r in rows)/frames
            aggregate.append(item)
    write_csv(out/"aggregate_metrics.csv",aggregate)
    overlaps=[]
    for r in sequences:
        if r["scope"]!="IP" or r["enhancement_bits"] is None or r["zero_no_stream_fallback"]:
            continue
        prefix="train2_full" if r["split"]=="Train2" else "val6"
        points=[]
        for qp in range(4):
            path=OLD/"results_v1"/f"{prefix}_ORIGINAL_FP32_q{qp}/summary.json"
            old=next(s for s in json.loads(path.read_text()) if s["video"]==r["video"] and s["scope"]=="IP")
            points.append(old)
        points.sort(key=lambda x:x["total_bpp"])
        item={"method":r["method"],"video":r["video"],"total_bpp":r["total_bpp"],
              "within_overlap":points[0]["total_bpp"]<=r["total_bpp"]<=points[-1]["total_bpp"],
              "reference":"existing original QP0-3 actual streams, matched FP32 P generation / unchanged FP16 I", 
              "interpolation":"piecewise linear metric vs log(total bpp), per video, no extrapolation"}
        if item["within_overlap"]:
            for m in ["psnr","ms_ssim","lpips"]:
                original=float(np.interp(np.log(r["total_bpp"]),np.log([p["total_bpp"] for p in points]),[p[m] for p in points]))
                item[f"{m}_original_interpolated"]=original
                item[f"{m}_minus_original_interpolated"]=r[m]-original
        overlaps.append(item)
    write_csv(out/"original_matched_rate.csv",overlaps)
    # Track teacher/E costs separately. They are evaluation-only dependencies for B ZERO_START.
    dependencies=[]
    for spec in json.loads((out/"samples.json").read_text()):
        cache=out/"input_cache"/f'{spec["sample"]}.pt'
        if cache.exists():
            e=torch.load(cache,map_location="cpu",weights_only=False)
            t,_=load_sample(spec)
            s=t["selection"]
            cost=s.get("full_latent_optimization_seconds",s.get("historic_full_optimizer_run_seconds"))
            dependencies.append({"sample":spec["sample"],"group":spec["group"],"teacher_optimizer_seconds_if_required":cost,
                                 "encoder_forward_seconds":e["encoder_forward_seconds"],
                                 "teacher_and_E_required_for_A_ENCODER_START":True,
                                 "teacher_and_E_required_for_B_ZERO_START":False})
    write_csv(out/"teacher_encoder_dependencies.csv",dependencies)
    chain=[]
    for spec in json.loads((out/"samples.json").read_text()):
        if not spec["video"].startswith(("30884276","66d34721")):
            continue
        path=out/"references"/f'{spec["sample"]}.json'
        if not path.exists():
            continue
        refs=json.loads(path.read_text())
        for kind in ["ZERO","TEACHER","ENCODER_CONTINUOUS","ENCODER_QUANTIZED"]:
            chain.append({"sample":spec["sample"],"video":spec["video"],"frame":spec["frame"],"method":kind,**refs[kind],"psnr_minus_ZERO":refs[kind]["psnr"]-refs["ZERO"]["psnr"]})
        chain += [r for r in frame_rows if r["video"]==spec["video"] and r["frame"]==spec["frame"]]
    write_csv(out/"tea_bridge_chain.csv",chain)
    if aggregate:
        lines=["# V1.3 measured complete-sequence points","","| Method | Split | Scope | base kbps | enh kbps | total kbps | enh/base | PSNR | MS-SSIM | LPIPS | PSNR-ZERO | PSNR-current quantized encoder |","|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in aggregate:
            if r["scope"]!="IP":
                continue
            def f(key):
                return "diagnostic" if r[key] is None else f'{r[key]:.6f}'
            lines.append(f'| {r["method"]} | {r["split"]} | IP | {f("base_kbps")} | {f("enhancement_kbps")} | {f("total_kbps")} | {f("enhancement_fraction")} | {f("psnr")} | {f("ms_ssim")} | {f("lpips")} | {f("psnr_minus_ZERO")} | {f("psnr_minus_ENCODER_QUANTIZED")} |')
        (out/"main_tables.md").write_text("\n".join(lines)+"\n")
    facts={"valid_code_runs":len(timings),"valid_code_updates":sum(t["code_updates"] for t in timings),
           "network_updates":0,"continuous_runs":sum(not t["stage"].startswith("B") for t in timings),
           "quantized_runs":sum(t["stage"].startswith("B") for t in timings),
           "invalid_domain_updates_separately_retained":900 if (out/"invalid_attempts.json").exists() else 0}
    save_json(out/"execution_totals.json",facts)
    print(json.dumps(facts,indent=2))


if __name__=="__main__":
    main()
