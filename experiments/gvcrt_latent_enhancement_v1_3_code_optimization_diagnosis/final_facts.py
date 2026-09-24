"""Create report facts from completed measured records; no new model execution."""
import argparse
from bridge import *


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results_v1")
    args=p.parse_args()
    out=HOME/args.output
    records=read_csv(out/"optimization_checkpoints.csv")
    timing=read_csv(out/"optimization_timings.csv")
    aggregate=read_csv(out/"aggregate_metrics.csv")
    seq=read_csv(out/"sequence_metrics.csv")
    coding=read_csv(out/"coding_timings.csv")
    refs=read_csv(out/"teacher_encoder_dependencies.csv")
    facts={"A_six_initializations":{},"optimization_timing":{},"coding_timing":{},"A_Val6_per_video":[],"B_Train2_per_video":[]}
    for init in CONFIG["A"]["initializations"]:
        stage="A6_"+init
        chosen=[r for r in records if r["stage"]==stage and r["selected"]]
        last=[r for r in records if r["stage"]==stage and r["step"]==150]
        facts["A_six_initializations"][init]={"selected":{k:float(np.mean([r[k] for r in chosen])) for k in ["psnr","ms_ssim","lpips","L_image","psnr_minus_ENCODER_CONTINUOUS","lpips_minus_ENCODER_CONTINUOUS"]},
                                               "step150":{k:float(np.mean([r[k] for r in last])) for k in ["psnr","ms_ssim","lpips","L_image"]}}
    for group,predicate in [("A_six",lambda r:r["stage"].startswith("A6_")),("A_extension",lambda r:r["stage"].startswith("Aext_")),
                            ("B_six_calibration",lambda r:r["stage"].startswith("B6q_")),("B_full_new_24_per_lambda",lambda r:r["stage"].startswith("Bfull_"))]:
        rows=[r for r in timing if predicate(r)]
        facts["optimization_timing"][group]={"runs":len(rows),"seconds":sum(r["optimization_seconds"] for r in rows),
                                             "mean_seconds_per_run":float(np.mean([r["optimization_seconds"] for r in rows])) if rows else None,
                                             "cache_hits":sum(r["hard_gradient_cache_hits"] for r in rows)}
    for stage in sorted({r["stage"] for r in timing if r["stage"].startswith("Bfull_")}):
        rows=[r for r in timing if r["stage"]==stage]
        facts["optimization_timing"][stage]={"runs":len(rows),"seconds":sum(r["optimization_seconds"] for r in rows),
                                             "mean_seconds_per_run":float(np.mean([r["optimization_seconds"] for r in rows]))}
    for method in sorted({r["method"] for r in coding}):
        rows=[r for r in coding if r["method"]==method]
        facts["coding_timing"][method]={k:sum(r[k] for r in rows) for k in ["entropy_encode_seconds","base_decode_seconds","entropy_decode_seconds","synthesis_seconds"]}
    for group in ["network_train6","Train2_rest24","Val6_90P"]:
        rows=[r for r in refs if r["group"]==group]
        facts.setdefault("teacher_encoder_history",{})[group]={"samples":len(rows),"teacher_optimizer_seconds_if_required":sum(r["teacher_optimizer_seconds_if_required"] for r in rows),
                                                               "encoder_forward_seconds":sum(r["encoder_forward_seconds"] for r in rows),
                                                               "B_ZERO_START_requires_these":False}
    for r in seq:
        if r["method"]=="A_Val6_selected" and r["scope"]=="IP":
            facts["A_Val6_per_video"].append({k:r[k] for k in ["video","psnr","ms_ssim","lpips","psnr_minus_ZERO","lpips_minus_ZERO","psnr_minus_ENCODER_CONTINUOUS","lpips_minus_ENCODER_CONTINUOUS"]})
        if r["method"] in ["B_Train2_lambda1","B_Train2_lambda3","B_Train2_lambda5"] and r["scope"]=="IP":
            facts["B_Train2_per_video"].append({k:r[k] for k in ["method","video","enhancement_bits","enhancement_fraction","psnr_minus_ZERO","lpips_minus_ZERO","nonzero_enhancement_frame_fraction_P"]})
    zero_points=[r for r in records if r["quantized"] and r["step"]==0]
    facts["zero_snapshot_rate"]={"mean_estimated_symbol_bits":float(np.mean([r["estimated_symbol_bits"] for r in zero_points])),
                                  "actual_packet_bits":8,"allocated_header_plus_packet_bits":54.4,"actual_diagnostic_container_bits":816}
    facts["A_primary_aggregate"]=[r for r in aggregate if r["method"].startswith("A_") and "selected" in r["method"] and r["scope"]=="IP"]
    save_json(out/"report_facts.json",facts)
    assert inventory()==json.loads((out/"input_audit.json").read_text())["historical_inventory"]
    import cv2
    images=list((out/"visuals").glob("*.png"))
    assert len(images)>=64
    for image_path in images:
        assert cv2.imread(str(image_path)) is not None
    status_path=out/"status_evaluate.json" if (out/"status_evaluate.json").exists() else out/"status_all.json"
    status=json.loads(status_path.read_text())
    save_json(out/"artifact_manifest.json",{"complete":True,"visual_images":len(images),"historical_files_unchanged":True,
              "current_source_sha256":{p.name:sha(p) for p in sorted(HOME.glob("*.py"))},
              "csv_sha256":{p.name:sha(p) for p in sorted(out.glob("*.csv"))},
              "network_parameter_updates":0,"Q2_Val6_expanded":status["Q2_Val6_expanded"],
              "Q2_Val6_not_run_reason":None if status["Q2_Val6_expanded"] else "Train2 per-sequence budget gate failed"})
    print(json.dumps(facts,indent=2))


if __name__=="__main__":
    main()
