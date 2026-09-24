"""Finish numeric report summaries and check saved visualization artifacts without rerunning models."""
import argparse
import json
from common import HERE,CFG,sha,save_json,np
from prepare import inventory
from postprocess import read_csv


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results_v1")
    args=parser.parse_args()
    out=HERE/args.output
    assert json.loads((out/"verification.json").read_text())["all_checks_passed"]
    refs=read_csv(out/"oracle_references.csv")
    contrasts=read_csv(out/"oracle_reference_comparisons.csv")
    oracle=read_csv(out/"oracle_metrics.csv")
    nondom=read_csv(out/"oracle_nondominated.csv")
    rate=next(r for r in read_csv(out/"rate_breakdown.csv") if r["video"]=="ALL_VAL6")
    ref_summary={mode:{m:float(np.mean([r[m] for r in refs if r["mode"]==mode])) for m in ["psnr","ms_ssim","lpips"]} for mode in sorted({r["mode"] for r in refs})}
    final=[r for r in contrasts if r["step"]==150]
    results={"oracle_reference_means_six_P_frames":ref_summary,
             "step150_runs_dominating_base_only":sum(r["dominates_BASE_ONLY_C"] for r in final),
             "step150_runs_dominating_original_fp16":sum(r["dominates_ORIGINAL_FP16"] for r in final),
             "nondominated_points":len(nondom),
             "relative_delta_l2_range_step150":[min(r["relative_delta_l2"] for r in final),max(r["relative_delta_l2"] for r in final)],
             "max_abs_delta_over_all_logged_points":max(r["delta_latent_max_abs"] for r in oracle),
             "rate_component_percent":{k:rate[k]/rate["actual_bits"]*100 for k in ["symbol_coding_bits","total_header_bits","termination_decision_bits","lookahead_bits","alignment_bits"]}}
    save_json(out/"report_data.json",results)
    import cv2
    images=sorted((out/"plots").glob("*.png"))
    assert len(images)==24
    for path in images:
        assert cv2.imread(str(path)) is not None
    videos=sorted((out/"ablation_videos").glob("*.mkv"))
    assert len(videos)==90
    from fractions import Fraction
    fps={e["video_id"]:float(Fraction(e["metadata"]["avg_frame_rate"])) for e in json.loads((HERE/"manifest.json").read_text())["val6"]}
    for path in videos:
        cap=cv2.VideoCapture(str(path))
        assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==16
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))==1920 and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))==1080
        assert abs(cap.get(cv2.CAP_PROP_FPS)-fps[path.name[:36]])<1e-6
        cap.release()
    assert inventory()==json.loads((HERE/"input_audit.json").read_text())["v1_inventory"]
    save_json(out/"artifact_manifest.json",{"complete":True,"plots":len(images),"ablation_videos":len(videos),
              "V1_inventory_still_unchanged":True,"current_source_sha256":{p.name:sha(p) for p in sorted(HERE.glob("*.py"))},
              "csv_sha256":{p.name:sha(p) for p in sorted(out.glob("*.csv"))},
              "note":"Numerical verification preceded a plotting-only dependency repair; current source hashes include OpenCV renderer. No metric or optimization rerun."})
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
