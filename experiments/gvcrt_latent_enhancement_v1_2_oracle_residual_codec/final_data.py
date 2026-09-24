"""Numerical report facts and final artifact inventory, without new optimization."""
import argparse
import cv2
from support import *
from run import inventory


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results_v1")
    args=p.parse_args()
    out=HERE/args.output
    rows=read_csv(out/"all_metrics.csv")
    def point(method,scope="IP",video="ALL"):
        return next(r for r in rows if r["method"]==method and r["scope"]==scope and r["video"]==video)
    data={"timing":{},"quality_chain":{},"per_video_coarse":[],"paired_counts":{}}
    for split in ["train2","val6"]:
        timing=read_csv(out/f"{split}_sender_optimization_timing.csv")
        seconds=[r["optimization_seconds"] for r in timing]
        data["timing"][split]={"newly_optimized_frames":len(timing),"optimizer_updates":sum(r["updates"] for r in timing),
                                 "total_seconds":sum(seconds),"mean_seconds_per_P":float(np.mean(seconds)),
                                 "min_seconds_per_P":min(seconds),"max_seconds_per_P":max(seconds),
                                 "all_teachers_eligible":all(r["eligible"] for r in timing)}
    historic=read_csv(out/"historic_teacher_optimizer_timing.csv")
    data["timing"]["historic_six"]={"original_optimizer_seconds":sum(r["historic_optimizer_seconds"] for r in historic),
                                      "current_restore_and_check_seconds":sum(r["current_restore_and_verification_seconds"] for r in historic)}
    ct=read_csv(out/"coding_timing.csv")
    for method in ["train2_full_CODED_l3","val6_CODED_l0","val6_CODED_l3","val6_DIRECT_l0"]:
        s=[r for r in ct if r["method"]==method]
        data["timing"][method]={k:sum(r[k] for r in s) for k in ["residual_analysis_seconds","actual_entropy_encode_seconds",
            "receiver_entropy_decode_seconds","receiver_base_decode_seconds","receiver_synthesis_seconds"]}
        data["timing"][method]["complete_sequences"]=len(s)
    for prefix in ["train2_full","val6"]:
        teacher,continuous,fine,coarse=[point(prefix+"_"+x) for x in ["TEACHER","CONTINUOUS","CODED_l0","CODED_l3"]]
        data["quality_chain"][prefix]={"teacher":{m:teacher[m] for m in ["psnr","ms_ssim","lpips"]},
             "continuous":{m:continuous[m] for m in ["psnr","ms_ssim","lpips"]},
             "fine_actual":{m:fine[m] for m in ["psnr","ms_ssim","lpips"]},
             "coarse_actual":{m:coarse[m] for m in ["psnr","ms_ssim","lpips"]},
             "teacher_minus_continuous_psnr":teacher["psnr"]-continuous["psnr"],
             "continuous_minus_fine_psnr":continuous["psnr"]-fine["psnr"],
             "continuous_minus_coarse_psnr":continuous["psnr"]-coarse["psnr"],
             "coarse_lpips_minus_original_fp32":coarse["lpips"]-point(prefix+"_ORIGINAL_FP32_q0")["lpips"]}
    for e in manifest()["val6"]:
        video=e["video_id"]
        c=point("val6_CODED_l3","P",video)
        w=point("val6_WRONG_l3","P",video)
        data["per_video_coarse"].append({"video":video,"P_psnr_gain":c["psnr_gain_vs_ZERO_D"],"P_lpips_change":c["lpips_gain_vs_ZERO_D"],
                                         "REAL_minus_WRONG_psnr":c["psnr"]-w["psnr"],"REAL_minus_WRONG_lpips":c["lpips"]-w["lpips"],
                                         "enhancement_bits":c["enhancement_bits"],"base_bits":c["base_bits"],
                                         "enhancement_fraction":c["actual_enhancement_fraction"]})
    for level in range(4):
        rr=[point(f"val6_CODED_l{level}","P",e["video_id"]) for e in manifest()["val6"]]
        ww=[point(f"val6_WRONG_l{level}","P",e["video_id"]) for e in manifest()["val6"]]
        data["paired_counts"][str(level)]={"psnr_improved_videos":sum(r["psnr_gain_vs_ZERO_D"]>0 for r in rr),
             "lpips_improved_videos":sum(r["lpips_gain_vs_ZERO_D"]<0 for r in rr),
             "joint_psnr_lpips_improved_videos":sum(r["psnr_gain_vs_ZERO_D"]>0 and r["lpips_gain_vs_ZERO_D"]<0 for r in rr),
             "REAL_jointly_beats_WRONG_videos":sum(r["psnr"]>w["psnr"] and r["lpips"]<w["lpips"] for r,w in zip(rr,ww))}
    for level in range(4):
        frames=read_csv(out/f"val6_CODED_l{level}/frames.csv")
        estimated=sum(r["estimated_enhancement_symbol_bits"] for r in frames)
        actual=point(f"val6_CODED_l{level}")["enhancement_bits"]
        data.setdefault("estimated_actual",{})[str(level)]={"estimated_symbol_bits":estimated,"actual_all_file_bits":actual,
                                                             "difference_including_headers":actual-estimated}
    save_json(out/"report_data.json",data)
    assert inventory()==json.loads((out/"input_audit.json").read_text())["history_inventory"]
    plots=list((out/"visuals").glob("*.png"))
    assert len(plots)==51
    for plot in plots:
        assert cv2.imread(str(plot)) is not None
    save_json(out/"artifact_manifest.json",{"complete":True,"visual_png_count":len(plots),"source_code_sha256":{p.name:sha(p) for p in sorted(HERE.glob("*.py"))},
              "csv_sha256":{p.name:sha(p) for p in sorted(out.glob("*.csv"))},"historical_files_unchanged":True,
              "note":"The first visualization command hit a tool timeout; --resume generated missing images only. No models or numerical results were rerun."})
    print(json.dumps(data,indent=2))


if __name__=="__main__":
    main()
