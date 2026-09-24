"""Evidence-only aggregation. Does not infer missing metrics or extrapolate RD curves."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-name",default="run_v1")
    args=parser.parse_args()
    run=HERE/args.run_name
    manifest=json.loads((HERE/"manifest.json").read_text())
    duration={e["video_id"]:e["duration_seconds"] for e in manifest["train"]+manifest["val"]}
    groups={}
    for path in sorted(run.glob("*/summary.json")):
        groups[path.parent.name]=json.loads(path.read_text())
    aggregate=[]
    for name,rows in groups.items():
        seconds=sum(duration[r["video"]] for r in rows)
        count=len(rows)*16
        base=sum(r["base_bits"] for r in rows)
        enhancement=sum(r["enhancement_bits"] for r in rows)
        estimated=sum(r["estimated_enhancement_bits"] for r in rows)
        aggregate.append({"method":name,"videos":len(rows),"frames":count,"duration_seconds":seconds,
                          "base_bits":base,"enhancement_bits":enhancement,"total_bits":base+enhancement,
                          "base_kbps":base/seconds/1000,"enhancement_kbps":enhancement/seconds/1000,"total_kbps":(base+enhancement)/seconds/1000,
                          "estimated_enhancement_bits":estimated,"actual_minus_estimated_bits":enhancement-estimated,
                          "bpp":(base+enhancement)/(count*1080*1920),
                          **{m:float(np.mean([r[m] for r in rows])) for m in ["psnr","ms_ssim","lpips"]},
                          "error_temporal_mse":float(np.mean([r["temporal"]["error_temporal_mse"] for r in rows])),
                          "transmittable":rows[0]["transmittable"]})
    with (run/"aggregate_metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (run/"aggregate_metrics.json").write_text(json.dumps(aggregate,indent=2))
    comparisons=[]
    for name,rows in groups.items():
        if not name.startswith("quantized"):
            continue
        val="val6" in name
        reference=groups.get("reference_fp32_val6" if val else "reference_fp32_train2",[])
        control=groups.get("base_only_d_val6" if val else "base_only_c_train2",[])
        continuous=groups.get("continuous_d_val6" if val else "continuous_c_train2",[])
        ref={r["video"]:r for r in reference}
        ctrl={r["video"]:r for r in control}
        cont={r["video"]:r for r in continuous}
        for row in rows:
            video=row["video"]
            item={"method":name,"video":video,"gain_vs_reference_db":row["psnr"]-ref[video]["psnr"],
                  "gain_vs_base_only_db":row["psnr"]-ctrl[video]["psnr"],
                  "continuous_minus_quantized_db":cont[video]["psnr"]-row["psnr"],
                  "delta_error_temporal_mse":row["temporal"]["error_temporal_mse"]-ref[video]["temporal"]["error_temporal_mse"]}
            prefix="original_val6_q" if val else "original_train2"
            original_points=[r for method,rs in groups.items() if method.startswith(prefix) for r in rs if r["video"]==video]
            original_points=sorted(original_points,key=lambda r:r["bpp"])
            x=np.array([r["bpp"] for r in original_points])
            y=np.array([r["psnr"] for r in original_points])
            if len(x)>=2 and x[0]<=row["bpp"]<=x[-1]:
                predicted=float(np.interp(np.log(row["bpp"]),np.log(x),y))
                item.update(original_interpolated_psnr=predicted,enhancement_minus_interpolated_original_db=row["psnr"]-predicted,
                            interpolation="piecewise linear PSNR vs log(bpp), per-video, only within sampled overlap")
            else:
                item.update(original_interpolated_psnr=None,enhancement_minus_interpolated_original_db=None,
                            interpolation="outside original sampled bitrate range; no extrapolation")
            comparisons.append(item)
    (run/"paired_comparisons.json").write_text(json.dumps(comparisons,indent=2))
    print("| 方法 | 基础 kbps | 增强 kbps | 总 kbps | PSNR | MS-SSIM | LPIPS |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in aggregate:
        total=f'{r["total_kbps"]:.3f}' if r["transmittable"] else "不可传输"
        print(f'| {r["method"]} | {r["base_kbps"]:.3f} | {r["enhancement_kbps"]:.3f} | {total} | {r["psnr"]:.4f} | {r["ms_ssim"]:.5f} | {r["lpips"]:.5f} |')
    print(json.dumps(comparisons,indent=2))


if __name__=="__main__":
    main()
