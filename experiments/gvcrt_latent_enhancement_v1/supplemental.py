"""Finish original Train2 rate-point metrics and create paired visual review sheets."""
import json
import os
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/"config.json").read_text())
os.environ["CUDA_VISIBLE_DEVICES"]=CFG["gpu_uuid"]
sys.path.insert(0,str(HERE.parents[1]))


def main():
    import lpips
    import torch
    import cv2
    import numpy as np
    from run_experiment import RUN,evaluate,fresh_generator
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((HERE/"manifest.json").read_text())
    train2=[e for e in manifest["train"] if e["video_id"] in manifest["train2"]]
    selected=json.loads((RUN/"qp_selection.json").read_text())["selected_low_rate_qp"]
    generator=fresh_generator()
    perceptual=lpips.LPIPS(net="alex",version="0.1").cuda().eval().requires_grad_(False)
    for qp in CFG["qp_candidates"]:
        if qp!=selected and not (RUN/f"original_train2_q{qp}/summary.json").exists():
            evaluate(f"original_train2_q{qp}",train2,qp,generator,perceptual,video=False)
    review=RUN/"review"
    review.mkdir(exist_ok=True)
    for entry in train2:
        video=entry["video_id"]
        methods=["source_videos","original_train2","continuous_b_train2","base_only_b_train2",
                 "quantized_c_train2_l0","quantized_c_train2_l1","quantized_c_train2_l2"]
        columns=[]
        for method in methods:
            path=RUN/method/f"{video}.mkv"
            if not path.exists():
                continue
            cap=cv2.VideoCapture(str(path))
            picks=[]
            for frame in range(entry["frames"]):
                ok,bgr=cap.read()
                if not ok:
                    raise RuntimeError("Review video truncated")
                if frame in [1,8,15]:
                    image=cv2.resize(bgr,(480,270),interpolation=cv2.INTER_AREA)
                    cv2.putText(image,f"{method} frame={frame}",(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3)
                    cv2.putText(image,f"{method} frame={frame}",(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                    picks.append(image)
            cap.release()
            columns.append(np.concatenate(picks,axis=0))
        cv2.imwrite(str(review/f"{video}.png"),np.concatenate(columns,axis=1))


if __name__=="__main__":
    main()
