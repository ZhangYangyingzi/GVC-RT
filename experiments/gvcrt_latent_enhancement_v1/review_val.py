"""Offline paired Val6 contact sheets, including the unenhanced I -> enhanced P transition."""
import json
import os
from pathlib import Path

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    run=HERE/os.environ.get("GVCRT_RUN_NAME","run_v1")
    out=run/"review_val6"
    out.mkdir(exist_ok=False)
    entries=json.loads((HERE/"manifest.json").read_text())["val"]
    notes=[]
    methods=["source_videos","original_val6_q0","base_only_d_val6","quantized_d_val6_l0","quantized_d_val6_l2"]
    for entry in entries:
        video=entry["video_id"]
        columns=[]
        arrays={}
        for method in methods:
            path=run/method/f"{video}.mkv"
            cap=cv2.VideoCapture(str(path))
            frames=[]
            for frame in range(entry["frames"]):
                ok,bgr=cap.read()
                if not ok:
                    raise RuntimeError(f"Missing or truncated review video {path}")
                frames.append(bgr.astype(np.float32)/255)
            cap.release()
            arrays[method]=frames
            picks=[]
            for frame in [0,1,8,15]:
                image=(cv2.resize(frames[frame],(480,270),interpolation=cv2.INTER_AREA)*255).round().astype(np.uint8)
                text=f'{method} f={frame} {"I" if frame==0 else "P"}'
                cv2.putText(image,text,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),3)
                cv2.putText(image,text,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                picks.append(image)
            columns.append(np.concatenate(picks,axis=0))
        cv2.imwrite(str(out/f"{video}.png"),np.concatenate(columns,axis=1))
        for method in methods[1:]:
            errors=[y-x for y,x in zip(arrays[method],arrays["source_videos"])]
            changes=[float(np.mean((b-a)**2)) for a,b in zip(errors,errors[1:])]
            notes.append({"video":video,"method":method,"I_to_P_error_difference_mse":changes[0],
                          "P_to_P_error_difference_mse":float(np.mean(changes[1:])),
                          "note":"8-bit lossless review images; separate from primary float quality metrics. Unwarped error differences are a diagnostic, not a flicker detector."})
    (out/"transition_diagnostics.json").write_text(json.dumps(notes,indent=2))


if __name__=="__main__":
    main()
