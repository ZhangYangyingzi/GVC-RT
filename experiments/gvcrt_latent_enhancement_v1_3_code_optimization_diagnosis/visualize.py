"""Measured optimization plots and fixed-frame comparisons, without assigning rates to diagnostics."""
import argparse
import cv2
from bridge import *


def frame(path,index):
    cap=cv2.VideoCapture(str(path))
    result=None
    for _ in range(index+1):
        ok,result=cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Missing video frame: {path}")
    cap.release()
    return result


def label(im,text):
    cv2.putText(im,text,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),3)
    cv2.putText(im,text,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
    return im


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results_v1")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    out=HOME/args.output
    target=out/"visuals"
    target.mkdir(exist_ok=args.resume)
    crop_map=json.loads((V11/"config.json").read_text())["oracle"]["crop_boxes_xyxy"]
    for split in ["Train2","Val6"]:
        oldprefix="train2_full" if split=="Train2" else "val6"
        budget_path=out/f"B_{split}_budget_selections.csv"
        selections=read_csv(budget_path) if budget_path.exists() else []
        for e in entries(split):
            video=e["video_id"]
            methods=[("SOURCE",V1/"run_v1/source_videos"/f"{video}.mkv"),
                     ("FIXED ZERO",OLD/"results_v1"/f"{oldprefix}_ZERO_D"/f"{video}.mkv"),
                     ("ENCODER CONTINUOUS",OLD/"results_v1"/f"{oldprefix}_CONTINUOUS"/f"{video}.mkv"),
                     ("FULL LATENT TEACHER",OLD/"results_v1"/f"{oldprefix}_TEACHER"/f"{video}.mkv")]
            a=out/"sequences"/f"A_{split}_selected"/f"{video}.mkv"
            if a.exists():
                methods.append(("OPTIMIZED CONTINUOUS",a))
            selected=next((r for r in selections if r["video"]==video and r["budget"]==.5),None)
            if selected:
                methods.append(("B 50%: "+("ZERO fallback" if selected["zero_no_stream_fallback"] else selected["method"]),out/"sequences"/selected["method"]/f"{video}.mkv"))
            methods=[(title,path) for title,path in methods if path.exists()]
            for f in [0,1,8,15]:
                full=target/f"{video}_f{f:02d}_comparison.png"
                crop=target/f"{video}_f{f:02d}_crop.png"
                if args.resume and full.exists() and crop.exists():
                    continue
                x1,y1,x2,y2=crop_map.get(video,[768,348,1152,732])
                panels=[]
                patches=[]
                for title,path in methods:
                    im=frame(path,f)
                    panels.append(label(cv2.resize(im,(480,270),interpolation=cv2.INTER_AREA),f"{title} f{f}"))
                    patches.append(label(im[y1:y2,x1:x2].copy(),title))
                if not full.exists():
                    cv2.imwrite(str(full),np.concatenate(panels,1))
                if not crop.exists():
                    cv2.imwrite(str(crop),np.concatenate(patches,1))
    # One plot per recorded code run, with image objective and actual packet cost.
    for path in sorted((out/"optimizations").glob("*/*/checkpoints.csv")):
        stage=path.parent.parent.name
        if not stage.startswith(("A6_","Aext_","B6q_","Bfull_")):
            continue
        destination=target/f"{stage}_{path.parent.name}_curve.png"
        if args.resume and destination.exists():
            continue
        rows=read_csv(path)
        canvas=np.full((440,1550,3),255,np.uint8)
        cv2.putText(canvas,f"{stage}: {path.parent.name}",(25,25),cv2.FONT_HERSHEY_SIMPLEX,.52,(30,30,30),1)
        for j,key in enumerate(["psnr","lpips","code_l2"]):
            l,t,r,b=70+j*515,80,485+j*515,335
            lo=min(x[key] for x in rows)
            hi=max(x[key] for x in rows)
            margin=max((hi-lo)*.1,1e-6)
            lo-=margin
            hi+=margin
            points=[]
            for row in rows:
                xy=(int(l+row["step"]/150*(r-l)),int(b-(row[key]-lo)/(hi-lo)*(b-t)))
                points.append(xy)
                cv2.circle(canvas,xy,4,(140,80,30),-1)
                if row["selected"]:
                    cv2.circle(canvas,xy,9,(0,0,0),1)
            cv2.polylines(canvas,[np.array(points,np.int32)],False,(140,80,30),2)
            cv2.rectangle(canvas,(l,t),(r,b),(120,120,120),1)
            cv2.putText(canvas,key,(l,t-18),cv2.FONT_HERSHEY_SIMPLEX,.55,(30,30,30),1)
            for pos,value in [(t,hi),(b,lo)]:
                cv2.putText(canvas,f"{value:.4f}",(l-65,pos+5),cv2.FONT_HERSHEY_SIMPLEX,.35,(50,50,50),1)
            for step in [0,25,50,100,150]:
                cv2.putText(canvas,str(step),(int(l+step/150*(r-l))-10,b+23),cv2.FONT_HERSHEY_SIMPLEX,.4,(50,50,50),1)
        cv2.putText(canvas,"Measured snapshots; black ring = predeclared objective selection. Not a bitrate RD curve.",(30,400),cv2.FONT_HERSHEY_SIMPLEX,.5,(35,35,35),1)
        cv2.imwrite(str(destination),canvas)
    if not (target/"manifest.json").exists():
        save_json(target/"manifest.json",{"frames":[0,1,8,15],"crop_rule":"existing Train2 fixed crops; Val6 fixed central 384 square",
                  "budget_panel":"actual selected whole candidate at 50%, or explicitly labeled ZERO no-stream fallback",
                  "temporal_claim":"short playback and fixed-time inspection only, no long-term stability claim"})


if __name__=="__main__":
    main()
