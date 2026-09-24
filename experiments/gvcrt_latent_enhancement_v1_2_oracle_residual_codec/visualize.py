"""Matched fixed-frame crops and measured RD samples; interventions have no RD coordinates."""
import argparse
import cv2
from support import *


def read_frame(path,index):
    cap=cv2.VideoCapture(str(path))
    image=None
    for _ in range(index+1):
        ok,image=cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Missing/truncated video: {path}")
    cap.release()
    return image


def label(image,text):
    cv2.putText(image,text,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),3)
    cv2.putText(image,text,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
    return image


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results_v1")
    parser.add_argument("--resume",action="store_true",help="Only create missing visualization files")
    args=parser.parse_args()
    out=HERE/args.output
    folder=out/"visuals"
    folder.mkdir(exist_ok=args.resume)
    crop_map=json.loads((V11/"config.json").read_text())["oracle"]["crop_boxes_xyxy"]
    for split,prefix in [("train2","train2_full"),("val6","val6")]:
        if not (out/(prefix+"_ZERO_D")).exists():
            continue
        for entry in manifest()[split]:
            video=entry["video_id"]
            sources=[("SOURCE",V1/"run_v1/source_videos"/f"{video}.mkv"),
                     ("ORIGINAL QP0",out/(prefix+"_ORIGINAL_q0")/f"{video}.mkv"),
                     ("FIXED ZERO D",out/(prefix+"_ZERO_D")/f"{video}.mkv"),
                     ("TEACHER (uncoded)",out/(prefix+"_TEACHER")/f"{video}.mkv"),
                     ("CONTINUOUS (diagnostic)",out/(prefix+"_CONTINUOUS")/f"{video}.mkv"),
                     ("LEARNED fine l0",out/(prefix+"_CODED_l0")/f"{video}.mkv"),
                     ("LEARNED coarse l3",out/(prefix+"_CODED_l3")/f"{video}.mkv"),
                     ("DIRECT l2",out/(prefix+"_DIRECT_l2")/f"{video}.mkv")]
            for frame in [1,8,15]:
                full_path=folder/f"{video}_f{frame:02d}_full.png"
                crop_path=folder/f"{video}_f{frame:02d}_crop.png"
                if args.resume and full_path.exists() and crop_path.exists():
                    continue
                full=[]
                crops=[]
                x1,y1,x2,y2=crop_map.get(video,[768,348,1152,732])
                for title,path in sources:
                    if not path.exists():
                        continue
                    im=read_frame(path,frame)
                    full.append(label(cv2.resize(im,(480,270),interpolation=cv2.INTER_AREA),f"{title} f{frame}"))
                    crops.append(label(im[y1:y2,x1:x2].copy(),title))
                if len(full)==8:
                    if not full_path.exists():
                        cv2.imwrite(str(full_path),np.concatenate([np.concatenate(full[:4],1),np.concatenate(full[4:],1)],0))
                    if not crop_path.exists():
                        cv2.imwrite(str(crop_path),np.concatenate([np.concatenate(crops[:4],1),np.concatenate(crops[4:],1)],0))
    if (out/"aggregate_metrics.csv").exists():
        rows=read_csv(out/"aggregate_metrics.csv")
        colors={"ORIGINAL_FP32":(150,90,25),"ZERO_D":(0,0,0),"DIRECT":(40,145,195),"LEARNED":(70,135,35)}
        for cohort in ["Train6_sparse","Train2_full","Val6"]:
            if args.resume and (folder/f"{cohort}_measured_RD.png").exists():
                continue
            points=[r for r in rows if r["cohort"]==cohort and r["scope"]=="IP" and r["kind"] in colors and r["total_bpp"] is not None]
            if not points:
                continue
            canvas=np.full((570,1770,3),255,np.uint8)
            cv2.putText(canvas,f"{cohort}: measured complete I/P points; no diagnostic RD points",(35,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(20,20,20),1)
            x=np.log10([r["total_bpp"] for r in points])
            xmin,xmax=x.min()-.08,x.max()+.08
            for j,key in enumerate(["psnr","ms_ssim","lpips"]):
                l,t,r,b=85+585*j,90,530+585*j,430
                vals=[p[key] for p in points]
                lo,hi=min(vals),max(vals)
                margin=max((hi-lo)*.1,1e-4)
                lo-=margin
                hi+=margin
                def xy(row):
                    return int(l+(np.log10(row["total_bpp"])-xmin)/(xmax-xmin)*(r-l)),int(b-(row[key]-lo)/(hi-lo)*(b-t))
                for s in np.linspace(0,1,5):
                    xx,yy=int(l+s*(r-l)),int(b-s*(b-t))
                    cv2.line(canvas,(xx,t),(xx,b),(225,225,225),1)
                    cv2.line(canvas,(l,yy),(r,yy),(225,225,225),1)
                    cv2.putText(canvas,f"{10**(xmin+s*(xmax-xmin)):.4f}",(xx-20,b+24),cv2.FONT_HERSHEY_SIMPLEX,.36,(50,50,50),1)
                    cv2.putText(canvas,f"{lo+s*(hi-lo):.4f}",(l-70,yy+4),cv2.FONT_HERSHEY_SIMPLEX,.36,(50,50,50),1)
                cv2.rectangle(canvas,(l,t),(r,b),(100,100,100),1)
                cv2.putText(canvas,key,(l,t-20),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,30,30),1)
                cv2.putText(canvas,"Total bpp (log scale)",(l+90,b+50),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1)
                for kind,color in colors.items():
                    subset=sorted([p for p in points if p["kind"]==kind],key=lambda p:p["total_bpp"])
                    pts=[xy(p) for p in subset]
                    if len(pts)>1:
                        cv2.polylines(canvas,[np.array(pts,np.int32)],False,color,2,cv2.LINE_AA)
                    for pos in pts:
                        cv2.circle(canvas,pos,5,color,-1)
            for k,(kind,color) in enumerate(colors.items()):
                cv2.putText(canvas,kind,(70+k*410,530),cv2.FONT_HERSHEY_SIMPLEX,.65,color,1)
            cv2.imwrite(str(folder/f"{cohort}_measured_RD.png"),canvas)
    if not (folder/"visualization_manifest.json").exists():
        save_json(folder/"visualization_manifest.json",{"frames":[1,8,15],"Train2_crops":"unchanged V1.1 fixed boxes","Val6_crop":[768,348,1152,732],
                  "RD":"actual coded points only; aggregate visualization does not replace per-video overlap analysis","temporal_claims":False})


if __name__=="__main__":
    main()
