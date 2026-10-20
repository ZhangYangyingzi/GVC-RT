"""Fixed v6.39 figures; no visual example selection by outcome."""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
FIG=HERE/"figures"
FIG.mkdir(exist_ok=True)
METHODS={"M00":"Y_originalBridge_fullG","M10":"C_zF_spatial_z0_stats",
         "M01":"S_z0_spatial_zF_stats","M11":"F_fullLoss_noP_500"}
COLORS={"M00":"#515151","M10":"#0072b2","M01":"#d55e00","M11":"#009e73"}


def read(name):
    with (HERE/name).open(newline="") as f:return list(csv.DictReader(f))


def save(fig,name):
    fig.tight_layout()
    fig.savefig(FIG/name,dpi=145)
    plt.close(fig)


def main():
    points=read("results/all_rd_points.csv")
    pooled=read("results/pooled_fid.csv")
    effects=read("results/two_by_two_effects.csv")
    gamma=read("results/adagn_gamma_beta_statistics.csv")
    mapping="M00=Y_originalBridge_fullG; M10=C_zF_spatial_z0_stats; M01=S_z0_spatial_zF_stats; M11=F_fullLoss_noP_500"
    made=[]
    for ds in ("ulong","uvg_holdout","hevc_b"):
        for metric in ("LPIPS","DISTS","FloLPIPS","FID"):
            fig,ax=plt.subplots(figsize=(9,5.5))
            if metric=="FID":
                for m in METHODS:
                    rr=sorted((r for r in pooled if r["dataset"]==ds and r["method"]==m),key=lambda x:float(x["mean_actual_bpp"]))
                    if len(rr)==3:
                        ax.plot([float(r["mean_actual_bpp"]) for r in rr],[float(r["FID"]) for r in rr],"o-",label=m,color=COLORS[m])
                ax.set_xlabel("Pooled dataset mean actual RANS bpp")
                ax.set_title(f"{ds} pooled FID, 64 frames/video, lower is better")
            else:
                videos=sorted({int(r["video"]) for r in points if r["dataset"]==ds})
                for j,v in enumerate(videos):
                    for m in METHODS:
                        rr=sorted((r for r in points if r["dataset"]==ds and int(r["video"])==v and r["method"]==m),key=lambda x:float(x["actual_bpp"]))
                        if len(rr)==3:
                            ax.plot([float(r["actual_bpp"]) for r in rr],[float(r[metric]) for r in rr],"o-",alpha=.35 if len(videos)>2 else .8,
                                    label=m if j==0 else None,color=COLORS[m],markersize=3)
                ax.set_xlabel("Actual complete RANS bpp; separate line per video")
                ax.set_title(f"{ds} {metric}, lower is better")
            ax.set_xscale("log")
            ax.set_ylabel(metric)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
            fig.text(.02,.005,mapping,fontsize=6)
            name=f"{ds}_{metric.lower()}_RD.png";save(fig,name);made.append(name)
    for metric in ("LPIPS","DISTS","FloLPIPS","FID"):
        fig,ax=plt.subplots(figsize=(9,4.5))
        rr=[r for r in effects if r["level"] in ("dataset_video_equal_weight","dataset_pooled_equal_rate") and r["metric"]==metric]
        labels=[];values=[]
        for ds in ("ulong","uvg_holdout","hevc_b"):
            for effect in ("spatial_given_old_stats","stats_given_old_spatial","joint"):
                row=next((r for r in rr if r["dataset"]==ds and r["effect"]==effect),None)
                if row:labels.append(f"{ds}\n{effect}");values.append(float(row["delta"]))
        ax.bar(np.arange(len(values)),values,color=["#0072b2","#d55e00","#009e73"]*(len(values)//3))
        ax.axhline(0,color="black",lw=.8)
        ax.set_xticks(np.arange(len(values)),labels,rotation=45,ha="right",fontsize=7)
        ax.set_ylabel(f"Quality distance delta ({metric}); negative improves")
        ax.set_title("2x2 conditional effects on each dataset's common actual-bpp interval")
        fig.text(.02,.005,mapping,fontsize=6)
        name=f"two_by_two_{metric.lower()}.png";save(fig,name);made.append(name)
    for metric in ("LPIPS","DISTS","FloLPIPS","FID"):
        fig,ax=plt.subplots(figsize=(8,4.5))
        rr=[r for r in effects if r["level"] in ("video_QP","dataset_QP_pooled") and r["metric"]==metric and r["effect"] in ("spatial_given_old_stats","stats_given_old_spatial")]
        for label,color in (("spatial_given_old_stats","#0072b2"),("stats_given_old_spatial","#d55e00")):
            for ds,marker in (("ulong","o"),("uvg_holdout","s"),("hevc_b","^")):
                xs=[];ys=[]
                for qp in (0,5,9):
                    x=[float(r["delta"]) for r in rr if r["dataset"]==ds and r["effect"]==label and int(r["QP"])==qp]
                    if x:xs.append(qp);ys.append(float(np.mean(x)))
                if xs:ax.plot(xs,ys,marker=marker,color=color,label=f"{ds}: {label}")
        ax.axhline(0,color="black",lw=.8);ax.set_xticks((0,5,9));ax.set_xlabel("External QP (same persisted stream)")
        ax.set_ylabel(f"Mean delta {metric}; negative improves")
        ax.legend(fontsize=7,ncol=2);ax.set_title("Prespecified conditional effects by QP")
        fig.text(.02,.005,mapping,fontsize=6)
        name=f"conditional_effects_by_qp_{metric.lower()}.png";save(fig,name);made.append(name)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,kind in zip(axs,("gamma","beta")):
        for m in METHODS:
            vals=[next((float(r["mean_rms"]) for r in gamma if r["dataset"]=="ulong" and r["method"]==m and r["kind"]==kind and r["layer"]==layer),math.nan)
                  for layer in ("ada1","ada2","ada3","ada4","ada_final")]
            ax.plot(range(5),vals,"o-",label=m,color=COLORS[m])
        ax.set_xticks(range(5),("ada1","ada2","ada3","ada4","final"),rotation=25)
        ax.set_title(f"U-Long {kind} linear output RMS")
        ax.set_ylabel("RMS")
    axs[1].legend(fontsize=7)
    fig.text(.02,.005,mapping,fontsize=6)
    save(fig,"adagn_gamma_beta_rms.png");made.append("adagn_gamma_beta_rms.png")
    # Fixed source/order/frame/region: no ranking by observed quality.
    from json import loads
    old=HERE.parent/"gvcrt_neural_wrapper_v6_37_loss_bridge_generator_mechanism_500"
    ev=loads((old/"eval_manifest.json").read_text())
    for ds in ("ulong","uvg_holdout","hevc_b"):
        video=ev[ds][0]["video_index"]
        for qp in (0,5,9):
            key=f"{ds}_{video}_q{qp}"
            row=next((p for p in points if p["dataset"]==ds and int(p["video"])==video and int(p["QP"])==qp),None)
            if row is None:continue
            for frame in (16,32,48,63):
                names=("GT",*METHODS)
                imgs=[]
                for name in names:
                    path=HERE/"artifacts"/"examples"/key/name/f"{frame}.png"
                    if not path.exists():break
                    image=Image.open(path).convert("RGB")
                    imgs.append(image.crop((832,384,1088,640)))
                if len(imgs)!=5:continue
                canvas=Image.new("RGB",(256*5,286),"white")
                draw=ImageDraw.Draw(canvas)
                for i,(name,image) in enumerate(zip(names,imgs)):
                    canvas.paste(image,(i*256,26));draw.text((i*256+3,5),name,fill="black")
                draw.text((4,268),f"{ds} video {video}; frame {frame}; QP {qp}; shared actual bpp {float(row['actual_bpp']):.8f}; fixed crop [832:1088,384:640]",fill="black")
                name=f"fixed_{key}_frame{frame}.png"
                canvas.save(FIG/name);made.append(name)
    (HERE/"FIGURE_INDEX.md").write_text("# Fixed figures\n\nAll distances are lower-is-better. Four methods share the same true stream at each video/QP. Negative conditional deltas improve quality; no rate saving is inferred.\n\n"+"\n".join(f"- `{x}`" for x in made)+"\n")
    print("FIGURES",len(made),flush=True)


if __name__=="__main__":main()
