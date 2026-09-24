"""Dependency-light scientific plots from measured CSV values (OpenCV only)."""
import cv2
import numpy as np

COLORS=[(145,85,30),(25,130,215),(80,135,40),(130,65,155)]
SERIES=[("ORIGINAL_START","MSE_ONLY"),("ORIGINAL_START","MSE_LPIPS"),
        ("ZERO_START","MSE_ONLY"),("ZERO_START","MSE_LPIPS")]


def text(im,s,xy,scale=.5,color=(35,35,35)):
    cv2.putText(im,s,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def axis(im,box,xvalues,yvalues,title,xlabel):
    l,t,r,b=box
    xmin,xmax=min(xvalues),max(xvalues)
    ymin,ymax=min(yvalues),max(yvalues)
    dx=max(xmax-xmin,1e-6)
    dy=max(ymax-ymin,1e-6)
    xmin-=dx*.04
    xmax+=dx*.04
    ymin-=dy*.08
    ymax+=dy*.08
    def point(x,y):
        return int(l+(x-xmin)/(xmax-xmin)*(r-l)),int(b-(y-ymin)/(ymax-ymin)*(b-t))
    for f in np.linspace(0,1,5):
        x,y=int(l+(r-l)*f),int(b-(b-t)*f)
        cv2.line(im,(x,t),(x,b),(225,225,225),1)
        cv2.line(im,(l,y),(r,y),(225,225,225),1)
        text(im,f"{xmin+(xmax-xmin)*f:.2f}",(x-18,b+23),.4)
        text(im,f"{ymin+(ymax-ymin)*f:.4f}",(l-68,y+5),.4)
    cv2.rectangle(im,(l,t),(r,b),(90,90,90),1)
    text(im,title,(l,t-20),.52)
    text(im,xlabel,(l+(r-l)//3,b+50),.45)
    return point


def curves(path,rows,sid):
    im=np.full((580,1770,3),255,np.uint8)
    text(im,sid,(35,30),.57)
    for j,(key,title) in enumerate([("psnr","PSNR (higher better)"),("lpips","LPIPS (lower better)"),("relative_delta_l2","Relative correction L2")]):
        l=85+j*585
        point=axis(im,(l,90,l+445,410),[r["step"] for r in rows],[r[key] for r in rows],title,"Optimization step")
        for color,(initialization,objective) in zip(COLORS,SERIES):
            sub=sorted([r for r in rows if r["initialization"]==initialization and r["objective"]==objective],key=lambda r:r["step"])
            pts=[point(r["step"],r[key]) for r in sub]
            if len(pts)>1:
                cv2.polylines(im,[np.array(pts,np.int32)],False,color,2,cv2.LINE_AA)
            for xy in pts:
                cv2.circle(im,xy,4,color,-1,cv2.LINE_AA)
    for n,(color,(i,o)) in enumerate(zip(COLORS,SERIES)):
        text(im,f"{i} / {o}",(75+(n%2)*830,505+(n//2)*35),.55,color)
    if not cv2.imwrite(str(path),im):
        raise RuntimeError("Plot write failed")


def pareto(path,rows,sid):
    im=np.full((820,1150,3),255,np.uint8)
    text(im,sid,(40,30),.56)
    point=axis(im,(100,100,1085,565),[r["psnr"] for r in rows],[r["lpips"] for r in rows],
               "LPIPS (lower better); black rings = non-dominated measured points","PSNR (higher better)")
    for color,(initialization,objective) in zip(COLORS,SERIES):
        sub=sorted([r for r in rows if r["kind"]=="ORACLE" and r["initialization"]==initialization and r["objective"]==objective],key=lambda r:r["step"])
        pts=[point(r["psnr"],r["lpips"]) for r in sub]
        if len(pts)>1:
            cv2.polylines(im,[np.array(pts,np.int32)],False,color,2,cv2.LINE_AA)
        for r,xy in zip(sub,pts):
            cv2.circle(im,xy,5,color,-1,cv2.LINE_AA)
            text(im,str(int(r["step"])),(xy[0]+6,xy[1]-6),.38,color)
            if r["nondominated_psnr_lpips"]:
                cv2.circle(im,xy,10,(0,0,0),1,cv2.LINE_AA)
    refs=[r for r in rows if r["kind"]=="REFERENCE"]
    for index,r in enumerate(refs):
        xy=point(r["psnr"],r["lpips"])
        cv2.drawMarker(im,xy,(30,30,30),cv2.MARKER_TILTED_CROSS,12,2)
        text(im,f"R{index+1}",(xy[0]+8,xy[1]+18),.38)
        text(im,f'R{index+1} {r["mode"]}: PSNR={r["psnr"]:.3f}, LPIPS={r["lpips"]:.4f}',(65,695+index*27),.43)
    for n,(color,(i,o)) in enumerate(zip(COLORS,SERIES)):
        text(im,f"{i} / {o}",(65+(n%2)*565,635+(n//2)*28),.44,color)
    if not cv2.imwrite(str(path),im):
        raise RuntimeError("Plot write failed")
