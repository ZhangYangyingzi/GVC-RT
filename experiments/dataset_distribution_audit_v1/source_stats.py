"""Multiprocess CPU spatial, temporal and fixed camera-motion diagnostics."""
import argparse
import concurrent.futures
import math
import traceback
import cv2
import numpy as np
from audit_io import *
from frames import view_frames
VIEWS=('standardized_content_view','actual_codec_input_view')

def gray(x):return x[:,:,0]*.2126+x[:,:,1]*.7152+x[:,:,2]*.0722
def gradient(y):
    dx=cv2.Sobel(y,cv2.CV_32F,1,0,ksize=3,scale=1/8,borderType=cv2.BORDER_REFLECT_101)
    dy=cv2.Sobel(y,cv2.CV_32F,0,1,ksize=3,scale=1/8,borderType=cv2.BORDER_REFLECT_101)
    return np.hypot(dx,dy)
def camera(prev,current):
    cv2.setRNGSeed(0);a=cv2.cvtColor(prev,cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(current,cv2.COLOR_RGB2GRAY)
    pts=cv2.goodFeaturesToTrack(a,maxCorners=400,qualityLevel=.01,minDistance=8)
    if pts is None or len(pts)<6:return dict(status='INSUFFICIENT_FEATURES')
    nxt,ok,_=cv2.calcOpticalFlowPyrLK(a,b,pts,None,winSize=(21,21),maxLevel=3)
    if nxt is None or int(ok.sum())<6:return dict(status='INSUFFICIENT_TRACKS')
    mask=ok.ravel().astype(bool);mat,inlier=cv2.estimateAffinePartial2D(pts[mask],nxt[mask],method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=2000,confidence=.99,refineIters=10)
    if mat is None or not np.isfinite(mat).all():return dict(status='AFFINE_FAILED')
    h,w=a.shape;warped=cv2.warpAffine(prev.astype(np.float32)/255,mat,(w,h),flags=cv2.INTER_LINEAR)
    valid=cv2.warpAffine(np.ones((h,w),np.uint8),mat,(w,h),flags=cv2.INTER_NEAREST).astype(bool)
    if valid.mean()<.25:return dict(status='INSUFFICIENT_VALID_WARP',affine=mat.tolist())
    diff=warped[valid]-current.astype(np.float32)[valid]/255
    scale=math.hypot(mat[0,0],mat[1,0]);translation=math.hypot(mat[0,2],mat[1,2])
    return dict(status='PASS',affine=mat.tolist(),global_translation_magnitude=translation,global_translation_normalized=translation/math.hypot(h,w),
        global_rotation_magnitude=abs(math.degrees(math.atan2(mat[1,0],mat[0,0]))),global_scale_change=abs(scale-1),
        compensated_residual_L1=float(np.abs(diff).mean()),compensated_residual_MSE=float(np.square(diff).mean()),
        inlier_fraction=float(inlier.mean()),valid_warp_fraction=float(valid.mean()))

def calculate(v):
    cv2.setNumThreads(1);path=ROOT/'parts/source'/f'{v["sample_id"]}.json'
    if path.exists():r=load(path);assert r['status']=='PASS';return v['sample_id']
    output=[]
    for view in VIEWS:
        frames,meta=view_frames(v,view);sp=[];temporal=[];cameras=[];prev=None
        for i,a in enumerate(frames):
            x=a.astype(np.float32)/255;y=gray(x);g=gradient(y);hsv=cv2.cvtColor(x,cv2.COLOR_RGB2HSV)
            high=y-cv2.GaussianBlur(y,(5,5),1,borderType=cv2.BORDER_REFLECT_101)
            s=dict(mean_luminance=float(y.mean()),luminance_std=float(y.std()),
                saturation_mean=float(hsv[:,:,1].mean()),saturation_std=float(hsv[:,:,1].std()),
                sobel_mean=float(g.mean()),sobel_p90=float(np.quantile(g,.9)),laplacian_variance=float(cv2.Laplacian(y,cv2.CV_32F,ksize=3).var()),
                high_frequency_energy=float(np.square(high).mean()),edge_density=float((g>.05).mean()))
            s.update({f'RGB_{c}_std':float(x[:,:,j].std()) for j,c in enumerate('RGB')});sp.append(s)
            if prev is not None:
                px,py,pg=prev;d=x-px
                temporal.append(dict(temporal_RGB_L1=float(np.abs(d).mean()),temporal_RGB_MSE=float(np.square(d).mean()),
                    temporal_luminance_L1=float(np.abs(y-py).mean()),temporal_gradient_difference=float(np.abs(g-pg).mean())))
                cameras.append(dict(from_frame=i-1,to_frame=i,**camera(frames[i-1],a)))
            prev=(x,y,g)
        spatial={k:float(np.mean([r[k] for r in sp])) for k in sp[0]};temp={k:float(np.mean([r[k] for r in temporal])) for k in temporal[0]}
        values=[r['temporal_RGB_L1'] for r in temporal]
        temp.update({f'frame_difference_{name}':float(np.quantile(values,q)) for name,q in [('p50',.5),('p75',.75),('p90',.9),('p95',.95),('max',1.)]})
        valid=[r for r in cameras if r['status']=='PASS'];camkeys=('global_translation_magnitude','global_translation_normalized','global_rotation_magnitude','global_scale_change','compensated_residual_L1','compensated_residual_MSE')
        cam={k:float(np.mean([r[k] for r in valid])) if valid else None for k in camkeys}
        cam.update(camera_valid_pairs=len(valid),camera_total_pairs=len(cameras),camera_valid_fraction=len(valid)/len(cameras),
            camera_status='PASS' if len(valid)==len(cameras) else 'PARTIAL' if valid else 'UNAVAILABLE')
        output.append(dict(dataset=v['dataset'],sample_id=v['sample_id'],view=view,**meta,spatial=spatial,temporal=temp,camera=cam,camera_pairs=cameras,
            input_rgb_sha256=hashlib.sha256(b''.join(a.tobytes() for a in frames)).hexdigest()))
    dump(path,dict(status='PASS',sample_id=v['sample_id'],views=output));return v['sample_id']

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=12);p.add_argument('--limit',type=int);a=p.parse_args()
    videos=load(ROOT/'manifests/all_sources.json')['videos'];videos=videos[:a.limit] if a.limit else videos
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(calculate,v):v for v in videos}
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            try:f.result()
            except Exception:
                dump(ROOT/'logs'/f'source_failure_{futures[f]["sample_id"]}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
            if i%25==0 or i==len(videos):
                dump(ROOT/'source_progress.json',dict(status='RUNNING',completed=i,total=len(videos),updated_unix=time.time()));print('SOURCE',i,'/',len(videos),flush=True)
    if a.limit:return
    spatial=[];temporal=[];camera_rows=[]
    for v in videos:
        r=load(ROOT/'parts/source'/f'{v["sample_id"]}.json')
        for view in r['views']:
            meta={k:view[k] for k in ('dataset','sample_id','view','width','height','effective_fps','frame_count','source_indices','input_rgb_sha256')}
            spatial.append(dict(**meta,**view['spatial']));temporal.append(dict(**meta,**view['temporal']))
            camera_rows.append(dict(**meta,**view['camera'],local_motion_residual_L1=view['camera']['compensated_residual_L1'],local_motion_residual_MSE=view['camera']['compensated_residual_MSE']))
    write(ROOT/'source_spatial_statistics.csv',spatial);write(ROOT/'source_temporal_statistics.csv',temporal);write(ROOT/'camera_motion_statistics.csv',camera_rows)
    dump(ROOT/'source_progress.json',dict(status='PASS',completed=len(videos),total=len(videos),views_per_sample=2));print('CPU SOURCE COMPLETE',flush=True)
if __name__=='__main__':main()
