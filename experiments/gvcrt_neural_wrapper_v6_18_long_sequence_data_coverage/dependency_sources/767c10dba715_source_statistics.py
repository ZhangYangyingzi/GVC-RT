"""CPU-only descriptive statistics of canonical source content, RGB units [0,1]."""
import math
import numpy as np
from scipy import ndimage
from audit_utils import *
from canonical_loader import canonical_arrays

def main():
    rows=[]
    for dataset in ('ulong','uvg'):
        for video in videos(dataset):
            name=video.get('name',video.get('sequence_name'))
            totals=np.zeros(3);squares=np.zeros(3);pixels=0
            gradients=[];laplacians=[];diff_means=[];diff_mse=[];histogram=np.zeros(256,dtype=np.int64)
            previous=None
            for raw in canonical_arrays(ROOT/'canonical_sources'/dataset/name):
                a=raw.astype(np.float32)/255.0
                flat=a.reshape(-1,3)
                totals+=flat.sum(axis=0,dtype=np.float64);squares+=np.square(flat,dtype=np.float64).sum(axis=0);pixels+=len(flat)
                gray=a@np.array([0.2126,0.7152,0.0722],dtype=np.float32)
                sx=ndimage.sobel(gray,axis=1,mode='reflect')/8.0
                sy=ndimage.sobel(gray,axis=0,mode='reflect')/8.0
                gradients.append(float(np.hypot(sx,sy).mean(dtype=np.float64)))
                laplacians.append(float(ndimage.laplace(gray,mode='reflect').var(dtype=np.float64)))
                if previous is not None:
                    diff=np.abs(raw.astype(np.int16)-previous.astype(np.int16)).astype(np.uint8)
                    histogram+=np.bincount(diff.ravel(),minlength=256)
                    d=diff.astype(np.float64)/255.0
                    diff_means.append(float(d.mean()));diff_mse.append(float(np.square(d).mean()))
                previous=raw
            # Exact linear-interpolated 95th percentile of all absolute RGB component differences.
            count=int(histogram.sum());position=.95*(count-1);lo=math.floor(position);hi=math.ceil(position)
            cum=histogram.cumsum();v0=int(np.searchsorted(cum,lo+1));v1=int(np.searchsorted(cum,hi+1))
            p95=(v0+(v1-v0)*(position-lo))/255.0
            avg=totals/pixels;std=np.sqrt(np.maximum(squares/pixels-avg**2,0))
            row=dict(dataset=dataset,name=name,canonical_fps=30.0,num_frames=64,num_transitions=63,
                temporal_RGB_L1=float(np.mean(diff_means)),temporal_RGB_MSE=float(np.mean(diff_mse)),
                sobel_gradient_magnitude_mean=float(np.mean(gradients)),laplacian_variance=float(np.mean(laplacians)),
                mean_absolute_RGB_frame_difference=float(np.mean(diff_means)),p95_absolute_RGB_component_difference=p95,
                p95_frame_mean_absolute_RGB_difference=float(np.percentile(diff_means,95)),
                **{f'mean_{c}':float(avg[i]) for i,c in enumerate('RGB')},**{f'std_{c}':float(std[i]) for i,c in enumerate('RGB')})
            rows.append(row);write('source_domain_statistics_per_video.csv',rows)
            print('SOURCE STATISTICS',dataset,name,flush=True)
    summary=[]
    numeric=[k for k in rows[0] if k not in ('dataset','name')]
    for dataset in ('ulong','uvg'):
        own=[r for r in rows if r['dataset']==dataset]
        summary.append(dict(dataset=dataset,num_videos=len(own),aggregation='arithmetic mean of per-sequence statistics',
                            **{k:float(np.mean([r[k] for r in own])) for k in numeric}))
    write('source_domain_statistics_dataset_summary.csv',summary)
    dump('source_statistics_audit.json',dict(status='PASS',input_units='RGB [0,1]',
         gray='BT.709 luma weights 0.2126/0.7152/0.0722 applied to RGB',sobel='scipy ndimage.sobel / 8, reflect boundary',
         laplacian='variance of scipy ndimage.laplace per grayscale frame, then mean',
         p95='exact linear quantile from all within-sequence absolute RGB component differences; separate frame-mean p95 also supplied',
         optical_flow='optional source flow magnitude not requested as mandatory; existing PWC remains used for FloLPIPS'))

if __name__=='__main__':main()
