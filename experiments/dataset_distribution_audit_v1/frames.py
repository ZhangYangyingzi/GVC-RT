"""Diagnostic frame views; native sources and old canonical inputs are read-only."""
import fcntl
import subprocess
import zipfile
import shutil
import cv2
import numpy as np
from PIL import Image
from audit_io import *

def materialize(v):
    p=Path(v['path']);p.parent.mkdir(parents=True,exist_ok=True) if p.is_relative_to(ROOT) else None
    if not p.exists():
        assert v['dataset']=='ulong_unused1024'
        with p.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not p.exists():
                with zipfile.ZipFile(v['archive']) as z,z.open(v['archive_member']) as src,p.with_suffix('.partial').open('wb') as dst:shutil.copyfileobj(src,dst,1<<20)
                assert sha(p.with_suffix('.partial'))==v['sha256'];p.with_suffix('.partial').replace(p)
    assert sha(p)==v['sha256'];return p

def native_frames(v,standardized=False):
    if v['kind']=='vimeo':
        out=[]
        for p,h in zip(v['frame_paths'],v['frame_sha256']):
            assert sha(p)==h
            with Image.open(p) as im:out.append(np.asarray(im,dtype=np.uint8).copy())
        return out,list(range(len(out))),None
    if v['kind']=='frozen':
        out=[];fps=v['fps'];stride=max(1.,fps/30.) if standardized else 1.
        indices=sorted(set(int(np.floor(i*stride+.5)) for i in range(32)));indices=[i for i in indices if i<v['source_frames']]
        for i in indices:
            p=Path(v['input_dir'])/(f'frame_{i:06d}.png' if v['frozen_dataset']=='uvg' else f'im{i+1:05d}.png')
            assert sha(p)==v['frame_file_sha256'][i]
            with Image.open(p) as im:out.append(np.asarray(im,dtype=np.uint8).copy())
        return out,indices,min(fps,30.) if standardized else fps
    p=materialize(v);fps=v['fps'];stride=max(1.,fps/30.) if standardized else 1.
    indices=sorted(set(int(np.floor(i*stride+.5)) for i in range(32)));indices=[i for i in indices if i<v['source_frames']]
    assert len(indices)>=2
    cmd=['ffmpeg','-v','error','-threads','1','-i',str(p),'-vsync','0','-frames:v',str(indices[-1]+1),'-threads','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE);out=[];size=v['width']*v['height']*3
    try:
        keep=set(indices)
        for i in range(indices[-1]+1):
            raw=proc.stdout.read(size);assert len(raw)==size,(v['sample_id'],i,'short decode')
            if i in keep:out.append(np.frombuffer(raw,np.uint8).reshape(v['height'],v['width'],3).copy())
        err=proc.stderr.read();assert proc.wait()==0,err.decode()
    finally:
        proc.stdout.close();proc.stderr.close()
        if proc.poll() is None:proc.terminate();proc.wait()
    return out,indices,min(fps,30.) if standardized else fps

def view_frames(v,view):
    standard=view=='standardized_content_view';frames,indices,fps=native_frames(v,standard)
    if standard:
        h,w=frames[0].shape[:2];ratio=512/max(h,w);dims=(round(w*ratio),round(h*ratio))
        frames=[cv2.resize(a,dims,interpolation=cv2.INTER_AREA if ratio<=1 else cv2.INTER_LINEAR) for a in frames]
    return frames,dict(source_indices=indices,effective_fps=fps,frame_count=len(frames),width=frames[0].shape[1],height=frames[0].shape[0])
