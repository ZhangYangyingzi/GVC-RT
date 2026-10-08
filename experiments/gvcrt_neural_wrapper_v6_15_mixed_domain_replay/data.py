"""Uniform source video, four adjacent canonical frames, one shared crop."""
import numpy as np
from PIL import Image
from mixed_io import *
_cache={}
def sample_uvg(rng,device):
    import torch
    records=load(ROOT/'train_manifest.json')['videos'];v=rng.choice(records);assert v['name'] in SPLITS['train_fit']
    start=rng.randrange(61);x=rng.randrange(1920-256+1);y=rng.randrange(1080-256+1)
    frames=[];hashes=[]
    for i in range(start,start+4):
        key=(v['name'],i)
        if key not in _cache:
            p=Path(v['input_dir'])/f'frame_{i:06d}.png';assert sha(p)==v['frame_file_sha256'][i]
            with Image.open(p) as im:assert im.mode=='RGB' and im.size==(1920,1080);a=np.array(im,dtype=np.uint8)
            assert hashlib.sha256(a.tobytes()).hexdigest()==v['frame_rgb_sha256'][i];_cache[key]=a
        crop=_cache[key][y:y+256,x:x+256].copy();hashes.append(hashlib.sha256(crop.tobytes()).hexdigest());frames.append(torch.from_numpy(crop).permute(2,0,1).unsqueeze(0).to(device).float()/255)
    return frames,dict(video=v['name'],video_index=v['video_index'],source_path=v['source_path'],canonical_indices=list(range(start,start+4)),source_frame_indices=v['source_frame_indices'][start:start+4],crop_x=x,crop_y=y,crop_size=256,cropped_RGB_sha256=hashes,source_RGB_sha256=v['rgb_sha256'],exposure='training_exposed')

_v4=None
def sample(domain,rng,device):
    global _v4
    if domain=='uvg':return sample_uvg(rng,device)
    if _v4 is None:_v4=module('mixed_historical_sampler',V4/'train.py')
    records=load(ROOT/'train_manifest_ulong.json')['videos']
    frames,p=_v4.sample_clip(records,rng,device)
    v=next(v for v in records if v['filename']==p['video'])
    return frames,dict(video=p['video'],video_index=records.index(v),source_path=v['path'],canonical_indices=list(range(p['start'],p['start']+4)),source_frame_indices=list(range(p['start'],p['start']+4)),crop_x=p['crop_x'],crop_y=p['crop_y'],crop_size=256,cropped_RGB_sha256=[hashlib.sha256((f.detach().cpu().squeeze(0).permute(1,2,0).numpy()*255).round().astype(np.uint8).tobytes()).hexdigest() for f in frames],source_RGB_sha256=None,exposure='training_exposed')
