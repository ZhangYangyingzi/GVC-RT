"""Unchanged U-Long sample_clip plus native Vimeo septuplet adapter."""
import random
from v63_io import *
_verified=set()
def sample_clip(source,rng,device,v4):
    import torch,numpy as np
    from PIL import Image
    if source=='ulong':
        records=load(ROOT/'manifests/ulong_train1024.json')['videos']
        frames,plan=v4.sample_clip(records,rng,device,crop=256)
        record=next(r for r in records if r['filename']==plan['video'])
        if record['path'] not in _verified:
            assert sha(record['path'])==record['sha256'];_verified.add(record['path'])
        plan.update(source=source,sample_id=record['filename'],source_sha256=record['sha256'],temporal_indices=list(range(plan['start'],plan['start']+4)))
    else:
        manifest=load(ROOT/'manifests/vimeo_official_train.json');sid=rng.choice(manifest['sequences'])
        start=rng.randrange(4);x=rng.randrange(193);y=rng.randrange(1);frames=[];hashes=[]
        for i in range(start+1,start+5):
            p=Path(manifest['sequence_root'])/sid/f'im{i}.png'
            with Image.open(p) as im:
                assert im.mode=='RGB' and im.size==(448,256)
                a=np.asarray(im,dtype=np.uint8)[y:y+256,x:x+256].copy()
            frames.append(torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(device).float()/255);hashes.append(sha(p))
        plan=dict(source=source,sample_id=sid,video=sid,start=start,crop_x=x,crop_y=y,temporal_indices=list(range(start,start+4)),native_image_indices=list(range(start+1,start+5)),frame_sha256=hashes)
    assert len(frames)==4 and all(tuple(f.shape)==(1,3,256,256) and bool(torch.isfinite(f).all()) and 0<=float(f.min())<=float(f.max())<=1 for f in frames)
    plan.update(crop_width=256,crop_height=256,stride=1,resize=False)
    return frames,plan
