"""V6.3 native Vimeo loader, restricted to the official training split."""
import random
from v64_io import *
_verified=set()
def sample_clip(source,rng,device,v4):
    import torch,numpy as np
    from PIL import Image
    assert source=='vimeo', 'Only Vimeo official training split is allowed'
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
