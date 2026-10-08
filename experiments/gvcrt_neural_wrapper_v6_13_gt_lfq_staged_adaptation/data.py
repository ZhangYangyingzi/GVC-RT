"""Deterministic plan-indexed native crops, identical between branches."""
import hashlib
from v68_io import *
_verified=set()
def sample(plan,device):
    import torch,cv2,numpy as np
    from PIL import Image
    x,y,w,h=plan['crop'];arrays=[]
    if plan['domain']=='ulong':
        if plan['path'] not in _verified:assert sha(plan['path'])==plan['source_sha256'];_verified.add(plan['path'])
        c=cv2.VideoCapture(plan['path']);c.set(cv2.CAP_PROP_POS_FRAMES,plan['frame_indices'][0])
        for _ in range(4):
            ok,bgr=c.read();assert ok;arrays.append(cv2.cvtColor(bgr[y:y+h,x:x+w],cv2.COLOR_BGR2RGB).copy())
        c.release()
    else:
        for path,expected in zip(plan['paths'],plan['source_sha256']):
            assert sha(path)==expected
            with Image.open(path) as im:assert im.mode=='RGB';arrays.append(np.asarray(im,dtype=np.uint8)[y:y+h,x:x+w].copy())
    hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
    if 'frame_rgb_sha256' in plan:assert hashes==plan['frame_rgb_sha256']
    frames=[torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(device).float()/255 for a in arrays]
    assert len(frames)==4 and all(f.shape==(1,3,256,256) for f in frames)
    return frames,hashes
