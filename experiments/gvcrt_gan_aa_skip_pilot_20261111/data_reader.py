"""New native windows get their own read receipts; no old-window RGB hashes are borrowed."""
from common_s2 import *
from inherited_reader import source_file,read_arrays as inherited_read_arrays

def read_arrays(row,full=False,verify=True):
    if not row.get('logical_sample'):return inherited_read_arrays(row,full,verify)
    assert row['domain']=='ulong' and row['source_frame_indices']==row['canonical_indices']
    indices=row['source_frame_indices'];assert len(indices)==64 and indices==list(range(indices[0],indices[0]+64))
    assert 0<=indices[0]<=row['source_total_frames']-64 and 'full_RGB_sha256' not in row and 'cropped_RGB_sha256' not in row
    expected=dict(row);sample_key=expected.pop('sample_key');assert digest(expected)==sample_key
    import cv2,numpy as np
    cv2.setNumThreads(1);result=[];hashes=[];crops=[]
    with source_file(row) as path:
        cap=cv2.VideoCapture(str(path))
        try:
            assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==row['source_total_frames']
            cap.set(cv2.CAP_PROP_POS_FRAMES,indices[0])
            for index in indices:
                ok,bgr=cap.read();assert ok,('NEW_WINDOW_READ_FAILED',row['video'],index)
                assert abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(index+1))<.1
                a=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert a.shape==(row['height'],row['width'],3),'NEW_NATIVE_GEOMETRY_MISMATCH'
                crop=a[row['crop_y']:row['crop_y']+256,row['crop_x']:row['crop_x']+256].copy();assert crop.shape==(256,256,3)
                hashes.append(hashlib.sha256(a.tobytes()).hexdigest());crops.append(hashlib.sha256(crop.tobytes()).hexdigest())
                result.append(a.copy() if full else crop)
        finally:cap.release()
    proof=dict(status='PASS',sample_key=sample_key,source_sha256=row['source_sha256'],source_frame_indices=indices,
        native_shape=[row['height'],row['width'],3],crop=[row['crop_x'],row['crop_y'],256,256],RGB_frame_SHA256=hashes,crop_frame_SHA256=crops,
        RGB_window_SHA256=digest(hashes),crop_window_SHA256=digest(crops),real_contiguous_frames=64,no_resize_repetition_padding_or_stitch=True)
    path=ROOT/f'audits/window_reads/{sample_key}.json.gz'
    with lock('locks/window_'+sample_key+'.lock',blocking=True):
        if path.exists():assert load(path)==proof,'REPEATED_NATIVE_WINDOW_HASH_MISMATCH'
        else:dump(path,proof)
    return result

class PlanFrames:
    def __init__(self,row,device,full=False):
        self.device=device;self.arrays=read_arrays(row,full)
    def __len__(self):return 64
    def __getitem__(self,i):
        import torch
        if i>=64:raise IndexError(i)
        return torch.from_numpy(self.arrays[i]).permute(2,0,1).unsqueeze(0).to(self.device).float()/255
    def close(self):self.arrays=None
