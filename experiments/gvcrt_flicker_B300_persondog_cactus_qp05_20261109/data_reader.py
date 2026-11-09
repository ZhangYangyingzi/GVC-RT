"""Task-private128-frame RGB8 windows; match every sealed first64 RGB hash."""
from common_s2 import *
import numpy as np

def read_arrays(row,full=True,verify=True):
    assert full and verify
    proof=load(ROOT/'audits/data'/f'{row["sequence"]}.json');assert proof['status']=='PASS'
    path=ROOT/proof['RGB_cache'];assert sha(path)==proof['RGB_cache_SHA256']
    values=np.load(path,mmap_mode='r');assert values.shape==(128,row['height'],row['width'],3) and values.dtype==np.uint8
    result=[values[i] for i in range(128)]
    assert row['source_frame_indices']==proof['source_frame_indices'] and row['frame_RGB_SHA256']==proof['frame_RGB_SHA256']
    for i,value in enumerate(result):assert hashlib.sha256(value.tobytes()).hexdigest()==row['frame_RGB_SHA256'][i]
    return result
