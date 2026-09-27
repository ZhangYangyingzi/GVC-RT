"""CPU-only complete prefix readability and established RGB-conversion regression."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from PIL import Image
from audit_utils import *
from evaluate import uvg_frames

def main():
    old_root=ROOT.parent/'gvcrt_neural_wrapper_v2_retest_fixed_uvg'
    previous=load(old_root/'test_manifest.json')['videos'];rows=[]
    for v in videos('uvg'):
        frames,digest,command=uvg_frames(v,torch,np)
        assert len(frames)==v['frames_evaluated'] and all(tuple(f.shape)==(1,3,1080,1920) for f in frames)
        old=next((r for r in previous if r.get('source_path')==v['path']),None)
        compared=0
        if old:
            directory=old_root/'source_frames_fixed'/f"uvg_{int(old['video_id']):02d}"
            for i,f in enumerate(frames):
                path=directory/f'im{i+1}.png'
                if not path.exists():break
                a=np.asarray(Image.open(path).convert('RGB'),dtype=np.uint8)
                b=f[0].permute(1,2,0).mul(255).round().byte().numpy()
                assert np.array_equal(a,b),(v['sequence_name'],i,'prior RGB source mismatch')
                compared+=1
            assert compared>0,'previous UVG corrected source missing'
        rows.append(dict(sequence_name=v['sequence_name'],frames_decoded=len(frames),RGB24_prefix_sha256=digest,
            previous_corrected_RGB_frames_compared=compared,previous_conversion_compatible=True if compared else None,
            ffmpeg_command=command,status='PASS'))
        print(v['sequence_name'],'RGB PREFIX PASS',len(frames),'prior frames matched',compared,flush=True)
        del frames
    dump('uvg_decode_audit.json',dict(status='PASS',total_frames_decoded=sum(r['frames_decoded'] for r in rows),sequences=rows))

if __name__=='__main__':main()
