import hashlib
import subprocess
import numpy as np
from diag_utils import *
FILTER='scale=in_color_matrix=bt709:in_range=limited:out_range=full,format=rgb24'

def command(s,bt709=False,n=96):
    # Reproduce the exact CURRENT command from the frozen source audit.
    record=old.load(f'parts/source_uvg_original_{s["video_index"]}.json')
    cmd=list(record['ffmpeg_command'])
    cmd[cmd.index('-frames:v')+1]=str(n)
    if bt709:cmd[cmd.index('-frames:v'):cmd.index('-frames:v')]=['-vf',FILTER]
    return cmd

def decode(s,bt709=False,n=96,path=None):
    cmd=command(s,bt709,n);print('FFMPEG COMMAND',json.dumps(cmd),flush=True)
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE)
    shape=(n,1080,1920,3)
    if path:
        path.parent.mkdir(parents=True,exist_ok=True)
        arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=shape)
    else:arr=np.empty(shape,dtype=np.uint8)
    h=hashlib.sha256()
    for i in range(n):
        raw=p.stdout.read(1080*1920*3);assert len(raw)==1080*1920*3
        h.update(raw);arr[i]=np.frombuffer(raw,np.uint8).reshape(1080,1920,3)
    p.stdout.close();assert p.wait()==0
    if path:arr.flush()
    return arr,h.hexdigest(),cmd

def ensure_source(s,torch,eco,smoke=False):
    path=rawpath(s,'source',smoke=smoke)
    if s['dataset']=='uvg':
        if not path.exists():
            arr,digest,cmd=decode(s,smoke,96,path)
            if not smoke:assert digest==old.load(f'parts/source_uvg_original_{s["video_index"]}.json')['source_hash']
            dump(f'parts/source_{tag(s)}_{"709" if smoke else "current"}.json',
                dict(status='PASS',RGB24_sha256=digest,command=cmd,path=str(path),num_frames=96))
        arr=np.load(path,mmap_mode='r')
        frames=[torch.from_numpy(np.array(a)).permute(2,0,1).unsqueeze(0).float()/255 for a in arr]
    else:
        final=module('diag_frozen_frames',old.V4/'final_evaluate.py')
        frames=final.load_frames(final.tag(s['video']))
        digest=hashlib.sha256(''.join(eco.tensor_sha(f) for f in frames).encode()).hexdigest()
        assert digest==old.load(f'parts/source_ulong_original_{s["video_index"]}.json')['source_hash']
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True);arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=(64,1080,1920,3))
            for i,f in enumerate(frames):arr[i]=f[0].permute(1,2,0).mul(255).round().byte().numpy()
            arr.flush()
        dump(f'parts/source_{tag(s)}_current.json',dict(status='PASS',source_tensor_sha256=digest,path=str(path),num_frames=64))
    return frames
