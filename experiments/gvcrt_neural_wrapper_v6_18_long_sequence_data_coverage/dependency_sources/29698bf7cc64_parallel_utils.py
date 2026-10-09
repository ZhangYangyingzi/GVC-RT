"""Isolated parallel experiment bookkeeping (distinct from codec common module)."""
import csv,hashlib,importlib.util,json,os,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
A=ROOT.parent/'gvcrt_neural_wrapper_v5a_3_virat_480p20_clip8'
B=ROOT.parent/'gvcrt_neural_wrapper_v6_0_cross_domain_attribution'
V52=ROOT.parent/'gvcrt_neural_wrapper_v5a_2_fair_30fps_rgb'
V41=ROOT.parent/'gvcrt_neural_wrapper_v4_1_high_convergence_metrics'
V5=ROOT.parent/'gvcrt_neural_wrapper_v5a_clip_ablation'
PUBLIC=ROOT.parent/'gvcrt_public_virat_720p_480p_qp01'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
PLOT_PYTHON='/data1/anaconda3_new/anaconda_program/bin/python'
METRICS=('LPIPS','DISTS','FloLPIPS','FID')
SPATIAL=('LPIPS','DISTS','PSNR','SSIM','MS_SSIM')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(p)
def write(p,rows):
    assert rows
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    temp.replace(p)
def read(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
def exp(which):return A if which=='A' else B
def point(root,dataset,method,i,q):return root/'parts'/dataset/method/f'video_{i}_qp{q}.json'
def raw_mp4(video):
    import numpy as np
    assert str(video['path']).endswith('__480p_854x480_20fps_original.mp4')
    command=['ffmpeg','-v','error','-threads','1','-i',video['path'],'-map','0:v:0','-vsync','0','-threads','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    with tempfile.TemporaryFile() as errors:
        p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=errors)
        try:
            count=0
            while True:
                raw=p.stdout.read(854*480*3)
                if not raw:break
                assert len(raw)==854*480*3,'partial source frame'
                count+=1
                yield np.frombuffer(raw,np.uint8).reshape(480,854,3).copy()
            assert p.wait()==0
            assert count==video['frames'],(video['video'],count,video['frames'])
        finally:
            p.stdout.close()
            if p.poll() is None:p.terminate();p.wait()
            errors.seek(0)
            if p.returncode:raise RuntimeError(errors.read().decode())
def source_arrays(video):
    if video['dataset']=='virat':yield from raw_mp4(video)
    else:
        loader=module('canonical_v52',V52/'canonical_loader.py')
        yield from loader.canonical_arrays(video['canonical_dir'])
def frames_for(video):
    import torch
    arrays=list(source_arrays(video));h=hashlib.sha256()
    for a in arrays:h.update(a.tobytes())
    assert h.hexdigest()==video['rgb_sha256'],video['name']
    return [torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/255 for a in arrays]
def tensor_dict_hash(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def verify_frozen(config):
    for p,h in config['frozen_hashes'].items():assert sha(p)==h,p
