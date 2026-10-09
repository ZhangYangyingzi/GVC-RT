"""Evaluation-only V6.2-B bookkeeping; all writes stay in this experiment."""
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
V41=ROOT.parent/'gvcrt_neural_wrapper_v4_1_high_convergence_metrics'
V52=ROOT.parent/'gvcrt_neural_wrapper_v5a_2_fair_30fps_rgb'
V61=ROOT.parent/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09'
V62=ROOT.parent/'gvcrt_neural_wrapper_v6_2_objective_calibration'
PUBLIC=ROOT.parent/'gvcrt_public_virat_720p_480p_qp01'
ENGINE=ROOT.parent/'gvcrt_parallel_v5a3_v60'
FID_SOURCE=REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
PLOT_PYTHON='/data1/anaconda3_new/anaconda_program/bin/python'
sys.path.insert(0,str(V61/'runtime_dependencies'))
METHODS=('original','v41','v62')
DATASETS=('ulong','uvg','virat720','virat480')
METRICS=('LPIPS','DISTS','FloLPIPS','FID','PSNR','SSIM','MS_SSIM')
SPATIAL=('LPIPS','DISTS','PSNR','SSIM','MS_SSIM')
HIGHER={'PSNR','SSIM','MS_SSIM'}
PAIRS=(('original','v41'),('original','v62'),('v41','v62'))
EXPECTED={'v41':'c185e1f69fd5f249a4aa3753ddee1dc23d4402d5d6f74d18ef6b51b559fbdb73',
          'v62':'fe63a6f00ca6bae59b23c8683146dbe64b6be67c1463a552af851bbfc31e9cba'}
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def dump(path,value):
    p=Path(path);assert p.is_relative_to(ROOT),p;p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)
def write(path,rows):
    assert rows;path=Path(path);assert path.is_relative_to(ROOT);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    temp.replace(path)
def read(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def point(dataset,method,index,q):return ROOT/'parts'/dataset/method/f'video_{index:02d}_qp{q}.json'
def sources(dataset):return [v for v in load(ROOT/'source_manifest.json')['videos'] if v['dataset']==dataset]
def old_inventory():
    result={}
    folders=[p for p in ROOT.parent.glob('gvcrt_neural_wrapper_v*') if p!=ROOT and p.name.startswith(('gvcrt_neural_wrapper_v4','gvcrt_neural_wrapper_v5','gvcrt_neural_wrapper_v6_1','gvcrt_neural_wrapper_v6_2_'))]+[PUBLIC,ENGINE]
    for folder in folders:
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.lock':
                s=p.stat();result[str(p)]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns)
    return result
def check_frozen():
    cfg=load(ROOT/'config.json')
    for k,v in cfg['checkpoints'].items():assert sha(v['path'])==v['sha256']==EXPECTED[k],k
    for path,digest in load(ROOT/'frozen_source_hashes.json').items():assert sha(path)==digest,('changed dependency',path)
    for file,digest in cfg['audit_hashes'].items():assert sha(ROOT/file)==digest,('changed audit',file)
    assert cfg['no_training'] and cfg['no_model_selection'] and cfg['force_zero_thres']==.12
    return cfg
def arrays(v):
    import numpy as np
    from PIL import Image
    if v['dataset'] in ('ulong','uvg'):
        loader=module('v62b_canonical',V52/'canonical_loader.py');yield from loader.canonical_arrays(v['input_dir'])
    else:
        for i in range(v['frames']):
            p=Path(v['input_dir'])/f'im{i+1:05d}.png'
            with Image.open(p) as image:
                assert image.mode=='RGB' and image.size==(v['width'],v['height']);yield np.asarray(image,dtype=np.uint8).copy()
def frames_for(v):
    import torch
    output=[];h=hashlib.sha256()
    for i,a in enumerate(arrays(v)):
        assert hashlib.sha256(a.tobytes()).hexdigest()==v['frame_rgb_sha256'][i]
        h.update(a.tobytes());output.append(torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/255)
    assert len(output)==v['frames'] and h.hexdigest()==v['rgb_sha256']
    return output
