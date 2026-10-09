"""Isolated V5-A.1 evaluation bookkeeping; no training entry points."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
V5=ROOT.parent/'gvcrt_neural_wrapper_v5a_clip_ablation'
V41=ROOT.parent/'gvcrt_neural_wrapper_v4_1_high_convergence_metrics'
V4=ROOT.parent/'gvcrt_neural_wrapper_v4_progressive_joint'
UVG=Path('/Huang_group/zyyz/datasets/UVG')
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
METHODS=('original','v41_20000','clip8')
METRICS=('LPIPS','DISTS','FloLPIPS','FID')
NUMERIC=('kbps','bpp','LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def load(path):return json.loads((ROOT/path).read_text())
def dump(path,obj):
    path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)
def read(path):
    with (ROOT/path).open(newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
    assert rows,'empty table'
    path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    temp.replace(path)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def checkpoint(method):
    value=load('config.json')['checkpoints'][method]['path'];return Path(value) if value else None
def videos(dataset):
    if dataset=='ulong': return load('test_manifest.json')['videos']
    return [dict(v,frames_evaluated=64,fps=30.0) for v in load('uvg_available_manifest.json')['videos']]
def point_path(dataset,method,index,qp):return ROOT/'parts'/dataset/method/f'video_{index}_qp{qp}.json'
def truth(value):return str(value).lower()=='true'
