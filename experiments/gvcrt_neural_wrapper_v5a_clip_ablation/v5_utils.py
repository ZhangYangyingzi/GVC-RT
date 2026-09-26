import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
V41=ROOT.parent/'gvcrt_neural_wrapper_v4_1_high_convergence_metrics'
V4=ROOT.parent/'gvcrt_neural_wrapper_v4_progressive_joint'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
BRANCHES={'clip4_control':{'clip_length':4,'updates':7000},'clip8':{'clip_length':8,'updates':3000}}
METHODS=('original','v41_20000','clip4_control','clip8')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def load(path): return json.loads((ROOT/path).read_text())
def dump(path,value):
    path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
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
    if method=='original':return None
    if method=='v41_20000':return Path(load('config.json')['source_checkpoint'])
    return ROOT/'checkpoints'/method/f'step_{BRANCHES[method]["updates"]:04d}.pt'
def truth(value):return str(value).lower()=='true'
