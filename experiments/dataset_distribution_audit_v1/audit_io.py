"""Independent, read-only dataset analysis; writes confined to this directory."""
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
VIMEO=Path('/Huang_group/zyyz/datasets/Vimeo90K')
V61=ROOT.parent/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09'
V62=ROOT.parent/'gvcrt_neural_wrapper_v6_2_objective_calibration'
V62B=ROOT.parent/'gvcrt_neural_wrapper_v6_2b_final_generalization_test'
V4=ROOT.parent/'gvcrt_neural_wrapper_v4_progressive_joint'
V52=ROOT.parent/'gvcrt_neural_wrapper_v5a_2_fair_30fps_rgb'
PUBLIC=ROOT.parent/'gvcrt_public_virat_720p_480p_qp01'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
SEED=20261001
sys.path.insert(0,str(V61/'runtime_dependencies'))
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def dump(path,obj):
    p=Path(path);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def write(path,rows):
    assert rows;path=Path(path);assert path.is_relative_to(ROOT);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix(path.suffix+'.tmp')
    with t.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    t.replace(path)
def read(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
