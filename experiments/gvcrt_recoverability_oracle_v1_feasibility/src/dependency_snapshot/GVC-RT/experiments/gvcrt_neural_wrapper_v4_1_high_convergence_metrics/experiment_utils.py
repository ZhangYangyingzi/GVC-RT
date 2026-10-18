import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V4 = ROOT.parent / 'gvcrt_neural_wrapper_v4_progressive_joint'
V3 = ROOT.parent / 'gvcrt_neural_wrapper_v3_pb_scale'
PYTHON = '/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def dump(path, obj):
    path = ROOT/path; path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def read(path):
    with open(ROOT/path, newline='') as f: return list(csv.DictReader(f))

def write(path, rows):
    path = ROOT/path; path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with open(tmp,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)
