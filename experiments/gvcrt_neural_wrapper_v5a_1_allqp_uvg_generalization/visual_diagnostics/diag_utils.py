import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
REPO=PARENT.parents[1]
sys.path.insert(0,str(PARENT))
import audit_utils as old
METHODS=('original','v41_20000','clip8')
QPS=(0,3,6,9)
LABELS={'original':'Original GVC-RT','v41_20000':'V4.1','clip4_control':'clip4','clip8':'clip8'}
PYTHON=old.PYTHON
sha=old.sha
module=old.module
def load(p):return json.loads((ROOT/p).read_text())
def dump(p,obj):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def read(p):
    with (ROOT/p).open(newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    assert rows
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    t.replace(p)
def samples():return load('sample_manifest.json')['samples']
def tag(s):return s['dataset']+'_'+str(s['video_index'])
def rawpath(s,name,q=None,smoke=False):
    return ROOT/('bt709_smoke/raw' if smoke else 'raw')/tag(s)/(name+(f'_qp{q}' if q is not None else '')+'.npy')
def point(s,m,q):return old.load(old.point_path(s['dataset'],m,s['video_index'],q))
def cp(m):return old.checkpoint(m)
def count(s):return 96 if s['dataset']=='uvg' else 64
def keyframes(s):return (0,24,48,95) if s['dataset']=='uvg' else (0,16,32,63)
