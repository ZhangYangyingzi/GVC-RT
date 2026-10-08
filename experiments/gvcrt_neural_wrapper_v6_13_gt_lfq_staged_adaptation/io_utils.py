"""Local experiment IO; no writes to historical experiments."""
import csv, hashlib, json, os, shlex, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
V612=ROOT.parent/'gvcrt_neural_wrapper_v6_12_rate_gradient_correction'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,obj):
    p=Path(p);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def write(p,rows,fields=None):
    p=Path(p);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with t.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(dict.fromkeys(k for r in rows for k in r)));w.writeheader()
        w.writerows([{k:json.dumps(v,allow_nan=False) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()} for r in rows])
    t.replace(p)
def command(argv):
    p=ROOT/'RUN_COMMANDS.md'
    with p.open('a') as f:
        f.write('\n```bash\ncd '+shlex.quote(str(REPO))+'\n'+shlex.join(list(map(str,argv)))+'\n```\n')
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=list(map(str,argv)),cwd=str(REPO),unix=time.time()))+'\n')
