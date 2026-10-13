"""Isolated v6.31 IO, immutable provenance and per-stage receipts."""
import csv,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
V29=ROOT.parent/'gvcrt_neural_wrapper_v6_29_fid_audit_feature_covariance_control'
V30=ROOT.parent/'gvcrt_neural_wrapper_v6_30_fid_cache_closure_sample_robustness_audit'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
QPS=(0,5,9)
CELLS=('R00','R10','R01','R11','R11_center')
METHODS=('original','B1_initial')
LAM=.038636700918173825
REC=.1383589248954108
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+f'.{os.getpid()}.{time.time_ns()}.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def write(p,rows,fields=None):
 p=Path(p);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
 fields=fields or list(dict.fromkeys(k for r in rows for k in r));assert fields
 tmp=p.with_name(p.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
 tmp.replace(p)
def read(p):return list(csv.DictReader(Path(p).open()))
def textfile(p,s):
 p=Path(p);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def legacy_io():
 # Definitions are reused with all writable paths rebound before calling anything.
 m=module('v631_legacy_io',V29/'io21.py');m.ROOT=ROOT
 m.evalcfg=lambda:load(ROOT/'evaluation_config.json')
 m.dump=dump;m.write=write;m.frozen=lambda:load(ROOT/'evaluation_config.json')
 return m
def legacy_code(name,filename,overrides=None):
 ns=dict(legacy_io().__dict__);ns.update(overrides or {})
 src=(V29/filename).read_text().replace('from io21 import *','')
 exec(compile(src,str(V29/filename),'exec'),ns)
 import types
 return types.SimpleNamespace(**ns)
def tensor_hash(s):
 h=hashlib.sha256()
 for k,t in s.items():h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def video(d,i):return next(v for v in load(ROOT/'eval_manifest.json')[d] if v['video_index']==i)
def oldpoint(d,m,i,q):return V29/'parts/ip_-1'/d/m/f'video_{i:02d}_qp{q}.json'
def uid(t):return '_'.join(str(t[k]) for k in ('stage','dataset','video','method','qp'))
def receipt(t):return ROOT/'tasks'/(uid(t)+'.json')
def codefiles(stage):
 shared=['audit31.py','adapter31.py','color31.py']
 return shared+(['color_worker.py'] if stage=='color' else ['intervention_worker.py'] if stage=='intervention' else ['probe_worker.py'])
def signature(t):
 content=dict(task=t,config=sha(ROOT/'config.json'),protocol=sha(ROOT/'protocol.json'),sources=sha(ROOT/'source_hashes.json'),code={n:sha(ROOT/n) for n in codefiles(t['stage'])})
 return hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
def valid(t):
 try:
  r=load(receipt(t));return r['status']=='PASS' and r['signature']==signature(t) and all(Path(p).is_file() and sha(p)==h for p,h in r['artifacts'].items())
 except (OSError,KeyError,ValueError):return False
def error(x):
 x.update(time=time.time())
 with (ROOT/'errors.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
def save_npz(p,**x):
 import numpy as np
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with tmp.open('wb') as f:np.savez(f,**x)
 tmp.replace(p)
def tasklist():return load(ROOT/'task_manifest.json')['tasks']
