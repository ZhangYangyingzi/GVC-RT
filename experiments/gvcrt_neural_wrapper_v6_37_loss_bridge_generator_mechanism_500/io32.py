"""v6.32 isolated output, immutable dependencies and transaction IO."""
import csv,hashlib,importlib.util,json,os,sys,time,types
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
OLD32=ROOT.parent/'gvcrt_neural_wrapper_v6_32_learning_rate_fd_coral_control'
V29=ROOT.parent/'gvcrt_neural_wrapper_v6_29_fid_audit_feature_covariance_control'
V30=ROOT.parent/'gvcrt_neural_wrapper_v6_30_fid_cache_closure_sample_robustness_audit'
V31=ROOT.parent/'gvcrt_neural_wrapper_v6_31_p_color_loss_effective_update_audit'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
PREV=ROOT.parent/'gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion'
BRANCHES=('N_noP_BG_FD','P_PBG_FD')
REC=.1383589248954108
STAT=.038636700918173825
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
  for r in rows:w.writerow({k:json.dumps(v,allow_nan=False) if isinstance(v,(list,tuple,dict)) else v for k,v in r.items()})
 tmp.replace(p)
def read(p):return list(csv.DictReader(Path(p).open()))
def textfile(p,s):
 p=Path(p);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def append(p,x):
 p=Path(p);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def error(x):
 append(ROOT/'errors.jsonl',x)
def module(n,p):
 # Historical helper modules import sibling files (for example fullqp_common
 # imports v61_io). Make that dependency resolution explicit in this process.
 sys.path.insert(0,str(Path(p).resolve().parent))
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m
def tensor_hash(s):
 h=hashlib.sha256()
 for k,t in s.items():h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def atomic_torch(p,x):
 import torch
 p=Path(p);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
 torch.save(x,t)
 with t.open('rb') as f:os.fsync(f.fileno())
 t.replace(p)
def save_npz(p,**x):
 import numpy as np
 p=Path(p);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with t.open('wb') as f:np.savez(f,**x)
 t.replace(p)
def legacy_io():
 m=module('v632_legacy_io',V29/'io21.py');m.ROOT=ROOT;m.dump=dump;m.write=write;m.atomic_torch=atomic_torch
 m.evalcfg=lambda:load(ROOT/'evaluation_config.json');m.frozen=lambda:load(ROOT/'evaluation_config.json')
 return m
def legacy_code(filename,overrides=None,origin=V29):
 ns=dict(legacy_io().__dict__);ns.update(overrides or {})
 src=(origin/filename).read_text().replace('from io21 import *','').replace('from audit31 import *','')
 exec(compile(src,str(origin/filename),'exec'),ns);return types.SimpleNamespace(**ns)
def code_signature(names):return {n:sha(ROOT/n) for n in names}
_SIGNATURES={}
def signature(kind):
 if kind in _SIGNATURES:return _SIGNATURES[kind]
 names=['io32.py','training32.py','frames32.py','train.py']
 value=hashlib.sha256(json.dumps(dict(config=sha(ROOT/'config.json'),protocol=sha(ROOT/'protocol.json'),sources=sha(ROOT/'source_hashes.json'),plan=sha(ROOT/'plans/shared.json'),code=code_signature(names)),sort_keys=True).encode()).hexdigest();_SIGNATURES[kind]=value;return value
def source_row(item):return item['source_row']

V36=ROOT.parent/"gvcrt_neural_wrapper_v6_36_noP_basic_vs_full_loss_matched_500"
METHODS=("O_original","B_basicLoss_noP_500","D_basicDISTS_noP_500","F_fullLoss_noP_500","X_fullBridge_originalG","Y_originalBridge_fullG","I_halfLatent_fullG")
QPS=(0,5,9)
