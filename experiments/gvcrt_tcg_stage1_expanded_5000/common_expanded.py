import hashlib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
ORIGINAL=Path('/Huang_group/zyyz/Projects/GVC-RT')
POOL=ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000'
SOURCE=ORIGINAL/'experiments/gvcrt_tcg_stage1_pilot/checkpoints/adaptation_step_0.pt'
RAFT=Path('/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth')
SEED=20261103
PREFIXES=('tcg_low.','tcg_mid.','tcg_out.')
STEPS=(0,2,500,1000,2500,5000)
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w') as f:json.dump(x,f,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def event(name,**x):
 x=dict(event=name,time=time.time(),**x);p=ROOT/'logs/events.jsonl';p.parent.mkdir(exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(x,allow_nan=False),flush=True)
def atomic_torch(p,x):
 import torch
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
 with t.open('wb') as f:torch.save(x,f);f.flush();os.fsync(f.fileno())
 t.replace(p)
 fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def tensor_hash(items):
 h=hashlib.sha256()
 for n,t in items:h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
