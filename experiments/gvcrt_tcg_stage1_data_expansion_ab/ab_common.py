"""Task-local atomic records and immutable provenance. No historical mutations."""
import contextlib,fcntl,hashlib,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
ORIGINAL=Path('/Huang_group/zyyz/Projects/GVC-RT')
P=Path('/Huang_group/zyyz/home_dir/tmp/gvcrt_tcg_stage1_expanded_20261103_worktree/experiments/gvcrt_tcg_stage1_expanded_5000')
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
ARCHIVES=Path('/Huang_group/zyyz/datasets/UltraVideo-Long/clips_long_1920')
BASELINE='60526a0f7ea69954c91b4e328de77ea50bedb55e'
PARENT_SHA='d398c3760ef1351ce21dc1663cc20337b0ce98dfaf5b06bf8b1d550511210264'
SEED=20261103
PREFIXES=('tcg_low.','tcg_mid.','tcg_out.')
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p);fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def disk():
 d=shutil.disk_usage(ROOT)
 if d.free<100*2**30:raise RuntimeError('DISK_RESERVE_BELOW_100_GiB')
 return dict(free_bytes=d.free,total_bytes=d.total)
def event(name,**kw):
 r=dict(event=name,time=time.time(),**kw)
 with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
 print(json.dumps(r,allow_nan=False),flush=True)
@contextlib.contextmanager
def lock(name):
 with (ROOT/name).open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield
def status(stage,state='RUNNING',**kw):dump(ROOT/'run_status.json',dict(status=state,stage=stage,updated=time.time(),**kw))
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
def atomic_torch(p,x):
 import torch
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
 with t.open('wb') as f:torch.save(x,f);f.flush();os.fsync(f.fileno())
 t.replace(p);fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
