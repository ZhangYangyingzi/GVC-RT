from pathlib import Path
import os,sys,json,hashlib,time,random,fcntl,contextlib,csv,subprocess,shutil
ROOT=Path(__file__).resolve().parent
PROJECT=Path('/Huang_group/zyyz/Projects/GVC-RT')
PREV=PROJECT/'GVC-RT_worktrees/bridge_aligned_latent_pilot_20261112/experiments/gvcrt_bridge_aligned_latent_pilot_20261112'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'vendor'))
for n,d in dict(TMPDIR='temporary',CUDA_CACHE_PATH='cache/cuda',CUPY_CACHE_DIR='cache/cupy',MPLCONFIGDIR='cache/mpl',TORCH_EXTENSIONS_DIR='cache/extensions',TRITON_CACHE_DIR='cache/triton',NUMBA_CACHE_DIR='cache/numba',XDG_CACHE_HOME='cache/xdg').items():
 p=ROOT/d;p.mkdir(parents=True,exist_ok=True);os.environ[n]=str(p)
os.environ.update(TZ='Asia/Shanghai',PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TORCH_HOME='/Huang_group/zyyz/home_dir/.cache/torch');time.tzset()
def stamp():return time.strftime('%Y-%m-%d %H:%M:%S %Z')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);assert p.resolve().is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.part')
 with t.open('w') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def disk():assert shutil.disk_usage(ROOT).free>3*2**30,'DISK_RESERVE_3GiB'
def save(p,x):
 import torch
 disk();p=Path(p);assert p.resolve().is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.part')
 with t.open('wb') as f:torch.save(x,f);f.flush();os.fsync(f.fileno())
 t.replace(p);dump(p.with_suffix(p.suffix+'.json'),dict(sha256=sha(p),bytes=p.stat().st_size,time=stamp()))
_verified=set()
def verify_file(p):
 p=Path(p);r=load(p.with_suffix(p.suffix+'.json'));s=p.stat();key=(str(p),s.st_size,s.st_mtime_ns,r['sha256']);assert s.st_size==r['bytes']
 if key not in _verified:assert sha(p)==r['sha256'],('CACHE_SHA_MISMATCH',str(p));_verified.add(key)
 return r

def readtensor(p):
 import torch
 verify_file(p);return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
def npzsave(p,arrays,meta):
 import numpy as np
 disk();p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.part')
 meta=dict(meta);meta['lossless_byte_shuffle']={k:dict(shape=list(a.shape),dtype=a.dtype.str) for k,a in arrays.items()}
 shuffled={k:np.moveaxis(np.ascontiguousarray(a).view(np.uint8).reshape(*a.shape,a.dtype.itemsize),-1,0).copy() for k,a in arrays.items()}
 with t.open('wb') as f:np.savez_compressed(f,**shuffled);f.flush();os.fsync(f.fileno())
 t.replace(p);dump(p.with_suffix('.npz.json'),dict(meta=meta,sha256=sha(p),bytes=p.stat().st_size,time=stamp()))
def npzload(p):
 import numpy as np
 verify_file(p)
 with np.load(p,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
 encoding=load(Path(p).with_suffix('.npz.json'))['meta'].get('lossless_byte_shuffle',{})
 return {k:np.moveaxis(a,0,-1).copy().reshape(-1).view(np.dtype(encoding[k]['dtype'])).reshape(encoding[k]['shape']) if k in encoding else a for k,a in arrays.items()}
def thash(items):
 h=hashlib.sha256()
 for n,t in items:h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def event(name,**kw):
 r=dict(time=time.time(),local_time=stamp(),event=name,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),**kw)
 with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
 print(json.dumps(r,allow_nan=False),flush=True)
def heartbeat(task,stage,**kw):
 p=ROOT/'receipts'/f'{task}.status.json';old=load(p) if p.exists() else {};old.update(task=task,stage=stage,pid=os.getpid(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),heartbeat=time.time(),local_time=stamp());old.update(kw);dump(p,old)
@contextlib.contextmanager
def lock(name,blocking=False):
 p=ROOT/'locks'/name;p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB));f.seek(0);f.truncate();f.write(str(os.getpid()));f.flush();yield

def setup():
 import torch
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['0','1','2','3'];torch.cuda.set_device(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True)
 def no_download(*a,**k):raise RuntimeError('NEW_DOWNLOAD_FORBIDDEN')
 torch.hub.download_url_to_file=no_download;return torch

def seed(n=20261113):
 import torch,numpy as np
 random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def rng():
 import torch,numpy as np
 n=np.random.get_state();return dict(python=random.getstate(),numpy=(n[0],torch.tensor(n[1].astype('int64')),*n[2:]),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def setrng(s):
 import torch,numpy as np
 random.setstate(s['python']);n=s['numpy'];np.random.set_state((n[0],n[1].numpy().astype('uint32'),*n[2:]));torch.set_rng_state(s['torch']);torch.cuda.set_rng_state(s['cuda'])
def csvwrite(p,rows):
 if not rows:return
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.part');keys=list(dict.fromkeys(k for r in rows for k in r))
 with t.open('w',newline='') as f:w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)
 t.replace(p)
def check_protocol():
 p=load(ROOT/'PROTOCOL.json');assert sha(ROOT/'training_plan.json')==p['train_plan_sha256'] and sha(ROOT/'data_manifest.json')==p['train_manifest_sha256'] and sha(ROOT/'development.json')==p['development_sha256'];return sha(ROOT/'PROTOCOL.json')
def plan():return load(ROOT/'training_plan.json')
def training_rows():return load(ROOT/'data_manifest.json')['train']
def devrows():return load(ROOT/'development.json')
def oldtrain(id):return readtensor(PREV/'cache/train'/f'{id}.pt')
def sidepath(id):return ROOT/'cache/decoded_refs'/f'{id}.npz'
def sidewait(id,task):
 p=sidepath(id)
 while not p.with_suffix('.npz.json').exists():heartbeat(task,'WAITING_FOR_CACHE',waiting_for=str(p));time.sleep(2)
 return npzload(p)
def oldstats():return readtensor(PREV/'cache/statistics.pt')
def stats():
 s=oldstats();s['evidence_rms']=readtensor(ROOT/'cache/evidence_stats.pt')['evidence_rms'];return s
