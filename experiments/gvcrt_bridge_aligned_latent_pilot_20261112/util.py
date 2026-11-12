from pathlib import Path
import os,sys,json,hashlib,time,random,fcntl,contextlib,csv,subprocess
ROOT=Path(__file__).resolve().parent
PROJECT=Path('/Huang_group/zyyz/Projects/GVC-RT')
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'vendor'))
for n,d in dict(TMPDIR='tmp',CUDA_CACHE_PATH='cache/cuda',CUPY_CACHE_DIR='cache/cupy',MPLCONFIGDIR='cache/mpl',TORCH_EXTENSIONS_DIR='cache/extensions',TRITON_CACHE_DIR='cache/triton',NUMBA_CACHE_DIR='cache/numba',XDG_CACHE_HOME='cache/xdg').items():
 p=ROOT/d;p.mkdir(parents=True,exist_ok=True);os.environ[n]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TORCH_HOME='/Huang_group/zyyz/home_dir/.cache/torch')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);assert p.resolve().is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with t.open('w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def save(p,x):
 import torch,shutil
 assert shutil.disk_usage(ROOT).free>8*2**30,'DISK_RESERVE_8GiB'
 p=Path(p);assert p.resolve().is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with t.open('wb') as f:torch.save(x,f);f.flush();os.fsync(f.fileno())
 t.replace(p)
 dump(p.with_suffix(p.suffix+'.json'),dict(sha256=sha(p),bytes=p.stat().st_size,time=time.time()))
def readtensor(p):
 import torch
 p=Path(p);r=load(p.with_suffix(p.suffix+'.json'));assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],('INVALID_COMMIT',str(p));return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
def thash(items):
 h=hashlib.sha256()
 for n,t in items:h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def event(name,**kw):
 r=dict(time=time.time(),event=name,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),**kw)
 with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
 print(json.dumps(r,allow_nan=False),flush=True)
def heartbeat(task,stage,**kw):dump(ROOT/'receipts'/f'{task}.status.json',dict(task=task,stage=stage,pid=os.getpid(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),heartbeat=time.time(),**kw))
@contextlib.contextmanager
def lock(n):
 p=ROOT/'locks'/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);f.seek(0);f.truncate();f.write(str(os.getpid()));f.flush();yield

def setup():
 import torch
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['0','1','2','3'],'PHYSICAL_GPU_RESTRICTION'
 torch.cuda.set_device(0);torch.set_num_threads(2)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True,warn_only=True)
 def forbidden(*a,**k):raise RuntimeError('DOWNLOAD_FORBIDDEN')
 torch.hub.download_url_to_file=forbidden
 return torch

def seed(n=20261112):
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
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');fields=list(dict.fromkeys(k for r in rows for k in r))
 with t.open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
 t.replace(p)
def cachepath(id,qp=None):
 if id.startswith('tr_'):return ROOT/'cache/train'/f'{id}.pt'
 return ROOT/'cache/dev'/f'{id}_q{qp}.pt'
def waitcache(id,task,qp=None):
 p=cachepath(id,qp)
 while not p.with_suffix('.pt.json').exists():
  heartbeat(task,'WAITING_FOR_CACHE',waiting_for=str(p));time.sleep(2)
 return readtensor(p)
def manifest():return load(ROOT/'data_manifest.json')
def plan():return load(ROOT/'training_plan.json')
def check_protocol():
 p=load(ROOT/'protocol.json');assert sha(ROOT/'training_plan.json')==p['plan_sha256'] and sha(ROOT/'data_manifest.json')==p['data_sha256'];return sha(ROOT/'protocol.json')
