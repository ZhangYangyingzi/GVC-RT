from pathlib import Path
import os,sys,json,hashlib,time,random,fcntl,contextlib,csv,shutil
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OLD=PROJECT/'experiments/gvcrt_history_detail_st_pilot_20261112'
SCRATCH=ROOT/'cache'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'vendor'))
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TORCH_HOME='/Huang_group/zyyz/home_dir/.cache/torch')
for key,rel in dict(TMPDIR='tmp',CUDA_CACHE_PATH='cache/cuda',CUPY_CACHE_DIR='cache/cupy',MPLCONFIGDIR='cache/mpl',TORCH_EXTENSIONS_DIR='cache/torch_extensions',TRITON_CACHE_DIR='cache/triton',NUMBA_CACHE_DIR='cache/numba',XDG_CACHE_HOME='cache/xdg').items():
    p=ROOT/rel;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,x):
    p=Path(p);assert p.is_relative_to(ROOT) or p.is_relative_to(SCRATCH)
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with t.open('w') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    t.replace(p)
def save(p,x):
    import torch
    p=Path(p);assert p.is_relative_to(ROOT) or p.is_relative_to(SCRATCH)
    p.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(p.parent).free>4*2**30,('DISK_RESERVE_BELOW_4GiB',str(p))
    t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with t.open('wb') as f:torch.save(x,f);f.flush();os.fsync(f.fileno())
    t.replace(p)
def thash(items):
    h=hashlib.sha256()
    for n,t in items:h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
@contextlib.contextmanager
def lock(name,blocking=False):
    p=ROOT/'locks'/name;p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB));f.seek(0);f.truncate();f.write(str(os.getpid()));f.flush();yield
def event(path,**kw):
    row=dict(time=time.time(),**kw);path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
    print(json.dumps(row,allow_nan=False),flush=True)
def videos(split=None):return [r for r in load(ROOT/'data_manifest.json')['videos'] if split is None or r['split']==split]
def setup_torch():
    import torch
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    def no_download(*a,**kw):raise RuntimeError('NETWORK_DOWNLOAD_FORBIDDEN')
    torch.hub.download_url_to_file=no_download
    return torch
def seed(s):
    import torch,numpy as np
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def rng_state():
    import torch,numpy as np
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=(n[0],torch.tensor(n[1].astype('int64')),*n[2:]),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def restore_rng(s):
    import torch,numpy as np
    random.setstate(s['python']);n=s['numpy'];np.random.set_state((n[0],n[1].numpy().astype('uint32'),*n[2:]));torch.set_rng_state(s['torch']);torch.cuda.set_rng_state(s['cuda'])
def csv_write(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix('.tmp');fields=list(dict.fromkeys(k for r in rows for k in r))
    with t.open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    t.replace(path)
def device_info():
    import torch
    return dict(pid=os.getpid(),physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),name=torch.cuda.get_device_name(0),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,uuid=str(torch.cuda.get_device_properties(0).uuid))
