"""Fixed protocol, atomic receipts and immutable historical dependencies."""
import csv,hashlib,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parent
ORIGINAL=Path('/Huang_group/zyyz/Projects/GVC-RT')
V2=ORIGINAL/'experiments/gvcrt_generation_aware_quant_v2_skip_domain_trend_300'
V1=ORIGINAL/'experiments/gvcrt_generation_aware_quant_visual_condition_v1_trend_300'
TEMPLATE=ORIGINAL/'experiments/gvcrt_generation_aware_quant_v3_rate_spatial_ablation_eval'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
MODES=('A','IDENTITY','FRAME','CHANNEL','SPATIAL')
SOURCES=('UVal0','UVal1','Jockey','ParkScene')
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);assert p.is_relative_to(P);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def table(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()} for r in rows)
 tmp.replace(p)
def signature():return hashlib.sha256(json.dumps({n:sha(P/n) for n in ('util.py','ablation.py','task.py','effective_config.json','evaluation_sources.json','dependency_manifest.json')},sort_keys=True).encode()).hexdigest()
def key(mode,source,qp):return f'{mode}_s300_{source}_q{qp}'
def checkpoint(mode):
 b='A' if mode=='A' else 'H11';return (V1 if b=='A' else V2)/'server_artifacts'/b/'step_300.pt'
def setup(gpu):
 assert gpu in (4,5,6,7)
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 sys.path.insert(0,str(V2))
 import train
 train.setup_path()
 sys.path.insert(0,str(P))
 assert Path(train.__file__).resolve()==V2/'train.py'
 import torch
 torch.set_num_threads(2)
 return train,torch
