import csv,hashlib,json,os,sys,time,fcntl
from pathlib import Path
P=Path(__file__).resolve().parent
ORIGINAL=Path('/Huang_group/zyyz/Projects/GVC-RT')
V2=ORIGINAL/'experiments/gvcrt_generation_aware_quant_v2_skip_domain_trend_300'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def dump(p,obj):
 p=Path(p);assert p.is_relative_to(P);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def write(p,rows):
 p=Path(p);assert p.is_relative_to(P);p.parent.mkdir(parents=True,exist_ok=True)
 fields=list(dict.fromkeys(k for r in rows for k in r));tmp=p.with_suffix('.csv.tmp')
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()} for r in rows)
 tmp.replace(p)
def append(p,x):
 p=Path(p);assert p.is_relative_to(P);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def state_hash(sd):
 h=hashlib.sha256()
 for n,t in sd.items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def model_hash(m):return state_hash(m.state_dict())
def gpu_env(gpu):
 assert gpu in (4,5,6,7);os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
def signature():
 return hashlib.sha256(json.dumps({n:sha(P/n) for n in ['train.py','runtime.py','model_io.py','util.py','effective_config.json','data_manifest.json','train_plan.json']},sort_keys=True).encode()).hexdigest()
def legacy_checkpoint(branch,step):
 e=load(V2/'checkpoint_index.json')[{'N0':'A','Q0':'H11'}[branch]][str(step)]
 path=Path(e['path']);assert path.is_file(),('missing indexed checkpoint',e)
 assert sha(path)==e['sha256'];return path
