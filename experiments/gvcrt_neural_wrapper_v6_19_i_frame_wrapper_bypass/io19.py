"""Isolated IO for DISTS verification and the conditional fixed repair."""
import csv,hashlib,importlib.util,inspect,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
V18=ROOT.parent/'gvcrt_neural_wrapper_v6_18_long_sequence_data_coverage'
V17=ROOT.parent/'gvcrt_neural_wrapper_v6_17_dists_weight_control'
V16=ROOT.parent/'gvcrt_neural_wrapper_v6_16_dists_gradient_verification'
V15=ROOT.parent/'gvcrt_neural_wrapper_v6_15_mixed_domain_replay'
V62=ROOT.parent/'gvcrt_neural_wrapper_v6_2_objective_calibration'
V4=ROOT.parent/'gvcrt_neural_wrapper_v4_progressive_joint'
BASE=ROOT.parent/'gvcrt_neural_wrapper_v6_2b_final_generalization_test'
ENGINE=ROOT.parent/'gvcrt_parallel_v5a3_v60'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
PLOT_PYTHON='/data1/anaconda3_new/anaconda_program/bin/python'
EXPECTED='3a9279f503346f58658a0792a02399a59e63b1542127c868f0a3401f56ba1629'
METHODS=('original','B_standard','B_i_bypass')
DATASETS=('ulong','uvg_holdout','hevc_b','uvg_validation')
METRICS=('LPIPS','DISTS','FloLPIPS','FID','PSNR','SSIM','MS_SSIM')
def evalcfg():return load(ROOT/'evaluation/config.json')
sys.dont_write_bytecode=True
sys.path.insert(0,str(V15.parent/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,v):
    p=Path(p);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def write(p,rows):
    p=Path(p);assert p.is_relative_to(ROOT) and rows;p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with t.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows([{k:json.dumps(v,allow_nan=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()} for r in rows])
    t.replace(p)
def read(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))
def module(n,p):
    sys.path.insert(0,str(Path(p).parent))
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    sys.path.insert(0,str(ROOT));return m
def tensor_hash(s):
    h=hashlib.sha256()
    for k,t in s.items():h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def command(argv):
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=list(map(str,argv)),cwd=str(REPO),unix=time.time()))+'\n')
def atomic_torch(p,s):
    import torch
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');torch.save(s,t);t.replace(p)
def sources(d):return load(ROOT/'manifests'/f'{d}.json')['videos']
def point(d,m,i,q):return ROOT/'parts'/d/m/f'video_{i:02d}_qp{q}.json'
def frozen():
    for p,h in load(ROOT/'audits/dependencies.json').items():assert sha(p)==h,('dependency changed',p)
    for p,h in load(ROOT/'audits/protocol_hashes.json').items():assert sha(ROOT/p)==h,('protocol changed',p)
    return load(ROOT/'evaluation/config.json')
def cp_path(step):return ROOT/'checkpoints'/f'adaptation_step_{step:04d}.pt'
def historical_inventory():return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for folder in (V18,V17,V16,V15,V62,V4,BASE,ENGINE,REPO/'checkpoints') for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.lock','.pid')}
