"""Independent UVG adaptation IO and immutable historical dependencies."""
import csv,hashlib,importlib.util,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
V62=ROOT.parent/'gvcrt_neural_wrapper_v6_2_objective_calibration'
BASE=ROOT.parent/'gvcrt_neural_wrapper_v6_2b_final_generalization_test'
V4=ROOT.parent/'gvcrt_neural_wrapper_v4_progressive_joint'
ENGINE=ROOT.parent/'gvcrt_parallel_v5a3_v60'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
PLOT_PYTHON='/data1/anaconda3_new/anaconda_program/bin/python'
EXPECTED='fe63a6f00ca6bae59b23c8683146dbe64b6be67c1463a552af851bbfc31e9cba'
STEPS=(0,250,500,1000,2000,3000,5000)
SPLITS={'train_fit':['Beauty','ReadySteadyGo','ShakeNDry','YachtRide'],'validation':['Bosphorus'],'holdout':['HoneyBee','Jockey']}
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT.parent/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies'))
def name(step):return f'mixed_{step}'
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
    assert rows;p=Path(p);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with t.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows([{k:json.dumps(v,allow_nan=False) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()} for r in rows])
    t.replace(p)
def read(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))
def module(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def command(argv):
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=list(map(str,argv)),cwd=str(REPO),unix=time.time()))+'\n')
def tensor_hash(state):
    h=hashlib.sha256()
    for k,t in state.items():h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def inventory():
    return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for folder in (V62,BASE,V4,ENGINE,REPO/'checkpoints') for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.lock','.pid')}
def frozen():
    for p,h in load(ROOT/'audits/dependencies.json').items():assert sha(p)==h,('dependency changed',p)
    for p,h in load(ROOT/'audits/local_hashes.json').items():assert sha(ROOT/p)==h,('local protocol changed',p)
    cfg=load(ROOT/'config.json');cfg.update(load(ROOT/'evaluation/config.json'));return cfg
def cp_path(step):return ROOT/'checkpoints'/f'adaptation_step_{step:04d}.pt'
def atomic_torch(p,obj):
    import torch
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');torch.save(obj,t);t.replace(p)

V614=ROOT.parent/'gvcrt_neural_wrapper_v6_14_uvg_target_domain_finetune'
V614B=ROOT.parent/'gvcrt_neural_wrapper_v6_14b_cross_domain_retention'
V61=ROOT.parent/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09'
METHODS=('original','v62_initial','uvg_adapt_1000','uvg_adapt_5000','mixed_1000','mixed_5000')
DATASETS=('ulong','uvg_holdout','hevc_b')
METRICS=('LPIPS','DISTS','FloLPIPS','FID','PSNR','SSIM','MS_SSIM')
def sources(d):return load(ROOT/'manifests'/f'{d}.json')['videos']
def point(d,m,i,q):return ROOT/'parts'/d/m/f'video_{i:02d}_qp{q}.json'
def evalcfg():return load(ROOT/'evaluation/config.json')
