"""Local output paths and immutable historical evaluation dependencies."""
import csv,hashlib,importlib.util,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
V15=ROOT.parent/'gvcrt_neural_wrapper_v6_15_mixed_domain_replay'
V16=ROOT.parent/'gvcrt_neural_wrapper_v6_16_dists_gradient_verification'
V17=ROOT.parent/'gvcrt_neural_wrapper_v6_17_dists_weight_control'
V62=ROOT.parent/'gvcrt_neural_wrapper_v6_2_objective_calibration'
BASE=ROOT.parent/'gvcrt_neural_wrapper_v6_2b_final_generalization_test'
ENGINE=ROOT.parent/'gvcrt_parallel_v5a3_v60'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
PLOT_PYTHON='/data1/anaconda3_new/anaconda_program/bin/python'
METHODS=('original','v62_initial','mixed_1000','dists_fixed_1000','dists_w05_1000')
DATASETS=('ulong','uvg_holdout','uvg_validation','hevc_b')
QPS=(0,4,9);METRICS=('LPIPS','DISTS','FloLPIPS')
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
    sys.path.insert(0,str(Path(p).parent));s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def tensor_hash(s):
    h=hashlib.sha256()
    for k,t in s.items():h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def evalcfg():return load(ROOT/'config.json')
def sources(d):return load(ROOT/'manifests'/f'{d}.json')['videos']
def point(d,m,i,q):return ROOT/'evaluation/points'/d/m/f'video_{i:02d}_qp{q}.json'
def command(argv):
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=list(map(str,argv)),cwd=str(REPO),unix=time.time()))+'\n')
def frozen():
    for p,h in load(ROOT/'audits/dependencies.json').items():assert sha(p)==h,('dependency changed',p)
    for p,h in load(ROOT/'audits/local_protocol_hashes.json').items():assert sha(ROOT/p)==h,('protocol changed',p)
    return evalcfg()
def seal():
    ps=[*ROOT.glob('*.py'),ROOT/'config.json',ROOT/'protocol.json',ROOT/'qp_semantics_audit.json',*(ROOT/'manifests').glob('*.json')]
    dump(ROOT/'audits/local_protocol_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in ps})
def gpu_snapshot():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return [dict(zip(('index','total_MiB','used_MiB','free_MiB','utilization_percent'),map(int,line.split(',')))) for line in raw.splitlines() if int(line.split(',')[0]) in (4,5,6,7)]
