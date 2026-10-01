"""V6.3 namespaced bookkeeping and read-only reuse of frozen implementations."""
import csv, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
V62=ROOT.parent/'gvcrt_neural_wrapper_v6_2_objective_calibration'
V62B=ROOT.parent/'gvcrt_neural_wrapper_v6_2b_final_generalization_test'
V61=ROOT.parent/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09'
V4=ROOT.parent/'gvcrt_neural_wrapper_v4_progressive_joint'
AUDIT=ROOT.parent/'dataset_distribution_audit_v1'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
BRANCHES=('ulong_only','vimeo_only','mixed_50_50')
METHODS=('original','v62',*BRANCHES)
DATASETS=('ulong','uvg','virat720','virat480')
METRICS=('LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM')
STEPS=(0,250,500,1000)
sys.path.insert(0,str(V61/'runtime_dependencies'))
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,obj):
    p=Path(p);assert p.is_relative_to(ROOT),p;p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def write(p,rows):
    assert rows;p=Path(p);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
    with t.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    t.replace(p)
def read(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def checkpoint(branch,step):return ROOT/'branches'/branch/'checkpoints'/f'step_{step:04d}.pt'
def point(d,m,i,q):return ROOT/'parts'/d/m/f'video_{i:02d}_qp{q}.json'
def sources(d):return [v for v in load(ROOT/'source_manifest.json')['videos'] if v['dataset']==d]
def inventory():
    result={}
    folders=[p for p in ROOT.parent.iterdir() if p.is_dir() and p!=ROOT]
    for folder in folders:
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.lock':
                s=p.stat();result[str(p)]=[s.st_size,s.st_mtime_ns]
    return result
def frozen(check_inventory=False):
    for p,h in load(ROOT/'audits/frozen_dependencies.json').items():assert sha(p)==h,('changed dependency',p)
    if check_inventory:assert inventory()==load(ROOT/'audits/old_inventory.json'),'Old experiments modified'
    for p,h in load(ROOT/'audits/local_protocol_hashes.json').items():assert sha(ROOT/p)==h,('changed local protocol',p)
    return load(ROOT/'config.json')
def old_eval():
    sys.path.insert(0,str(V62B));return module('v63_original_evaluator',V62B/'evaluate.py')
def evaluation_config():return load(ROOT/'evaluation/config.json')
def eval_adapter():
    m=old_eval();m.ROOT=ROOT;m.METHODS=METHODS;m.sources=sources;m.point=point;m.dump=dump;m.write=write
    m.load=lambda p:evaluation_config() if Path(p)==ROOT/'config.json' else load(p)
    def check():
        frozen();cfg=evaluation_config()
        for cp in cfg['checkpoints'].values():assert sha(cp['path'])==cp['sha256']
        return cfg
    m.check_frozen=check;m.visualization=lambda v,q:None
    return m
def source_for(branch,step):return 'ulong' if branch=='ulong_only' or (branch=='mixed_50_50' and step%2==1) else 'vimeo'
