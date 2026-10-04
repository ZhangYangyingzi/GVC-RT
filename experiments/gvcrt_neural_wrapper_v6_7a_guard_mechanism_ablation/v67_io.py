"""Isolated controlled-training protocol and read-only adapters."""
import csv,hashlib,importlib.util,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
V64=ROOT.parent/'gvcrt_neural_wrapper_v6_4_vimeo_pretrain'
V66=ROOT.parent/'gvcrt_neural_wrapper_v6_6_controlled_training'
V65=ROOT.parent/'gvcrt_neural_wrapper_v6_5_pbg_recoverability_attribution'
V62=ROOT.parent/'gvcrt_neural_wrapper_v6_2_objective_calibration'
V62B=ROOT.parent/'gvcrt_neural_wrapper_v6_2b_final_generalization_test'
V4=ROOT.parent/'gvcrt_neural_wrapper_v4_progressive_joint'
ENGINE=ROOT.parent/'gvcrt_parallel_v5a3_v60'
ACTIVE_OLD=ROOT.parent/'gvcrt_neural_wrapper_v6_5b_uvg_temporal_stride_recoverability'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
A='A_baseline';B='B_proxy_msssim';D='D_proxy_l1_matched';E='E_output_msssim_matched'
OLD_B='B_joint_structure_guard'
BRANCHES=(D,E);METHODS=('original',A,B,D,E);REPORT_METHODS=(A,B,D,E);ATTR_BRANCHES=(B,D,E)
DATASETS=('uvg','ulong','virat720','virat480');METRICS=('LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM');STEPS=(0,250,500,1000)
sys.path.insert(0,str(ROOT.parent/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies'))
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,obj):
    p=Path(p);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
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
def sources(d=None):return [v for v in load(ROOT/'source_manifest.json')['videos'] if d is None or v['dataset']==d]
def point(d,m,i,q):return ROOT/'parts'/d/m/f'video_{i:02d}_qp{q}.json'
def checkpoint(b,s):return ROOT/'branches'/b/'checkpoints'/f'step_{s:04d}.pt'
def sample_clip(rng,device,v4):
    sys.path.insert(0,str(V64));loader=module('v67_native_vimeo_loader',V64/'loader.py')
    return loader.sample_clip('vimeo',rng,device,v4)
def old_inventory():
    out={}
    for folder in (V64,V65,V66,V62,V62B,V4,ENGINE,ROOT.parent/'gvcrt_neural_wrapper_v6_3_dataset_ablation',ROOT.parent/'gvcrt_neural_wrapper_v4_1_high_convergence_metrics',ROOT.parent/'gvcrt_neural_wrapper_v2_joint'):
        if not folder.is_dir() or folder in (ROOT,ACTIVE_OLD):continue
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.lock':
                s=p.stat();out[str(p)]=[s.st_size,s.st_mtime_ns]
    return out
def frozen(full=False):
    for p,h in load(ROOT/'audits/dependencies.json').items():assert sha(p)==h,('dependency changed',p)
    for p,h in load(ROOT/'audits/local_protocol.json').items():assert sha(ROOT/p)==h,('local protocol changed',p)
    if full:assert old_inventory()==load(ROOT/'audits/old_inventory.json'),'Completed old experiment changed'
    return load(ROOT/'config.json')
def engine():
    sys.path.insert(0,str(ENGINE));import engine as e;return e
def eval_adapter():
    sys.path.insert(0,str(V62B));e=module('v67_frozen_evaluator',V62B/'evaluate.py')
    e.ROOT=ROOT;e.METHODS=METHODS;e.sources=sources;e.point=point;e.dump=dump;e.write=write;e.visualization=lambda v,q:None
    e.load=lambda p:load(ROOT/'evaluation/config.json') if Path(p)==ROOT/'config.json' else load(p)
    def check():
        frozen();cfg=load(ROOT/'evaluation/config.json')
        for cp in cfg['checkpoints'].values():assert sha(cp['path'])==cp['sha256']
        return cfg
    e.check_frozen=check;return e
