"""Isolated inference-only convergence audit with read-only evaluator adapters."""
import csv,hashlib,importlib.util,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
V64=ROOT.parent/'gvcrt_neural_wrapper_v6_4_vimeo_pretrain'
V67A=ROOT.parent/'gvcrt_neural_wrapper_v6_7a_guard_mechanism_ablation'
V66=ROOT.parent/'gvcrt_neural_wrapper_v6_6_controlled_training'
V65=ROOT.parent/'gvcrt_neural_wrapper_v6_5_pbg_recoverability_attribution'
V62=ROOT.parent/'gvcrt_neural_wrapper_v6_2_objective_calibration'
V62B=ROOT.parent/'gvcrt_neural_wrapper_v6_2b_final_generalization_test'
V4=ROOT.parent/'gvcrt_neural_wrapper_v4_progressive_joint'
ENGINE=ROOT.parent/'gvcrt_parallel_v5a3_v60'
ACTIVE_OLD=ROOT.parent/'gvcrt_neural_wrapper_v6_5b_uvg_temporal_stride_recoverability'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
A='A_baseline';O='original';B_METHODS=('B250','B500','B1000');FRESH=('B250','B500')
METHODS=(O,A,*B_METHODS);DATASETS=('uvg','ulong','virat720','virat480')
METRICS=('LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM');PRIMARY=('LPIPS','DISTS','FloLPIPS','FID')
OLD_B='B_joint_structure_guard'
FID_SOURCE=REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py'
FID_WEIGHTS=Path('/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth')
PLOT_PYTHON='/data1/anaconda3_new/anaconda_program/bin/python'
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
def old_inventory():
    out={}
    for folder in (V67A,V64,V65,V66,V62,V62B,V4,ENGINE,ROOT.parent/'gvcrt_neural_wrapper_v6_3_dataset_ablation',ROOT.parent/'gvcrt_neural_wrapper_v4_1_high_convergence_metrics',ROOT.parent/'gvcrt_neural_wrapper_v2_joint'):
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
    sys.path.insert(0,str(V62B));e=module('v67b_frozen_evaluator',V62B/'evaluate.py')
    e.ROOT=ROOT;e.METHODS=METHODS;e.sources=sources;e.point=point;e.dump=dump;e.write=write;e.visualization=lambda v,q:None
    e.load=lambda p:load(ROOT/'evaluation/config.json') if Path(p)==ROOT/'config.json' else load(p)
    def check():
        frozen();cfg=load(ROOT/'evaluation/config.json')
        for cp in cfg['checkpoints'].values():assert sha(cp['path'])==cp['sha256']
        return cfg
    e.check_frozen=check;return e

def export_rows(rows):
    return [{k:('Original_GVCRT' if v==O else v) if isinstance(v,str) and k in ('method','anchor') else v for k,v in r.items()} for r in rows]
def fid_function():
    import ast,math,numpy as np
    tree=ast.parse(FID_SOURCE.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='low_rank_fid']
    context=dict(math=math,np=np);exec(compile(tree,str(FID_SOURCE),'exec'),context);return context['low_rank_fid']
def array_hash(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def validate_point(r,v,q,m):
    eval_adapter().validate(r,v,q,m)
    assert r['compression_hash_before']==load(ROOT/'config.json')['compression_hash']
    assert r['source_frame_indices']==v['source_frame_indices']
    if r.get('reused'):assert sha(r['reused_from'])==r['reused_point_sha256']
