"""Isolated V6.9 paths and read-only V6.8b protocol adapters."""
import csv,hashlib,importlib.util,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
V67=ROOT.parent/'gvcrt_neural_wrapper_v6_7b_b_convergence_audit'
V66=ROOT.parent/'gvcrt_neural_wrapper_v6_6_controlled_training'
V64=ROOT.parent/'gvcrt_neural_wrapper_v6_4_vimeo_pretrain'
V62=ROOT.parent/'gvcrt_neural_wrapper_v6_2_objective_calibration'
V62B=ROOT.parent/'gvcrt_neural_wrapper_v6_2b_final_generalization_test'
V4=ROOT.parent/'gvcrt_neural_wrapper_v4_progressive_joint'
ENGINE=ROOT.parent/'gvcrt_parallel_v5a3_v60'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
V68A=ROOT.parent/'gvcrt_neural_wrapper_v6_8a_b_continue_vs_interface_preservation'
V68B=ROOT.parent/'gvcrt_neural_wrapper_v6_8b_soft_generator_update_control'
V69=ROOT.parent/'gvcrt_neural_wrapper_v6_9_no_interface_alignment'
V610=ROOT.parent/'gvcrt_neural_wrapper_v6_10_conditional_perceptual_supervision'
BRANCHES=('F_frozen_core','J_joint_core')
LRS={b:5e-7 for b in BRANCHES}
def tag(b):return b.split('_')[0]
REUSED=('original','B1000')
FRESH=tuple(f'{b[0]}{s}' for b in BRANCHES for s in (1500,2000))
METHODS=REUSED+FRESH;B_METHODS=METHODS
DIAG_METHODS=()
DATASETS=('uvg','ulong','virat720','virat480');METRICS=('LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM')
SOURCE=V66/'branches/B_joint_structure_guard/checkpoints/step_1000.pt'
MODE='no_interface_alignment'
sys.path.insert(0,str(ROOT.parent/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies'))
sys.dont_write_bytecode=True
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
def tensor_hash(state):
    h=hashlib.sha256()
    for k,t in state.items():h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def state_hash(obj):
    import torch
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):h.update(str((str(x.dtype),tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for k in sorted(x,key=lambda z:(type(z).__name__,repr(z))):visit(k);visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for v in x:visit(v)
        else:h.update((type(x).__name__+':'+repr(x)+';').encode())
    visit(obj);return h.hexdigest()
def optimizer_summary(s):
    # JSON serializes Adam betas tuples as lists; canonicalize the summary too.
    # The recursive hash above still verifies the exact original optimizer state.
    groups=[{k:v for k,v in g.items() if k!='params'}|{'parameters':len(g['params'])} for g in s['param_groups']]
    return dict(hash=state_hash(s),state_entries=len(s['state']),groups=json.loads(json.dumps(groups)))
def sources(d=None):return [v for v in load(ROOT/'source_manifest.json')['videos'] if d is None or v['dataset']==d]
def point(d,m,i,q):return ROOT/'parts'/d/m/f'video_{i:02d}_qp{q}.json'
def checkpoint(b,s):return ROOT/'branches'/b/'checkpoints'/f'step_{s}.pt'
def sample_clip(rng,device,v4):
    sys.path.insert(0,str(V64));m=module('v68_vimeo_loader',V64/'loader.py');return m.sample_clip('vimeo',rng,device,v4)
def old_inventory():
    return {str(p):[p.stat().st_size,p.stat().st_mtime_ns]
            for folder in (V66,V68A,V69,V610) for p in folder.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.lock'}
def frozen(full=False):
    for name,h in load(ROOT/'audits/local_protocol.json').items():assert sha(ROOT/name)==h,('protocol changed',name)
    for p,h in load(ROOT/'audits/dependencies.json').items():assert sha(p)==h,('dependency changed',p)
    if full:assert old_inventory()==load(ROOT/'audits/old_inventory.json'),'old experiments changed'
    cfg=load(ROOT/'config.json');cfg.update(load(ROOT/'evaluation/config.json'));return cfg
def engine():
    # Isolated module adapter; old sources remain read-only.
    key='v611_engine'
    if key not in sys.modules:
        e=module(key,ROOT/'codec_runtime.py');sys.modules[key]=e
    return sys.modules[key]
def eval_adapter():
    sys.path.insert(0,str(V62B));e=module('v611_evaluation',V62B/'evaluate.py')
    sys.modules['engine']=engine()
    e.ROOT=ROOT;e.METHODS=METHODS;e.sources=sources;e.point=point;e.dump=dump;e.write=write;e.visualization=lambda v,q: q in (0,4,9) and ((v['dataset']=='ulong' and v['video_index']<2) or (v['dataset']=='uvg' and v['name'] in ('Beauty','Jockey','ReadySteadyGo')) or (v['dataset'].startswith('virat') and v['video_index']<2))
    e.load=lambda p:load(ROOT/'evaluation/config.json') if Path(p)==ROOT/'config.json' else load(p)
    def check():
        frozen();cfg=load(ROOT/'evaluation/config.json')
        for cp in cfg['checkpoints'].values():assert sha(cp['path'])==cp['sha256']
        return cfg
    e.check_frozen=check;return e
def validate_point(r,v,q,m):
    eval_adapter().validate(r,v,q,m)
    expected=load(ROOT/'evaluation/config.json')['core_hashes'][m]
    assert r['compression_hash_before']==r['compression_hash_after']==expected
    if r.get('reused'):assert sha(r['reused_from'])==r['reused_point_sha256']
def array_hash(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def export_rows(rows):return rows
def fid_function():
    sys.path.insert(0,str(V67));return module('v68_fid_reference',V67/'v67b_io.py').fid_function()
def adapted_script(name):
    """Execute identical V6.7b worker code, overriding only IO/cohort/method bindings."""
    sys.path.insert(0,str(V67));m=module('v68_reuse_'+name,V67/(name+'.py'))
    for k in ('ROOT','METHODS','B_METHODS','DATASETS','load','dump','write','read','frozen','sources','point','validate_point','fid_function','array_hash','export_rows','engine'):
        setattr(m,k,globals()[k])
    # Workers reading config directly also need newly available checkpoints.
    m.load=lambda p:frozen() if Path(p)==ROOT/'config.json' else load(p)
    return m
