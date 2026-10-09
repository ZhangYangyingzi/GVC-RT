"""Isolated IO for DISTS verification and the conditional fixed repair."""
import csv,hashlib,importlib.util,inspect,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
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
METHODS=('original','v617_initial','A_1000','B_1000','C_1000')
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
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
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
def historical_inventory():return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for folder in (V17,V16,V15,V62,V4,BASE,ENGINE,REPO/'checkpoints') for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.lock','.pid')}
def load_training(device):
    import torch
    v4=module('long18_v4',V4/'train.py');cfg=load(ROOT/'config.json');cp=cfg['source_checkpoint'];assert sha(cp['path'])==EXPECTED
    wrapper=v4.NeuralWrapper().to(device).float().train();im,pm=v4.load_models(device,force_zero_thres=.12);im.requires_grad_(False);pm.requires_grad_(False)
    models=dict(wrapper=wrapper,bridge=pm.recon_generation_net.mlp.float().train().requires_grad_(True),generator=pm.recon_generation_net.decoder.float().train().requires_grad_(True))
    s=torch.load(cp['path'],map_location='cpu',weights_only=True)
    for k,m in models.items():m.load_state_dict(s[k],strict=True)
    assert {k:v4.module_hash(m) for k,m in models.items()}==cp['module_hashes'];del s
    ids={id(p) for m in models.values() for p in m.parameters()}
    assert all(not p.requires_grad for m in (im,pm) for p in m.parameters() if id(p) not in ids)
    quality=v4.quality_models(device);assert all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in quality)
    return v4,im,pm,models,quality
def objective(v4,im,pm,models,quality,fixed=True,lambda_dists=None):
    import ast,math,torch
    import torch.nn.functional as F
    from types import SimpleNamespace
    common=module('long18_fullqp_common',V62/'fullqp_common.py')
    tree=ast.parse((V62/'fullqp_train.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');node=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='run_clip')
    if lambda_dists is None:lambda_dists=load(ROOT/'config.json')['lambda_dists']
    assert fixed and lambda_dists in (0.5,1.0)
    def perceptual(output,target,quality):
        lp=quality[0](output,target,normalize=True).mean();ds=quality[1](output,target,require_grad=True).mean();return lp+lambda_dists*ds,lp,ds
    wrapped=SimpleNamespace(**{n:getattr(v4,n) for n in dir(v4) if not n.startswith('__')})
    if fixed:wrapped.perceptual=perceptual
    cfg=load(ROOT/'config.json')
    ns=dict(torch=torch,F=F,math=math,v4=wrapped,im=im,pm=pm,wrapper=models['wrapper'],quality=quality,cfg=cfg,args=SimpleNamespace(branch='schedule_s1p0'),beta_q=common.beta_q,lambda_q=common.lambda_q,frame_qp=common.frame_qp,load=lambda p:load(V62/'qp_train_eval_semantics_audit.json') if Path(p).name=='qp_train_eval_semantics_audit.json' else load(p),ROOT=ROOT)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(V62/'fullqp_train.py'),'exec'),ns)
    dump(ROOT/'audits/objective_source.json',dict(status='PASS',source=str(V62/'fullqp_train.py'),sha256=sha(V62/'fullqp_train.py'),nested_function='main.run_clip',AST_sha256=hashlib.sha256(ast.dump(node).encode()).hexdigest(),AST_unmodified=True,only_override='quality[1](output,target,require_grad=True).mean()' if fixed else None,lambda_dists=lambda_dists,unweighted_DISTS_logged=True,quality_parameters_frozen=True,extra_loss=False,rate_gradient_fix=False))
    return ns['run_clip']
