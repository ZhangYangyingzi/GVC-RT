import csv,hashlib,importlib.util,json,os,sys,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if (ROOT/'runtime_dependencies').is_dir():sys.path.insert(0,str(ROOT/'runtime_dependencies'))
REPO=ROOT.parents[1]
V3=ROOT.parent/'gvcrt_neural_wrapper_v3_pb_scale'
V4=ROOT.parent/'gvcrt_neural_wrapper_v4_progressive_joint'
V41=ROOT.parent/'gvcrt_neural_wrapper_v4_1_high_convergence_metrics'
V52=ROOT.parent/'gvcrt_neural_wrapper_v5a_2_fair_30fps_rgb'
OLD_ENGINE=ROOT.parent/'gvcrt_parallel_v5a3_v60'
V720=ROOT.parent/'gvcrt_neural_wrapper_v5a_4_virat_720p20'
ARCHIVES=Path('/Huang_group/zyyz/datasets/UltraVideo-Long/clips_long_1920')
SPLIT=REPO/'expericent_generation_input/expericent_generator_aware_latent_distortion_v11/data_split.json'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
PLOT_PYTHON='/data1/anaconda3_new/anaconda_program/bin/python'
METRICS=('LPIPS','DISTS','FloLPIPS','FID')
SPATIAL=('LPIPS','DISTS','PSNR','SSIM','MS_SSIM')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');t.replace(p)
def read(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    assert rows;p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
    with t.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    t.replace(p)
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def frozen():
    for p,h in load(ROOT/'old_source_hashes.json').items():assert sha(p)==h,('Frozen existing artifact changed',p)
def point(split,method,i,q):return ROOT/'parts'/split/method/f'video_{i}_qp{q}.json'
def checkpoint(step):return ROOT/'checkpoints'/f'additional_{step:05d}.pt'
def status(**kw):dump(ROOT/'pipeline_status.json',dict(updated_unix=time.time(),**kw))
