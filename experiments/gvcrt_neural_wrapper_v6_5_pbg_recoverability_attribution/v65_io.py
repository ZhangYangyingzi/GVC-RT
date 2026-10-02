"""Isolated inference-only experiment utilities."""
import csv,hashlib,importlib.util,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
V64=ROOT.parent/'gvcrt_neural_wrapper_v6_4_vimeo_pretrain'
V62B=ROOT.parent/'gvcrt_neural_wrapper_v6_2b_final_generalization_test'
AUDIT=ROOT.parent/'dataset_distribution_audit_v1'
ENGINE=ROOT.parent/'gvcrt_parallel_v5a3_v60'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
sys.path.insert(0,str(ROOT.parent/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies'))
METHODS={'M00_original':[None,None],'V62_P_only':['V62',None],'V62_BG_only':[None,'V62'],'V62_full':['V62','V62'],
         'V64_P_only':['V64',None],'V64_BG_only':[None,'V64'],'V64_full':['V64','V64']}
QPS=(0,4,9)
METRICS=('LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM')
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
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def videos():return load(ROOT/'source_manifest.json')['videos']
def sid(v):return f'{v["dataset"]}_{v["video_index"]:02d}'
def point(v,m,q):return ROOT/'parts/factorial'/sid(v)/m/f'qp{q}.json'
def chosen(v):return v['name'] in ('Beauty','Jockey','ReadySteadyGo','ShakeNDry','YachtRide') if v['dataset']=='uvg' else v['video_index'] in (0,1)
def visual_indices(v):return [0,15,31,63]
def inventory():
    result={}
    for folder in ROOT.parent.iterdir():
        if not folder.is_dir() or folder==ROOT:continue
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.lock':
                st=p.stat();result[str(p)]=[st.st_size,st.st_mtime_ns]
    return result
def frozen(full=False):
    for p,h in load(ROOT/'audits/dependency_hashes.json').items():assert sha(p)==h,('changed dependency',p)
    for p,h in load(ROOT/'audits/local_hashes.json').items():assert sha(ROOT/p)==h,('changed experiment source',p)
    if full:assert inventory()==load(ROOT/'audits/old_inventory.json'),'Old experiments changed'
    return load(ROOT/'config.json')
def frames_for(v):
    old=module('v65_frozen_loader',V62B/'v62b_io.py')
    return old.frames_for(v)
def engine():
    sys.path.insert(0,str(ENGINE));import engine as impl
    return impl
def bitstream_checks(v,rows,qps=QPS):
    result=[]
    for q in qps:
        own={r['method']:r for r in rows if r['QP']==q}
        for a,b in [('V62_P_only','V62_full'),('V64_P_only','V64_full'),('M00_original','V62_BG_only'),('M00_original','V64_BG_only')]:
            x,y=own[a],own[b];ok=x['real_bytes']==y['real_bytes'] and x['bitstream_sha256']==y['bitstream_sha256']
            result.append(dict(dataset=v['dataset'],sequence=v['name'],video_index=v['video_index'],QP=q,method_a=a,method_b=b,
                bytes_a=x['real_bytes'],bytes_b=y['real_bytes'],sha_a=x['bitstream_sha256'],sha_b=y['bitstream_sha256'],status='PASS' if ok else 'FAIL'))
    return result
