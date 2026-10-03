"""Frozen V6.5 dependencies and local output helpers for paired UVG diagnostics."""
import ast,csv,hashlib,importlib.util,json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
V65=ROOT.parent/'gvcrt_neural_wrapper_v6_5_pbg_recoverability_attribution'
V52=ROOT.parent/'gvcrt_neural_wrapper_v5a_2_fair_30fps_rgb'
AUDIT=ROOT.parent/'dataset_distribution_audit_v1'
ENGINE=ROOT.parent/'gvcrt_parallel_v5a3_v60'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
sys.path.insert(0,str(V65))
import v65_io as old
VIDEOS=('Beauty','Jockey','ReadySteadyGo','ShakeNDry','YachtRide')
ANCHORS=(0,32,64,96,128,160,192,224)
STRIDES=(1,2,4)
METHODS={'M00_original':[None,None],'V62_P_only':['V62',None],'V62_full':['V62','V62'],'V64_P_only':['V64',None],'V64_full':['V64','V64']}
METRICS=('LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM')
LOWER=('LPIPS','DISTS','FloLPIPS')
def load(p):return json.loads(Path(p).read_text())
def sha(p):return old.sha(p)
def dump(p,obj):
 p=Path(p);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def write(p,rows):
 p=Path(p);assert p.is_relative_to(ROOT) and rows;p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
 tmp.replace(p)
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def module(name,p):return old.module(name,p)
def videos():return load(ROOT/'source_manifest.json')['videos']
def part(v,anchor,stride):return ROOT/'parts'/v['name']/f'a{anchor:03d}_s{stride}.json'
def engine():return old.engine()
def source_frames(v):
 """Use exactly V5.2's audited FFmpeg YUV420 BT.709 limited-range conversion."""
 import numpy as np
 command=list(load(V52/'ffmpeg_conversion_audit.json')['inherited_command'])
 command[command.index('-i')+1]=v['source_path'];command[command.index('-frames:v')+1]='229'
 wanted={t+s for t in ANCHORS for s in (0,*STRIDES)}
 out={};nbytes=1920*1080*3
 with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
  for i in range(229):
   raw=proc.stdout.read(nbytes)
   assert len(raw)==nbytes,(v['name'],i,len(raw))
   if i in wanted:out[i]=np.frombuffer(raw,dtype=np.uint8).reshape(1080,1920,3).copy()
  assert proc.stdout.read(1)==b''
  err=proc.stderr.read();assert proc.wait()==0,err.decode(errors='replace')
 assert set(out)==wanted
 return out

def motion_functions():
 import cv2,numpy as np
 tree=ast.parse((AUDIT/'source_stats.py').read_text())
 tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('gray','gradient','camera')]
 ns=dict(cv2=cv2,np=np,math=math);exec(compile(tree,str(AUDIT/'source_stats.py'),'exec'),ns)
 return ns['gray'],ns['gradient'],ns['camera']
def cfg():return load(ROOT/'config.json')
