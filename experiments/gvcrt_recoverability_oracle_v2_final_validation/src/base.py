import os,json,hashlib,time,csv,sys,importlib.util,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
V32=REPO/'experiments/gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion'
V33=REPO/'experiments/gvcrt_neural_wrapper_v6_33_p_causal_rate_utility_ablation'
ENGINE=REPO/'experiments/gvcrt_parallel_v5a3_v60/engine.py'
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,value):
 p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
 t.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def write(p,rows):
 p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def module(name,path):
 sys.path.insert(0,str(Path(path).parent));s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def key(row,qp):return row['dataset']+'_'+str(row['video_index'])+'_q'+str(qp)
def job_id(job):return job['id']
