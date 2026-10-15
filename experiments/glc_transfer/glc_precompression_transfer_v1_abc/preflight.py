"""Read-only source/weight checks; all audit writes isolated to this experiment."""
import os,sys,json,hashlib,subprocess,inspect,traceback,time
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYTHONNOUSERSITE']='1'
R=Path(__file__).resolve().parent
GLC=Path(os.environ.get('GLC_PROJECT','/Huang_group/zyyz/Projects/GLC'))
GVC=Path('/Huang_group/zyyz/Projects/GVC-RT')
sys.path.insert(0,str(GLC))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,x):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True)
 temporary=p.with_suffix(p.suffix+'.tmp');temporary.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temporary.replace(p)
def state_hash(s):
 h=hashlib.sha256()
 for k,t in sorted(s.items()):
  h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 import torch
 from src.models.image_model import GLC_Image
 from src.models.video_model import GLC_Video
 from src.utils.test_utils import get_state_dict
 files=[GLC/'README.md',GLC/'test_video.py',GLC/'test_bitstream.py']+list((GLC/'src').rglob('*.py'))
 repos={str(p):dict(commit=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip(),status=subprocess.check_output(['git','-C',str(p),'status','--porcelain=v1'],text=True)) for p in (GLC,GVC)}
 save('audits/source_inventory.json',dict(repos=repos,files={str(p):sha(p) for p in files},python=sys.executable,torch=torch.__version__,cuda_build=torch.version.cuda,GLC_import=inspect.getfile(GLC_Video),bytecode_disabled=sys.dont_write_bytecode,disk_free_bytes=__import__('shutil').disk_usage(R).free))
 rows=[]
 for name,cls,path in [('I',GLC_Image,GLC/'ckpts/GLC_image.pth.tar'),('P',GLC_Video,GLC/'ckpts/GLC_Video.pth.tar')]:
  model=cls();s=get_state_dict(str(path));wanted=model.state_dict()
  missing=sorted(set(wanted)-set(s));unexpected=sorted(set(s)-set(wanted));mismatch=[k for k in wanted.keys()&s.keys() if wanted[k].shape!=s[k].shape]
  row=dict(name=name,path=str(path),size=path.stat().st_size,sha256=sha(path),missing=missing,unexpected=unexpected,shape_mismatch=mismatch,critical_missing=[k for k in missing if k.startswith(('vqgan.','encoder.','enc.','decoder.','dec.','hyper_','y_','z_vq.','feature_','temporal_','prior_','weight_generator.'))])
  rows.append(row);save('audits/checkpoint_audit.json',dict(status='CHECKING',models=rows))
  if missing or mismatch:raise RuntimeError(f'{name} checkpoint incomplete: missing={missing}, shape_mismatch={mismatch}')
  # Explicitly account for checkpoint-only training heads, never swallow missing keys.
  row['checkpoint_only_training_keys']=unexpected
  result=model.load_state_dict({k:s[k] for k in wanted},strict=True)
  row.update(load_missing=list(result.missing_keys),load_unexpected=list(result.unexpected_keys),module_hash=state_hash(model.state_dict()),VQGAN_encoder_hash=state_hash(model.vqgan.encoder.state_dict()),VQGAN_generator_hash=state_hash(model.vqgan.generator.state_dict()),parameters=sum(p.numel() for p in model.parameters()))
  save('audits/checkpoint_audit.json',dict(status='PASS' if name=='P' else 'CHECKING',models=rows));del model,s
 save('audits/cpu_preflight.json',dict(status='PASS',time=time.time(),q_supported=list(range(64)),q_grid=[0,7,14,21,28,35,42,49,56,63],original_author_training_data='unknown'))
if __name__=='__main__':
 try:main()
 except Exception:
  e=traceback.format_exc();save('BLOCKED.json',dict(status='BLOCKED',stage='cpu_preflight',error=e,B_updates=0,C_updates=0,evaluation_completed=0));print(e,flush=True);sys.exit(1)
