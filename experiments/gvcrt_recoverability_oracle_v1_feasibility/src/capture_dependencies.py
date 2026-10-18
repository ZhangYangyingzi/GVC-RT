"""Read-only import discovery and source snapshots; no models or training jobs."""
from base import *
import importlib.metadata
import shutil

def main():
 import torch
 engine=module('recoverability_dependency_engine',ENGINE)
 sys.path.insert(0,str(V32))
 import adapter32,training32,lpips,DISTS_pytorch
 sys.path.insert(0,str(training32.V29/'vendor/pytorch-fid/src'))
 import pytorch_fid.inception
 sources={Path(m.__file__).resolve() for m in tuple(sys.modules.values())
          if getattr(m,'__file__',None) and str(m.__file__).startswith('/Huang_group/zyyz/Projects/') and str(m.__file__).endswith('.py')}
 compat=Path('/Huang_group/zyyz/Projects/cosmos-predict1/upsample_LUVE/stage28_exact_stage23_fallback')
 sources.update((compat/'third_party/flolpips').rglob('*.py'))
 sources.update((training32.V29/'vendor/pytorch-fid').rglob('LICENSE*'))
 sources.update((compat/'third_party/flolpips').glob('LICENSE*'))
 sources.update([V32/'evaluation_config.json',V32/'qp_semantics_audit.json',ENGINE])
 records=[]
 for p in sorted(sources):
  if p.is_relative_to(ROOT):continue
  rel=p.relative_to('/Huang_group/zyyz/Projects')
  target=ROOT/'src/dependency_snapshot'/rel
  target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert sha(target)==sha(p)
  else:shutil.copy2(p,target)
  records.append(dict(original_path=str(p),snapshot=str(target.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
 dump('manifests/dependency_sources.json',dict(records=records,restoration='Snapshots preserve original project-relative paths. They are evidence and recovery copies; no original dependency was overwritten. Runtime reuses the audited server paths.'))
 distributions=sorted((dict(name=d.metadata['Name'],version=d.version) for d in importlib.metadata.distributions()),key=lambda x:x['name'].lower())
 dump('manifests/environment.json',dict(python=sys.version,executable=sys.executable,torch=torch.__version__,CUDA_runtime=torch.version.cuda,packages=distributions))
 weights={}
 paths=[Path('/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints')/n for n in ('pwc-network-default.pytorch','weights-inception-2015-12-05-6726825d.pth','alexnet-owt-7be5be79.pth','vgg16-397923af.pth')]
 paths+=list((Path(lpips.__file__).parent/'weights').rglob('*.pth'))
 paths+=list(Path(DISTS_pytorch.__file__).parent.rglob('*.pt'))
 paths+=[compat/'third_party/flolpips/weights/v0.1/alex.pth']
 for p in paths:
  if p.is_file():weights[str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size)
 dump('manifests/metric_weights.json',weights)
 print('Captured',len(records),'dependency source/config files and',len(weights),'metric weight hashes')

if __name__=='__main__':main()
