from preflight import *
from data import load,Frames
import argparse
def main(gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch,importlib.metadata
 from runtime import models,Sender,step,quality
 im,pm=models('cuda:0');before=[state_hash(m.state_dict()) for m in (im,pm)];row=load(R/'plans/shared.json')['updates'][0];records=[];reference=None
 for kind in ('A_original','B_rgb_P','C_latent_P'):
  torch.manual_seed(20261015);p=None if kind=='A_original' else Sender(kind).cuda().eval();f=Frames(row,True);dpb=None;hashes=[];bits=[]
  with torch.no_grad():
   for i in range(64):
    e,out,dpb,pad=step(im,pm,f.tensor(i,'cuda:0'),dpb,row['q'],i,p);hashes.append(state_hash({k:t for k,t in dpb.items() if k in ('ref_frame','ref_latent','ref_y') and t is not None}));bits.append(float(e['bit']))
  f.close();current=dict(reference_hashes=hashes,estimated_bits=bits)
  if reference is None:reference=current
  else:assert current==reference,('identity_initialization_changed_reference_or_rate',kind)
  records.append(dict(branch=kind,frames=64,estimated_total_bits=sum(bits),reference_trajectory_sha256=hashlib.sha256(''.join(hashes).encode()).hexdigest(),identity_exact=True,parameters=0 if p is None else sum(t.numel() for t in p.parameters()),crop_sender_conv_FLOPs=0 if p is None else sum(2*m.weight.numel()*(256 if kind=='B_rgb_P' else 16)**2 for m in p.modules() if isinstance(m,torch.nn.Conv2d))))
 assert before==[state_hash(m.state_dict()) for m in (im,pm)]
 save('audits/identity.json',dict(status='PASS',rows=records,tolerance='exact same FP32 output/rate/reference state hashes',optimizer_updates=0,frozen_unchanged=True,plan_step=row['step']))
 import inspect
 ql=quality('cuda:0');deps=[]
 for m in ql:
  path=Path(inspect.getfile(type(m)));deps.append(dict(path=str(path),source_sha256=sha(path),module_hash=state_hash(m.state_dict()),forward_signature=str(inspect.signature(m.forward)),frozen=all(not p.requires_grad for p in m.parameters())))
 save('audits/metric_dependencies.json',dict(quality=deps,versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','lpips','DISTS-pytorch')},Inception_weights_sha256=sha('/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth'),eval_implementation='GVC fid_metric.py and flolpips_compat.py, direct isolated imports; no GVC src import'))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True);args=a.parse_args()
 try:main(args.gpu)
 except Exception:
  e=traceback.format_exc();save('BLOCKED.json',dict(status='BLOCKED',stage='identity',error=e));print(e);sys.exit(1)
