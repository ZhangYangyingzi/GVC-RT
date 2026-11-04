"""Reconstruct the complete frozen parent, then overlay trained TCG and AdamW."""
from ab_common import *
import importlib.util
PILOT=ORIGINAL/'experiments/gvcrt_tcg_stage1_pilot/checkpoints/adaptation_step_0.pt'
RAFT=Path('/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def build(rank=0):
 import torch
 def no_download(*a,**kw):raise RuntimeError('LOCAL_MODEL_CACHE_REQUIRED; downloading weights is prohibited')
 torch.hub.download_url_to_file=no_download
 sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'experiments/gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100'))
 sys.path.insert(0,str(ORIGINAL/'expericent_generation_input/expericent_interface_causal_controls_v9/src'))
 import training32 as tr
 tr.backend()
 v4=tr.module('ab_v4',REPO/'experiments/gvcrt_neural_wrapper_v4_progressive_joint/train.py')
 device=torch.device('cuda',rank);im,pm=v4.load_models(device,force_zero_thres=.12)
 models=dict(wrapper=torch.nn.Identity().to(device),bridge=pm.recon_generation_net.mlp.float(),generator=pm.recon_generation_net.decoder.float())
 full=torch.load(PILOT,map_location='cpu',weights_only=True)
 assert sha(PILOT)==load(P/'protocol.json')['source_checkpoint']['sha256']
 for k,m in models.items():m.load_state_dict(full['models'][k],strict=True)
 quality=v4.quality_models(device)
 tr.restore_buffers(dict(models,im=im,pm=pm,lp=quality[0],dists=quality[1]),full['buffers'])
 from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
 assert sha(RAFT)==load(P/'protocol.json')['raft_sha256']
 raft=raft_large(weights=None,progress=False);raft.load_state_dict(torch.load(RAFT,map_location='cpu',weights_only=True),strict=True)
 raft=raft.to(device).eval().requires_grad_(False);transform=Raft_Large_Weights.C_T_SKHT_V2.transforms()
 for m in (im,pm,*quality,raft,*models.values()):m.eval().requires_grad_(False)
 gen=models['generator'];gen.train();gen.set_temporal_cascade_grad(False)
 for n,p in gen.named_parameters():p.requires_grad_(n.startswith(PREFIXES))
 named=[(n,p) for n,p in gen.named_parameters() if p.requires_grad];params=[p for _,p in named]
 assert sum(p.numel() for p in params)==137760 and all(p.dtype==torch.float32 for p in params)
 parent=torch.load(P/'checkpoints/step_5000.pt',map_location='cpu',weights_only=True);assert sha(P/'checkpoints/step_5000.pt')==PARENT_SHA
 missing,unexpected=gen.load_state_dict(parent['tcg_model_state'],strict=False)
 assert not unexpected and all(not n.startswith(PREFIXES) for n in missing)
 assert all(torch.equal(gen.state_dict()[n].detach().cpu(),v) for n,v in parent['tcg_model_state'].items())
 optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
 optimizer.load_state_dict(parent['optimizer']);assert {int(s['step']) for s in optimizer.state.values()}=={5000}
 for g in optimizer.param_groups:assert g['lr']==1e-4 and g['betas']==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==1e-4
 common=tr.module('ab_qp',REPO/'experiments/gvcrt_neural_wrapper_v6_2_objective_calibration/fullqp_common.py')
 torch.cuda.set_device(rank);tr.restore_rng(parent['all_rank_RNG_states'][rank])
 return dict(tr=tr,im=im,pm=pm,models=models,quality=quality,raft=raft,transform=transform,named=named,params=params,optimizer=optimizer,parent=parent,ctx=(v4,im,pm,models,quality,common,params,None))
def tensor_hash(items):
 h=hashlib.sha256()
 for n,t in items:h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def frozen_items(c):
 for prefix,m in [('im',c['im']),('pm',c['pm']),('lp',c['quality'][0]),('dists',c['quality'][1]),('raft',c['raft'])]:
  for n,t in m.state_dict().items():
   if not(prefix=='pm' and n.startswith(tuple('recon_generation_net.decoder.'+p for p in PREFIXES))):yield prefix+'.'+n,t
