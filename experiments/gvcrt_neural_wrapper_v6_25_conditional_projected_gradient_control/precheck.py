"""Independent full-window checks. No formal optimizer state or plan is consumed."""
import argparse,math,traceback
from io21 import *
from window_training import replay,make_runner

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True,choices=(4,5,6,7));ap.add_argument('--branch',required=True,choices=BRANCHES);a=ap.parse_args()
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(a.gpu))
 import torch
 from gradient_tools import stats,dists_details
 torch.set_num_threads(2);frozen();v4,im,pm,models,quality=load_training(torch.device('cuda:0'));params=[p for m in models.values() for p in m.parameters()];initial={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm)
 row=load(ROOT/'plans/calibration.json')['updates'][0];frames=replay(row,a.branch,torch.device('cuda:0'));q=row['external_qp'];run=make_runner(v4,im,pm,models,quality)
 # Compare against unchanged v6.24 complete B1 implementation.
 oldio=module('v625_precheck_oldio',V24/'io21.py');ns=dict(oldio.__dict__);exec(compile((V24/'window_training.py').read_text().replace('from io21 import *',''),str(V24/'window_training.py'),'exec'),ns)
 old=ns['make_runner'](v4,im,pm,models,quality);torch.cuda.reset_peak_memory_stats()
 oldloss,oldqp,oldtr,oldwarm,_=old(frames,q,'A_continue',adv_weight=0);og=[p.grad.detach().clone() for p in params]
 for p in params:p.grad=None
 loss,qp,tr,warm,_=run(frames,q,a.branch,adv_weight=0);differences={};off=0
 for k,m in models.items():
  ps=list(m.parameters());pairs=list(zip(ps,og[off:off+len(ps)]));off+=len(ps)
  norm=sum(float(g.double().square().sum()) for _,g in pairs)**.5;err=sum(float((p.grad.double()-g.double()).square().sum()) for p,g in pairs)**.5
  differences[k]=dict(relative_L2=err/max(norm,1e-12),norm=norm);assert err/max(norm,1e-12)<.001
 assert qp==oldqp and warm==oldwarm and [x['reconstruction_sha256'] for x in tr]==[x['reconstruction_sha256'] for x in oldtr]
 assert abs(loss['loss']-oldloss['loss'])<1e-6;del og
 for p in params:p.grad=None
 x=torch.rand_like(frames[49],requires_grad=True);y=torch.rand_like(x);_,layers,_=dists_details(quality,x,y,True);assert all(t['output_gradient']['finite'] and t['output_gradient']['effective_parameters']>0 for t in layers)
 disc=None;checks={}
 if a.branch!='A_continue':
  sys.path.insert(0,str(ROOT))
  from projected import create
  disc=create(torch.device('cuda:0'));disc.g_mode();fh=tensor_hash(disc.feature.state_dict());dh=disc.d_hash();runner=make_runner(v4,im,pm,models,quality,disc)
  loss,qp,tr,warm,fakes=runner(frames,q,a.branch);assert loss['adv_B1_ratio']<=.05+1e-8 and disc.d_hash()==dh
  module_grads={k:stats([p.grad for p in m.parameters()],list(m.parameters())) for k,m in models.items()};assert all(s['finite'] and s['norm']>0 for s in module_grads.values())
  fake,c=fakes[0];assert not c.requires_grad and c.shape==torch.Size([1,1024,16,16])
  if a.branch=='B_capped_projected':
   assert torch.count_nonzero(c)==0 and all(torch.count_nonzero(d(c))==0 for d in disc.adapters.values())
  z=fake.clone().requires_grad_(True);dz=torch.autograd.grad(disc.adversarial(z,c),z)[0];assert torch.isfinite(dz).all() and dz.abs().max()>0
  s1=[s.detach() for s in disc.scores(fake,c)];s2=[s.detach() for s in disc.scores(fake,c+1)];changed=max(float((x-y).abs().max()) for x,y in zip(s1,s2));assert changed>0
  opt=torch.optim.AdamW([dict(params=list(m.parameters()),lr=load(ROOT/'config.json')['optimizer'][k+'_lr']) for k,m in models.items()],weight_decay=1e-4)
  torch.nn.utils.clip_grad_norm_(params,1.);opt.step();assert dh==disc.d_hash() and fh==tensor_hash(disc.feature.state_dict())
  gh={k:v4.module_hash(m) for k,m in models.items()};disc.d_mode();dop=torch.optim.Adam(disc.d_parameters(),lr=2e-4,betas=(0.,.9));dop.zero_grad();dv=0.
  for gt,(fake,c) in zip(frames[49:64],fakes):
   dl=disc.d_loss(gt,fake,c)/15;dl.backward();dv+=float(dl.detach())
  an=sum(float(p.grad.double().square().sum()) for p in disc.adapters.parameters() if p.grad is not None)**.5
  if a.branch=='C_capped_conditional_projected':assert math.isfinite(an) and an>0
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in disc.d_parameters());dop.step();assert gh=={k:v4.module_hash(m) for k,m in models.items()} and fh==tensor_hash(disc.feature.state_dict())
  checks=dict(D_parameters_buffers_unchanged_in_G=True,G_unchanged_in_D=True,feature_projection_frozen=True,feature_input_gradient_norm=float(dz.norm()),condition_shape=list(c.shape),condition_detached=True,adapter_gradient_norm=an,condition_replacement_score_difference=changed,B_zero_preserved=(a.branch=='B_capped_projected'),lambda_eff=loss['lambda_eff'],adv_B1_ratio=loss['adv_B1_ratio'],module_gradients=module_grads,D_loss=dv,G_updates=1,D_updates=1)
 else:
  opt=torch.optim.AdamW([dict(params=list(m.parameters()),lr=load(ROOT/'config.json')['optimizer'][k+'_lr']) for k,m in models.items()],weight_decay=1e-4);torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
 assert core==v4.compression_hash(im,pm)
 dump(ROOT/'audits'/f'{a.branch}_pretraining.json',dict(status='PASS',branch=a.branch,initial_modules=initial,source_sha256=EXPECTED,source_plan_sha256=sha(ROOT/'plans/shared.json'),sample=row,full_window=True,formal_plan_consumed=0,formal_optimizer_updates=0,isolated_process=True,zero_adversarial_legacy_gradient_difference=differences,forward_reconstruction_identical=True,DISTS_layers=layers,compression_frozen=True,**checks,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20))
 print('PRECHECK PASS',a.branch,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump(ROOT/'audits'/f'precheck_failure_{os.getpid()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
