"""Independent full-window checks. No formal optimizer state or plan is consumed."""
import argparse,math,traceback
from io21 import *
from window_training import replay,make_runner

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True,choices=(4,5,6,7));ap.add_argument('--branch',required=True,choices=BRANCHES);a=ap.parse_args()
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(a.gpu))
 import torch
 from gradient_tools import stats,dists_details
 torch.set_num_threads(2);torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025);frozen();v4,im,pm,models,quality=load_training(torch.device('cuda:0'));params=[p for m in models.values() for p in m.parameters() if p.requires_grad];initial={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm)
 row=load(ROOT/'plans/calibration.json')['updates'][0];frames=replay(row,a.branch,torch.device('cuda:0'));q=row['external_qp'];run=make_runner(v4,im,pm,models,quality)
 actual_dists={}
 if a.branch=='A_decoder_continue':
  ds=quality[1];original_forward=ds.forward
  def traced_dists(output,target,**kwargs):
   if actual_dists:return original_forward(output,target,**kwargs)
   assert kwargs.get('require_grad') is True and output.requires_grad
   captured=[];once=ds.forward_once
   def grab(z):
    fs=once(z);captured.append(fs);return fs
   ds.forward_once=grab
   try:score=original_forward(output,target,**kwargs)
   finally:ds.forward_once=once
   allg=torch.autograd.grad(score.mean()*.5/15,params,retain_graph=True,allow_unused=True)
   weights=ds.alpha.sum()+ds.beta.sum();alpha=torch.split(ds.alpha/weights,ds.chns,dim=1);beta=torch.split(ds.beta/weights,ds.chns,dim=1)
   u,v=captured[0][-1],captured[1][-1];um=u.mean([2,3],keepdim=True);vm=v.mean([2,3],keepdim=True)
   s1=(2*um*vm+1e-6)/(um.square()+vm.square()+1e-6);uv=(u-um).square().mean([2,3],keepdim=True);vv=(v-vm).square().mean([2,3],keepdim=True);cov=(u*v).mean([2,3],keepdim=True)-um*vm
   s2=(2*cov+1e-6)/(uv+vv+1e-6);term=-((alpha[-1]*s1).sum(1,keepdim=True)+(beta[-1]*s2).sum(1,keepdim=True)).mean()
   deepg=torch.autograd.grad(term*.5/15,params,retain_graph=True,allow_unused=True);off=0;details={}
   for k,m in models.items():
    ps=[p for p in m.parameters() if p.requires_grad];n=len(ps);details[k]=dict(full_DISTS=stats(allg[off:off+n],ps),deepest_DISTS_term=stats(deepg[off:off+n],ps));off+=n
   assert all(z['full_DISTS']['finite'] and z['deepest_DISTS_term']['finite'] for z in details.values())
   assert details['generator']['deepest_DISTS_term']['norm']>0 and details['bridge']['deepest_DISTS_term']['norm']>0
   actual_dists.update(status='PASS',source_SHA256=EXPECTED,actual_reconstruction_graph=True,require_grad=True,lambda_dists=.5,frame_normalizer=15,frame_position=49,deep_layer=5,feature_shape=list(u.shape),module_gradients=details,optimizer_updates=0,formal_plan_consumed=0)
   return score
  ds.forward=traced_dists
 # Compare against unchanged v6.24 complete B1 implementation.
 oldio=module('v626_precheck_oldio',V25/'io21.py');ns=dict(oldio.__dict__);exec(compile((V25/'window_training.py').read_text().replace('from io21 import *','').replace('params=[p for m in models.values() for p in m.parameters()]','params=[p for m in models.values() for p in m.parameters() if p.requires_grad]').replace('n=len(list(m.parameters()))','n=len([p for p in m.parameters() if p.requires_grad])'),str(V25/'window_training.py'),'exec'),ns)
 old=ns['make_runner'](v4,im,pm,models,quality);torch.cuda.reset_peak_memory_stats()
 oldloss,oldqp,oldtr,oldwarm,_=old(frames,q,'A_continue',adv_weight=0);og=[p.grad.detach().clone() for p in params]
 for p in params:p.grad=None
 loss,qp,tr,warm,_=run(frames,q,a.branch,adv_weight=0);differences={};off=0
 for k,m in models.items():
  ps=[p for p in m.parameters() if p.requires_grad];pairs=list(zip(ps,og[off:off+len(ps)]));off+=len(ps)
  norm=sum(float(g.double().square().sum()) for _,g in pairs)**.5;err=sum(float((p.grad.double()-g.double()).square().sum()) for p,g in pairs)**.5
  differences[k]=dict(relative_L2=err/max(norm,1e-12),norm=norm);assert err/max(norm,1e-12)<.001
 assert qp==oldqp and warm==oldwarm and [x['reconstruction_sha256'] for x in tr]==[x['reconstruction_sha256'] for x in oldtr]
 assert abs(loss['loss']-oldloss['loss'])<1e-6;del og
 base_grads=[p.grad.detach().clone() for p in params]
 for p in params:p.grad=None
 x=torch.rand_like(frames[49],requires_grad=True);y=torch.rand_like(x);_,layers,_=dists_details(quality,x,y,True);assert all(t['output_gradient']['finite'] and t['output_gradient']['effective_parameters']>0 for t in layers)
 disc=None;checks={}
 if a.branch!='A_decoder_continue':
  sys.path.insert(0,str(ROOT))
  from illm_runtime import create_for
  disc=create_for(a.branch,torch.device('cuda:0'));disc.g_mode();fh=tensor_hash(disc.feature.state_dict());dh=disc.d_hash();runner=make_runner(v4,im,pm,models,quality,disc)
  zero,zq,zt,zw,_=runner(frames,q,a.branch,adv_weight=0.)
  zero_differences={};off=0
  for k,m in models.items():
   ps=[p for p in m.parameters() if p.requires_grad];pairs=list(zip(ps,base_grads[off:off+len(ps)]));off+=len(ps)
   norm=sum(float(g.double().square().sum()) for _,g in pairs)**.5;err=sum(float((p.grad.double()-g.double()).square().sum()) for p,g in pairs)**.5
   zero_differences[k]=dict(relative_L2=err/max(norm,1e-12),max_absolute_error=max([float((p.grad-g).abs().max()) for p,g in pairs] or [0.]));assert err/max(norm,1e-12)<.001
  assert abs(zero['loss']-oldloss['loss'])<1e-6 and zq==oldqp and zw==oldwarm and disc.d_hash()==dh and fh==tensor_hash(disc.feature.state_dict())
  del base_grads
  loss,qp,tr,warm,fakes=runner(frames,q,a.branch);assert loss['adv_B1_ratio']<=.05+1e-8 and disc.d_hash()==dh
  module_grads={k:stats([p.grad for p in m.parameters()],list(m.parameters())) for k,m in models.items()};assert all(s['finite'] for s in module_grads.values()) and all(module_grads[k]['norm']>0 for k in ('bridge','generator'))
  fake,c=fakes[0];assert not c.requires_grad and c.shape==torch.Size([1,1024,16,16])
  if hasattr(disc,'set_target'):disc.set_target(frames[49])
  z=fake.clone().requires_grad_(True);dz=torch.autograd.grad(disc.adversarial(z,c),z)[0];assert torch.isfinite(dz).all() and dz.abs().max()>0
  s1=[s.detach() for s in disc.scores(fake,c)];s2=[s.detach() for s in disc.scores(fake,c+1)];changed=max(float((x-y).abs().max()) for x,y in zip(s1,s2));assert changed>0
  opt=torch.optim.AdamW([dict(params=list(m.parameters()),lr=load(ROOT/'config.json')['optimizer'][k+'_lr']) for k,m in models.items() if k!='wrapper'],weight_decay=1e-4)
  torch.nn.utils.clip_grad_norm_(params,1.);opt.step();assert dh==disc.d_hash() and fh==tensor_hash(disc.feature.state_dict())
  gh={k:v4.module_hash(m) for k,m in models.items()};disc.d_mode();dop=torch.optim.Adam(disc.d_parameters(),lr=2e-4,betas=(0.,.9));dop.zero_grad();dv=0.
  for gt,(fake,c) in zip(frames[49:64],fakes):
   dl=disc.d_loss(gt,fake,c)/15;dl.backward();dv+=float(dl.detach())
  an=sum(float(p.grad.double().square().sum()) for p in (disc.discriminator.input_sequence.parameters() if a.branch=='C_decoder_illm' else disc.adapters.parameters()) if p.grad is not None)**.5
  assert math.isfinite(an) and an>0
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in disc.d_parameters());dop.step();assert gh=={k:v4.module_hash(m) for k,m in models.items()} and fh==tensor_hash(disc.feature.state_dict())
  checks=dict(zero_weight_actual_discriminator_path_gradient_difference=zero_differences,zero_weight_actual_discriminator_path_loss_error=abs(zero['loss']-oldloss['loss']),D_parameters_buffers_unchanged_in_G=True,G_unchanged_in_D=True,feature_projection_frozen=True,feature_input_gradient_norm=float(dz.norm()),condition_shape=list(c.shape),condition_detached=True,adapter_gradient_norm=an,condition_replacement_score_difference=changed,B_zero_preserved=(a.branch=='B_decoder_conditional_projected'),lambda_eff=loss['lambda_eff'],adv_B1_ratio=loss['adv_B1_ratio'],module_gradients=module_grads,D_loss=dv,G_updates=1,D_updates=1)
 else:
  opt=torch.optim.AdamW([dict(params=list(m.parameters()),lr=load(ROOT/'config.json')['optimizer'][k+'_lr']) for k,m in models.items() if k!='wrapper'],weight_decay=1e-4);torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
 assert core==v4.compression_hash(im,pm) and initial['wrapper']==v4.module_hash(models['wrapper'])
 if a.branch=='C_decoder_illm':
  import torch.nn.functional as F
  gt=frames[49];before=tensor_hash(disc.feature.state_dict());y=disc.labels(gt);raw=disc.feature(gt)
  with torch.no_grad():
   f=disc.feature.encoder(gt);z=F.normalize(disc.feature.bottleneck_op.input_conv(f),dim=1);flat=z.permute(0,2,3,1).reshape(-1,8);code=disc.feature.bottleneck_op.child.child.codebooks[0];manual=(flat.square().sum(1,keepdim=True)+code.square().sum(1)[None]-2*flat@code.T).argmin(1).reshape(1,32,32)
  assert torch.equal(y,raw.reshape(1,32,32)) and torch.equal(y,manual) and tensor_hash(disc.feature.state_dict())==before
  assert y.dtype==torch.long and y.min()>=0 and y.max()<1024
  dump(ROOT/'audits/labeler_probe.json',dict(status='PASS',weight_SHA256=sha(ROOT/'weights/vqvae_xcit_p8_ch64_cb1024_h8.ckpt'),shape=list(y.shape),dtype=str(y.dtype),min=int(y.min()),max=int(y.max()),row_major_BHW_layout=True,manual_nearest_codebook_indices_equal=True,stable_output=True,frozen_buffers=True,real_targets='y+1',fake_target=0,labels_no_codec_path=True))
 if actual_dists:dump(ROOT/'audits/actual_DISTS_optimizer_path.json',actual_dists)
 dump(ROOT/'audits'/f'{a.branch}_pretraining.json',dict(status='PASS',branch=a.branch,initial_modules=initial,source_sha256=EXPECTED,source_plan_sha256=sha(ROOT/'plans/shared.json'),sample=row,full_window=True,formal_plan_consumed=0,formal_optimizer_updates=0,isolated_process=True,zero_adversarial_legacy_gradient_difference=differences,forward_reconstruction_identical=True,DISTS_layers=layers,compression_frozen=True,**checks,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20))
 print('PRECHECK PASS',a.branch,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump(ROOT/'audits'/f'precheck_failure_{os.getpid()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
