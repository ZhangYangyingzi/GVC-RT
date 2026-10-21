"""Paired step0 objective/VJP validation. This function never calls optimizer.step."""
from io32 import *
def verify(ctx,source,allmods,basic_crop=None):
 if basic_crop is None:basic_crop=sys.modules["__main__"].basic_crop
 assert Path(basic_crop.__code__.co_filename).resolve()==ROOT/"train.py"
 import torch,training32 as tr
 from frames32 import PlanFrames
 row=load(ROOT/'plans/first100.json')['updates'][0]['source_row'];v4,im,pm,models,quality,common,params,_=ctx
 before={k:v4.module_hash(m) for k,m in allmods.items()};summaries=[];gradients=[];components=[]
 for weight in (0.,.5):
  tr.restore_rng(source['crop_rng']);tr.restore_buffers(allmods,source['buffers'])
  frames=PlanFrames(row,'cuda:0')
  gs,s=basic_crop(ctx,frames,row['external_qp'],audit=True,dists_weight=weight);frames.close()
  components.append(s.pop('audit_component_tensors'));summaries.append(s);gradients.append(gs)
 a,b=[s['losses'] for s in summaries]
 for k in ['LPIPS','DISTS','rate_bpp','beta_rate','proxy_L1','reconstruction_L1','weighted_reconstruction_L1']:assert abs(a[k]-b[k])<1e-8,(k,a[k],b[k])
 scalar_error=abs((b['total_loss']-a['total_loss'])-.5*b['DISTS']);assert scalar_error<1e-7
 d=components[1]['weighted_DISTS'];dn=tr.norm(d);assert dn>0 and all(torch.isfinite(x).all() for x in d)
 difference=[y-x-z for x,y,z in zip(gradients[0],gradients[1],d)]
 absolute_error=tr.norm(difference);rel=absolute_error/max(dn,1e-12);assert rel<1e-3,(absolute_error,dn,rel)
 assert all(tr.norm(c['rate_proxy'])==0 for c in components)
 assert a['proxy_L1']==b['proxy_L1']==0
 assert all(p.grad is None and not p.requires_grad for m in (models['bridge'],im,*quality) for p in m.parameters())
 assert all(p.grad is None and not p.requires_grad for n,p in pm.named_parameters() if not n.startswith('recon_generation_net.decoder.'))
 tr.restore_rng(source['crop_rng']);tr.restore_buffers(allmods,source['buffers'])
 assert before=={k:v4.module_hash(m) for k,m in allmods.items()}
 dump(ROOT/'audits/objective_gradient_step0.json',dict(status='PASS',same_step0_same_input=True,extra_optimizer_steps=0,scalar_difference_error=scalar_error,gradient_difference_relative_L2_error=rel,gradient_difference_absolute_L2_error=absolute_error,gradient_subtraction_relative_tolerance=1e-3,DISTS_generator_gradient_norm=dn,rate_proxy_generator_gradient_norm=0,identity_proxy_L1=0,bridge_and_codec_parameter_gradients_absent=True,modules_unchanged=True,common_components=summaries,RNG_and_buffers_restored=True))
