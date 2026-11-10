"""Same-dtype B300 equality, corrected math, history isolation and H2 controls."""
from common_s2 import *
import torch,statistics

def main(rank=0):
 env();torch.cuda.set_device(rank)
 if (ROOT/'audits/preflight.json').exists():assert load(ROOT/'audits/preflight.json')['status']=='PASS';return
 from pipeline import wait_rows,request
 from cache_codec import cache_paths
 from experiment_model import construct,reconstruct,auxiliary,cache_snapshot,assert_cache
 from b300_loader import construct as reference,reconstruct as reference_forward
 from model_context import assert_reset
 from loss import latent_loss,norm
 from temporal import checks
 rows=load(ROOT/'manifests/calibration_windows.json')
 for row in rows:request(row,row['external_qp'],True)
 wait_rows([rows[0]]);_,p,rp=cache_paths(rows[0],rows[0]['external_qp'],True);assert sha(p)==load(rp)['cache_sha256'];data=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
 equality=[];parameter_record=None
 for train in [True,False]:
  ref=reference('B300',train,rank);expected=tensor_hash(ref['pm'].recon_generation_net.state_dict().items())
  for group in GROUPS:
   c=construct(group,train,rank);assert expected==tensor_hash(c['pm'].recon_generation_net.state_dict().items())
   if train and parameter_record is None:
    parameter_record=dict(counts=dict(Bridge=801124,TCG=137760,LoRA=59392,total=998276),trainables=[dict(name=n,numel=p.numel(),shape=list(p.shape),dtype=str(p.dtype)) for n,p in c['named']],frozen=[dict(name=n,numel=p.numel(),shape=list(p.shape),dtype=str(p.dtype)) for n,p in c['pm'].named_parameters() if not p.requires_grad],no_duplicates=len({id(p) for p in c['params']})==len(c['params']),LoRA_wrappers=5)
    common=ROOT/'checkpoints/common_B300_step0.pt';atomic_torch(common,dict(B300_SHA256=B300_SHA,recon_state={n:t.detach().cpu().clone() for n,t in c['pm'].recon_generation_net.state_dict().items()},trainable_state={n:p.detach().cpu().clone() for n,p in c['named']},provenance=c['provenance'],s=.5))
   assert_reset(ref);assert_reset(c);errors=[]
   with torch.no_grad():
    for i in range(1,64):
     a=reference_forward(ref,data['features'][i].to(ref['device']),data['info']['actual_qps'][i]);b=reconstruct(c,data['features'][i],data['info']['actual_qps'][i]);err=float((a-b).abs().max());assert err==0,('STEP0_MISMATCH',train,group,i,err);errors.append(err)
   equality.append(dict(group=group,precision='training_FP32' if train else 'deployment_matched_FP16_FP32_islands',frames=63,max_abs=max(errors),full_history_rebuilt=True,recon_SHA256=expected));assert_reset(c);del c;torch.cuda.empty_cache()
  del ref;torch.cuda.empty_cache()
 dump(ROOT/'trainable_parameters.json',parameter_record);dump(ROOT/'initialization_audit.json',dict(status='PASS',B300_SHA256=B300_SHA,common_step0_SHA256=sha(ROOT/'checkpoints/common_B300_step0.pt'),equality=equality,LoRA_gain=.5,LoRA_merged_or_removed=False,source52keys_overlay_last=True,time=time.time()))
 c=construct(GROUPS[2],True,rank);mathcheck=checks(c['device']);a=torch.randn(1,18,16,16,device=c['device']);b=torch.randn_like(a);fl=torch.zeros(1,2,16,16,device=c['device']);O=torch.ones(1,1,16,16,device=c['device'],dtype=torch.bool);conf=torch.ones_like(O,dtype=torch.float32)
 scale=torch.ones((),device=c['device'],requires_grad=True);v,_=latent_loss(a*scale,b*scale,fl,O,conf);grad=torch.autograd.grad(v,scale)[0];assert abs(float(grad))<1e-5
 high,_=latent_loss(a,b,fl,O,conf);low,_=latent_loss(a,b,fl,O,conf*.01);assert abs(float(low/high)-.01)<1e-6
 empty,_=latent_loss(a,b,fl,torch.zeros_like(O),conf);assert empty is None
 # Main-history object/value/gradient references survive both auxiliary sides.
 assert_reset(c);_,z1=reconstruct(c,data['features'][49],data['info']['actual_qps'][49],True,True);snap=cache_snapshot(c);gn=[None if obj is None else obj.grad_fn for obj,val in snap.values()]
 with torch.no_grad():target=auxiliary(c,data['features'][50],data['info']['actual_qps'][50])
 shifted=auxiliary(c,torch.roll(data['features'][50],(0,1),(-2,-1)),data['info']['actual_qps'][50]);assert_cache(c,snap);assert [None if obj is None else obj.grad_fn for obj,val in snap.values()]==gn
 g=torch.autograd.grad((shifted-torch.roll(target,(0,8),(-2,-1)))[...,64:-64,64:-64].abs().mean(),c['params'],allow_unused=True)
 norms={k:norm([v for (n,p),v in zip(c['named'],g) if __import__('experiment_model').category(n)==k]) for k in ['Bridge','TCG','LoRA']};assert norms['Bridge']>0 and norms['LoRA']>0 and norms['TCG']==0
 after=reconstruct(c,data['features'][50],data['info']['actual_qps'][50],True)
 # Rebuild exactly the same history without auxiliary calls and compare the next main frame.
 assert_reset(c);reconstruct(c,data['features'][49],data['info']['actual_qps'][49],True);plain=reconstruct(c,data['features'][50],data['info']['actual_qps'][50],True);assert torch.equal(after,plain);assert_reset(c)
 controls=[];ratios=[];table=load(ROOT/'plans/auxiliary_shifts.json')
 for row in rows:
  wait_rows([row]);_,path,rp=cache_paths(row,row['external_qp'],True);assert sha(path)==load(rp)['cache_sha256'];d=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
  with torch.no_grad():
   for i in [52,56,60,63]:
    f=d['features'][i];q=d['info']['actual_qps'][i];delta=table[row['sample_key']][str(i)];base=auxiliary(c,f,q);errs={}
    for magnitude in [0,1,2]:
     dy,dx=[magnitude*x for x in delta];pred=auxiliary(c,torch.roll(f,(dy,dx),(-2,-1)),q);tar=torch.roll(base,(8*dy,8*dx),(-2,-1))
     for crop in [16,32,64]:
      val=float((pred-tar).abs()[...,crop:-crop,crop:-crop].mean());assert math.isfinite(val)
      if magnitude==0:assert val==0
      errs[(magnitude,crop)]=val;controls.append(dict(sample_key=row['sample_key'],QP=row['external_qp'],frame=i,feature_dy=dy,feature_dx=dx,RGB_dy=8*dy,RGB_dx=8*dx,crop_RGB=crop,error=val,padding='unchanged_original',no_history=True))
    assert errs[(1,64)]>0;ratios.append(errs[(2,64)]/errs[(1,64)])
 write_csv(ROOT/'equivariance_controls.csv',controls)
 dump(ROOT/'audits/H2_gate.json',dict(status='DIAGNOSTIC_ONLY_USER_DIRECTED_TRAINING',median_16px_to_8px_error_ratio=statistics.median(ratios),ratios=ratios,unchanged_padding=True,crop64_not_guarantee_boundary_eliminated=True,H2_training_allowed=True,decision_authority='explicit user instruction',time=time.time()))
 dump(ROOT/'audits/preflight.json',dict(status='PASS',initialization='initialization_audit.json',flow=mathcheck,confidence_low_high_loss_ratio=float(low/high),common_scale_gradient=float(grad),empty_region_is_missing=True,auxiliary_gradients=norms,main_cache_identity_value_graph_unchanged=True,next_main_forward_exactly_unchanged=True,H2_gate='audits/H2_gate.json',time=time.time()))
 print('H_PREFLIGHT_PASS H2_CONTROLS_DIAGNOSTIC',statistics.median(ratios),flush=True)
if __name__=='__main__':main()
