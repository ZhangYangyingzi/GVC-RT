"""Ten frozen training windows, unclipped per-component gradients, no updates."""
from common_s2 import *
import torch,statistics

def main(rank=1):
 env();torch.cuda.set_device(rank)
 if (ROOT/'gradient_calibration.json').exists():assert load(ROOT/'gradient_calibration.json')['status']=='PASS';return
 from pipeline import wait_rows,request
 from cache_codec import cache_paths
 from experiment_model import construct,category
 from model_context import frozen_items
 from loss import backward_window,window_masks,norm
 rows=load(ROOT/'manifests/calibration_windows.json')
 for r in rows:request(r,r['external_qp'],True)
 c=construct(GROUPS[2],True,rank);initial=tensor_hash(c['pm'].state_dict().items());frozen=tensor_hash(frozen_items(c));records=[];vectors=[]
 def stats(vec):return {k:norm([v for (n,p),v in zip(c['named'],vec) if category(n)==k]) for k in ['Bridge','TCG','LoRA']}
 for row in rows:
  wait_rows([row]);_,path,rp=cache_paths(row,row['external_qp'],True);assert sha(path)==load(rp)['cache_sha256'];d=torch.load(path,map_location='cpu',weights_only=True,mmap=True);m=window_masks(d,c['device']);counts=dict(batch=1,output=int(m['output_frames']>0),latent=int(m['latent_frames']>0));assert counts['latent']
  component={};traces={};grouped={}
  for mode in ['spatial','latent','equiv','base']:
   c['optimizer'].zero_grad(set_to_none=True);trace=backward_window(c,d,counts,dict(lambda_lat=0.,lambda_equiv=0.),mode,masks=m)
   v=[p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p,device='cpu') for p in c['params']];component[mode]=v;grouped[mode]=stats(v)
   assert math.isfinite(grouped[mode]['Bridge']) and grouped[mode]['Bridge']>0
   if mode=='latent':assert grouped[mode]['TCG']==grouped[mode]['LoRA']==0
   if mode=='equiv':assert grouped[mode]['TCG']==0 and grouped[mode]['LoRA']>0
   if mode=='base':assert grouped[mode]['TCG']>0 and grouped[mode]['LoRA']>0
   if mode=='latent':traces=trace
  def angle(v,w,k):
   pairs=[(a,b) for (name,p),a,b in zip(c['named'],v,w) if category(name)==k];den=norm([a for a,b in pairs])*norm([b for a,b in pairs]);return sum(float((a.double()*b.double()).sum()) for a,b in pairs)/den if den else None
  cosines={loss:{k:angle(component['spatial'],component[loss],k) for k in ['Bridge','TCG','LoRA']} for loss in ['latent','equiv','base']}
  records.append(dict(sample_key=row['sample_key'],QP=row['external_qp'],norms=grouped,gradient_cosines_to_spatial=cosines,ratio_lat=grouped['spatial']['Bridge']/grouped['latent']['Bridge'],ratio_equiv=grouped['spatial']['Bridge']/grouped['equiv']['Bridge'],normalization=traces['normalization'],latent_trace=traces['trace']))
  vectors.append(component);dump(ROOT/'audits/calibration_progress.json',dict(status='RUNNING',windows=len(records),expected=10,time=time.time()));assert tensor_hash(frozen_items(c))==frozen
 lambda_lat=.1*statistics.median(a['ratio_lat'] for a in records);lambda_equiv=.1*statistics.median(a['ratio_equiv'] for a in records);assert all(math.isfinite(v) and v>0 for v in [lambda_lat,lambda_equiv])
 for rec,vec in zip(records,vectors):
  rec['composite_gradients']={}
  for group in GROUPS:
   total=[base+(lambda_lat*lat if group in GROUPS[1:] else 0)+(lambda_equiv*eq if group==GROUPS[2] else 0) for base,lat,eq in zip(vec['base'],vec['latent'],vec['equiv'])];n=norm(total);clip=min(1.,1/(n+1e-6));rec['composite_gradients'][group]=dict(unclipped=stats(total),all_parameter_norm=n,clip_multiplier=clip,clipped=stats([x*clip for x in total]))
 assert initial==tensor_hash(c['pm'].state_dict().items()) and not c['optimizer'].state
 output=dict(status='PASS',lambda_lat=lambda_lat,lambda_equiv=lambda_equiv,rule='.1*median(per-window unclipped Bridge spatial/new gradient norm)',windows=records,full_window_equiv_calls=4,equiv_not_divided_by15=True,updates=0,unclipped_ratios=True,development_or_test_used=False,time=time.time())
 dump(ROOT/'gradient_calibration.json',output);protocol=load(ROOT/'protocol.json');protocol['new_losses'].update(lambda_lat=lambda_lat,lambda_equiv=lambda_equiv);protocol['status']='FROZEN';dump(ROOT/'protocol.json',protocol)
 print('CALIBRATION_PASS',lambda_lat,lambda_equiv,flush=True)
if __name__=='__main__':main(0)
