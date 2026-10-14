"""Exactly one isolated AdamW step per fixed case; full causal replay."""
from audit31 import *
import argparse,fcntl,math,random,resource,traceback
def main(t,gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch,numpy as np
 import torch.nn.functional as F
 from adapter31 import backend,official_model
 from color31 import color
 settings=backend();start=time.time();torch.cuda.reset_peak_memory_stats()
 lf=receipt(t).with_suffix('.lock').open('a');fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if valid(t):return
 cfg=load(ROOT/'config.json');seed=cfg['seed_probe'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 rng=torch.get_rng_state();crng=torch.cuda.get_rng_state()
 row=next(r for r in load(ROOT/'probe_manifest.json')['updates'] if r['calibration_index']==t['video']);assert row['external_qp']==t['qp']
 li=legacy_io();v4,im,pm,models,quality=li.load_training(torch.device('cuda:0'))
 def oldload(p):
  p=Path(p)
  if p.name.startswith('train_manifest_'):return load(V29/p.name)
  return load(p)
 stat_loss=module('stat_loss',V29/'stat_loss.py')
 wt=legacy_code('window','window_training.py',dict(load=oldload));frames=wt.replay(row,'B_covstats',torch.device('cuda:0'))
 common=li.module('v631_frame_qp',li.V62/'fullqp_common.py')
 params=[p for k,m in models.items() if k!='wrapper' for p in m.parameters()];counts={k:len(list(models[k].parameters())) for k in ('bridge','generator')}
 named=[(k,n,p) for k in ('bridge','generator') for n,p in models[k].named_parameters()]
 assert len(params)==len(named) and all(p.dtype==torch.float32 and p.requires_grad for p in params)
 initial={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm);qh=[v4.module_hash(m) for m in quality]
 assert core==load(ROOT/'evaluation_config.json')['compression_hash']
 c=int(t['method'][1:]);lam=(0.,LAM,.5*LAM,2*LAM,LAM,LAM)[c];lrs=(6e-6,1e-6) if c==4 else (3e-6,5e-7)
 oc=cfg['optimizer_effective'];opt=torch.optim.AdamW([dict(name=k,params=list(models[k].parameters()),lr=lr) for k,lr in zip(('bridge','generator'),lrs)],betas=tuple(oc['betas']),eps=oc['eps'],weight_decay=1e-4)
 assert not opt.state and {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in params}
 optimizer_members=[dict(module=k,name=n,parameter_ID=id(p),dtype=str(p.dtype),shape=list(p.shape),lr=lrs[0 if k=='bridge' else 1]) for k,n,p in named]
 official=official_model();official_hash=tensor_hash(official.state_dict())
 captured=[];hook=pm.dec.register_forward_hook(lambda _m,_a,o:captured.append(o))
 def ref(rec):
  assert len(captured)==1
  with torch.no_grad():pm.add_ref_frame(captured.pop().detach(),(rec.detach()*2-1).half())
  assert not pm.dpb[0].feature.requires_grad and not pm.dpb[0].frame.requires_grad
 def norm(gs):return sum(float(g.double().square().sum()) for g in gs if g is not None)**.5
 def dot(gs,hs):return sum(float((g.double()*h.double()).sum()) for g,h in zip(gs,hs))
 def zero():return [torch.zeros_like(p) for p in params]
 def add(dst,gs,scale=1.):
  for b,g in zip(dst,gs):
   if g is not None:
    value=g.detach().float()/scale;assert torch.isfinite(value).all();b.add_(value)
 def summarize(gs):
  rows={};off=0
  for k,n in counts.items():
   a=gs[off:off+n];off+=n
   rows[k]=dict(norm=norm(a),connected_parameters=sum(g is not None for g in a),nonzero_parameters=sum(g is not None and bool((g!=0).any()) for g in a),finite=all(g is None or bool(torch.isfinite(g).all()) for g in a))
  return rows
 component_names=('LPIPS','DISTS','reconstruction_L1','stat','proxy_L1','rate')
 weights=dict(LPIPS=1.,DISTS=.5,reconstruction_L1=REC,stat=lam,proxy_L1=.01,rate=common.beta_q('schedule_s1p0',row['external_qp']))
 component=[zero() for _ in range(4)];base_ref=zero();gradient_rows=[];deep=[];frame_grad_connection={n:[False]*len(params) for n in component_names}
 def replay(grads):
  torch.set_rng_state(rng);torch.cuda.set_rng_state(crng);pm.clear_dpb();pm.set_curr_poc(0);captured.clear();lossrows=[];feature_rec=[];feature_gt=[]
  with torch.no_grad():pm.add_ref_frame(None,im.compress(v4.codec_input(frames[0]),row['external_qp'])['x_hat'])
  for pos in range(1,64):
   qa=common.frame_qp(pm,row['external_qp'],pos);gt=frames[pos]
   context=torch.enable_grad() if grads and pos>=49 else torch.no_grad()
   with context:
    proxy=models['wrapper'](gt);rec,rate,_=v4.joint_ste_forward(pm,proxy,qa)
    if pos>=49:
     lp=quality[0](rec,gt,normalize=True).mean()
     target=gt.detach().requires_grad_(grads)
     ds,st,terms,shapes=stat_loss.dists_and_stat(quality[1],rec,target)
     vals=dict(LPIPS=lp,DISTS=ds,reconstruction_L1=F.l1_loss(rec,gt),stat=st,proxy_L1=F.l1_loss(proxy,gt),rate=rate/(256*256))
     assert all(torch.isfinite(x) for x in vals.values())
     total=sum(vals[n]*weights[n] for n in component_names)
     if grads:
      # The exact historical base backprop is an independent reference; no optimizer step.
      baseloss=lp+.5*ds+weights['rate']*vals['rate']+.01*vals['proxy_L1']
      gb=torch.autograd.grad(baseloss/15,params,retain_graph=True,allow_unused=True);add(base_ref,gb)
      gr=torch.autograd.grad(vals['reconstruction_L1']*(4096/15),params,retain_graph=True,allow_unused=True)
      add(base_ref,[None if g is None else g*REC for g in gr],4096)
      for j,n in enumerate(component_names):
       val=vals[n];scale=4096. if n in ('reconstruction_L1','stat') else 1.
       if val.requires_grad:
        gs=torch.autograd.grad(val*(scale/15),[*params,rec],retain_graph=True,allow_unused=True)
        gs=[None if g is None else g.detach().float()/scale for g in gs];gpar,gimg=gs[:-1],gs[-1]
       else:gpar=[None]*len(params);gimg=None
       for z,g in enumerate(gpar):frame_grad_connection[n][z]|=g is not None
       expected=n in ('proxy_L1','rate')
       if n in ('LPIPS','DISTS','reconstruction_L1','stat'):
        assert gimg is not None and norm([gimg])>0 and torch.isfinite(gimg).all()
        assert all(x['norm']>0 and x['finite'] for x in summarize(gpar).values())
       if n=='proxy_L1':assert all(g is None for g in gpar)
       if j<4:add(component[j],gpar)
       gradient_rows.append(dict(case=t['method'],calibration_index=t['video'],domain=row['domain'],QP=t['qp'],frame=pos,component=n,requires_grad=val.requires_grad,grad_fn=None if val.grad_fn is None else type(val.grad_fn).__name__,weight=weights[n],normalization_denominator=15,backward_scale=scale,scale_restored_dtype='FP32',raw_image_gradient_norm=norm([gimg]),weighted_image_gradient_norm=norm([gimg])*abs(weights[n]),raw_global_gradient_norm=norm(gpar),weighted_global_gradient_norm=norm(gpar)*abs(weights[n]),modules=summarize(gpar),status='EXPECTED_FROZEN_OR_INDEPENDENT' if expected and all(g is None for g in gpar) else 'CONNECTED',basis='P frozen, proxy independent of receiver; rate depends on frozen compression path' if expected else 'end-to-end receiver and reconstruction input'))
      if pos==49:
       gg=torch.autograd.grad(st*(4096/15),target,retain_graph=True,allow_unused=True)[0];assert gg is None
       gd=legacy_code('deep','gradient_tools.py');score,layers,_=gd.dists_details(quality,rec,gt,True)
       assert all(a['connected_to_output'] and a['output_gradient']['finite'] and a['output_gradient']['norm']>0 for a in layers if a['kind']=='deep');deep.extend(layers)
     with torch.no_grad():
      feature_rec.append(official(rec).cpu().numpy());feature_gt.append(official(gt).cpu().numpy())
      lossrows.append(dict(frame=pos,actual_qp=qa,**{n:float(x.detach()) for n,x in vals.items()},total_loss=float(total.detach()),**color(rec.detach(),gt)))
   ref(rec)
  assert len(lossrows)==15
  return lossrows,np.concatenate(feature_gt),np.concatenate(feature_rec)
 before,real,init_features=replay(True)
 # Raw gradients have exactly one 15-frame division; weighting happens once below.
 base=zero()
 for name,buff in zip(component_names,component):
  if name!='stat':
   for g,a in zip(base,buff):g.add_(a,alpha=weights[name])
 isolated_combination_error=norm([a-b for a,b in zip(base,base_ref)])/max(norm(base_ref),1e-12)
 # The historical implementation backpropagates LPIPS+.5*DISTS+rate+proxy jointly.
 # Separate diagnostic backwards are not distributive across native FP16 boundaries.
 # Preserve that exact fused base, followed by the independently restored rec term.
 base=base_ref
 rel=norm([a-b for a,b in zip(base,base_ref)])/max(norm(base_ref),1e-12);assert rel<=1e-6
 stat=component[3];total=[a+b*lam for a,b in zip(base,stat)]
 for p,g in zip(params,total):p.grad=g.clone()
 synth=norm([p.grad-(a+b*lam) for p,a,b in zip(params,base,stat)])/max(norm(total),1e-12);assert synth<=1e-6
 preclip=norm(total)
 if c!=5:reported=float(torch.nn.utils.clip_grad_norm_(params,1.));coefficient=min(1.,1./(reported+1e-6))
 else:reported=preclip;coefficient=1.
 afterclip=norm([p.grad for p in params]);assert math.isfinite(afterclip)
 # Independent first-step AdamW reference. Do not land an extra optimizer update.
 originals=[p.detach().clone() for p in params];expected=[]
 for (k,n,p),old in zip(named,originals):
  lr=lrs[0 if k=='bridge' else 1];g=p.grad
  m=g*(1-oc['betas'][0]);v=g.square()*(1-oc['betas'][1]);den=v.sqrt()/(1-oc['betas'][1])**.5+oc['eps']
  value=old*(1-lr*1e-4)-(lr/(1-oc['betas'][0]))*m/den;expected.append(value)
 opt.step()  # Exactly one landed temporary update, never exported as a model.
 dump(ROOT/'raw/probes'/(uid(t)+'.landed.json'),dict(temporary_updates=1,task=t,time=time.time(),must_not_repeat_optimizer_step=True))
 update=[];deltas={};offset=0;max_error=0.
 for k,count in counts.items():
  ps=params[offset:offset+count];olds=originals[offset:offset+count];exp=expected[offset:offset+count]
  changes=[p.detach()-o for p,o in zip(ps,olds)];err=max(float((p.detach()-e).abs().max()) for p,e in zip(ps,exp));max_error=max(max_error,err)
  assert err<=3e-7,('AdamW predicted value',k,err)
  dl=norm(changes);pn=norm(olds);assert dl>0 and all(torch.isfinite(p).all() for p in ps)
  update.append(dict(case=t['method'],calibration_index=t['video'],domain=row['domain'],QP=t['qp'],module=k,delta_L2=dl,delta_maxabs=max(float(x.abs().max()) for x in changes),relative_parameter_norm=dl/max(pn,1e-12),changed_elements=sum(int((x!=0).sum()) for x in changes),parameter_elements=sum(p.numel() for p in ps),AdamW_expected_maxabs_error=err,preclip_norm=preclip,postclip_norm=afterclip,clip_coefficient=coefficient,lambda_stat=lam,lr=lrs[0 if k=='bridge' else 1],effective_optimizer_members=optimizer_members[offset:offset+count],optimizer_update_count=1))
  deltas[k]=torch.cat([x.flatten() for x in changes]).cpu().numpy();offset+=count
 for j,n in enumerate(component_names[:4]):
  raw=component[j];weighted=[g*weights[n] for g in raw];rest=[a-b for a,b in zip(total,weighted)];cos=dot(weighted,rest)/max(norm(weighted)*norm(rest),1e-30)
  gradient_rows.append(dict(case=t['method'],calibration_index=t['video'],domain=row['domain'],QP=t['qp'],frame='aggregate',component=n,weight=weights[n],raw_global_gradient_norm=norm(raw),weighted_global_gradient_norm=norm(weighted),cosine_with_other_base=cos,modules=summarize(raw),connected_parameters=sum(frame_grad_connection[n]),normalization_denominator=15,backward_scale=4096 if j>=2 else 1,scale_restored_dtype='FP32',status='CONNECTED'))
 after,real_after,after_features=replay(False);hook.remove();assert np.array_equal(real,real_after)
 assert initial['wrapper']==v4.module_hash(models['wrapper']) and core==v4.compression_hash(im,pm) and qh==[v4.module_hash(m) for m in quality] and official_hash==tensor_hash(official.state_dict())
 art={};basepath=ROOT/'raw/probes'/uid(t)
 for suffix,data in (('.gradients.csv',gradient_rows),('.updates.csv',update),('.before.csv',before),('.after.csv',after)):
  p=basepath.with_suffix(suffix);write(p,data);art[str(p)]=sha(p)
 p=ROOT/'features/probes'/(uid(t)+'.npz');save_npz(p,real=real,INIT=init_features,UPDATED=after_features);art[str(p)]=sha(p)
 dp=ROOT/'clones/deltas'/(uid(t)+'.npz');save_npz(dp,**deltas);art[str(dp)]=sha(dp)
 dump(receipt(t),dict(status='PASS',task=t,signature=signature(t),artifacts=art,temporary_optimizer_updates=1,formal_updates=0,D_updates=0,source_step=1000,optimizer_fresh=True,lambda_stat=lam,weights=weights,base_synthesis_relative_L2=rel,isolated_component_sum_vs_fused_base_relative_L2=isolated_combination_error,base_gradient_path='historical fused base/15 + separately restored 4096 reconstruction gradient',total_synthesis_relative_L2=synth,AdamW_expected_maxabs_error=max_error,GT_stat_gradient_is_None=True,deep_DISTS=deep,frozen_hashes=dict(wrapper=initial['wrapper'],compression=core,quality=qh),module_hashes_before=initial,module_hashes_after={k:v4.module_hash(m) for k,m in models.items()},model_modes={k:m.training for k,m in models.items()},core_modes=dict(I=im.training,P=pm.training),optimizer_effective=oc,replay_rng_restored=True,input_plan=row,actual_qps=[x['actual_qp'] for x in before],feature_path=str(p),delta_path=str(dp),backend=settings,physical_GPU=gpu,seconds=time.time()-start,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024));print('DONE',uid(t),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--gpu',required=True,type=int,choices=(4,5,6,7));a=ap.parse_args()
 try:main(json.loads(a.task),a.gpu)
 except Exception:error(dict(task=json.loads(a.task),traceback=traceback.format_exc()));raise
