"""Fixed-budget paired-seed training; committed step means an atomic checkpoint exists."""
from util import *
import argparse,math,gc,statistics
import torch
import torch.nn.functional as F
from model import AlignedLatent,MotionAligner,bridge,decode
from objective import quality_models,spatial,temporal,align_loss

def frozen(c):return thash(c['pm'].state_dict().items())+thash(c['im'].state_dict().items())
def versionguard(c):return {n:p._version for n,p in c['pm'].named_parameters()}
def assertguard(c,v):
 assert all(not p.requires_grad and p.grad is None and p._version==v[n] for n,p in c['pm'].named_parameters()),'FROZEN_PARAMETER_MUTATION'
def latent(c,d,i):return bridge(c,d['features'][i].cuda().float()).detach()
def gtframe(d,i):return d['GT'][i].permute(2,0,1)[None].cuda().float()/255

def prepare(c):
 p=ROOT/'cache/statistics.pt'
 if p.with_suffix('.pt.json').exists():return readtensor(p)
 ids=plan()['stats_ids'];fs=[];ls=[];qps=[]
 with torch.no_grad():
  for id in ids:
   d=waitcache(id,'aligner')
   for i in range(1,len(d['features'])):
    f=d['features'][i].cuda().float();l=bridge(c,f);fs.append(f.square().mean((0,2,3)).cpu());ls.append(l.square().mean((0,2,3)).cpu());qps.append(d['actual_qps'][i])
 stats=dict(f_rms=torch.stack(fs).mean(0).sqrt().clamp_min(.01),l_rms=torch.stack(ls).mean(0).sqrt().clamp_min(.01),qp_mean=float(torch.tensor(qps).float().mean()),qp_std=float(torch.tensor(qps).float().std(unbiased=False)),ids=ids,train_only=True)
 save(p,stats);dump(ROOT/'audits/statistics.json',dict(ids=ids,train_only=True,frames=len(qps),floor=.01,f_rms=stats['f_rms'].tolist(),l_rms=stats['l_rms'].tolist(),qp_mean=stats['qp_mean'],qp_std=stats['qp_std'],statistics_sha256=sha(p)));return stats

def rollout(c,m,d,start=1,end=None,grad=False):
 g=c['models']['generator'];g.reset_temporal_buffer();m.reset();end=len(d['features']) if end is None else end;ys=[];xs=[];aux=[]
 for i in range(1,end):
  enabled=grad and i>=start
  with torch.set_grad_enabled(enabled):
   f=d['features'][i].cuda().float()
   with torch.no_grad():l=bridge(c,f)
   lp,a=m(l,f,d['actual_qps'][i],d['id'],i);y=decode(c,lp,d['actual_qps'][i],enabled)[...,:d['native_shape'][0],:d['native_shape'][1]]
   if i>=start:ys.append(y);xs.append(gtframe(d,i));aux.append(a)
 return ys,xs,aux

def segment(c,m,q,s,backward=True):
 d=waitcache(s['id'],m.group);start=s['start'];ys,xs,aux=rollout(c,m,d,start,start+4,True)
 sp=torch.stack([spatial(q,y,x) for y,x in zip(ys,xs)]).mean();temp=temporal(ys,xs,d['flow'][start:start+4],d['valid'][start:start+4]);als=[]
 if m.aligner is not None:
  for j,a in enumerate(aux):
   if a['flows'] is not None:als.append(align_loss(a['flows'],d['flow'][start+j].cuda(),d['valid'][start+j].cuda())[0])
 al=torch.stack(als).mean() if als else sp.new_zeros(())
 lam=load(ROOT/'audits/calibration.json')['lambda_temp'];total=sp+lam*temp+.1*al;assert torch.isfinite(total),'NONFINITE_LOSS'
 if backward:(total/4).backward()
 with torch.no_grad():
  a=aux[-1];l=a['original_l'];lp=a['corrected_l'];act=dict(q_mean=float(a['q'].mean()),valid=float(a['v'].mean()),u_RMS=float(a['u'].square().mean().sqrt()),delta_RMS=float(a['delta'].square().mean().sqrt()),l_RMS=float(l.square().mean().sqrt()),latent_mean=l.mean((0,2,3)).tolist(),corrected_latent_mean=lp.mean((0,2,3)).tolist(),latent_std=l.std((0,2,3),unbiased=False).tolist(),corrected_latent_std=lp.std((0,2,3),unbiased=False).tolist())
 # Detach all candidate TCG state only after the complete short-segment backward.
 g=c['models']['generator'];g.set_temporal_cascade_grad(False)
 for n in ['tcg_low','tcg_mid','tcg_out']:
  z=getattr(g,n)
  if z._previous is not None:z._previous=z._previous.detach()
 return dict(loss=float(total.detach()),L_sp=float(sp.detach()),L_temp=float(temp.detach()),L_align=float(al.detach()),activation=act)

def calibrate(c,quality):
 p=ROOT/'audits/calibration.json'
 if p.exists():return load(p)
 rows=[];g=c['models']['generator']
 for s in plan()['calibration']:
  d=waitcache(s['id'],'aligner');start=s['start'];g.reset_temporal_buffer();ys=[];xs=[]
  with torch.no_grad():
   for i in range(1,start+4):
    y=decode(c,latent(c,d,i),d['actual_qps'][i])
    if i>=start:ys.append(y.detach().requires_grad_());xs.append(gtframe(d,i))
  sp=torch.stack([spatial(quality,y,x) for y,x in zip(ys,xs)]).mean();tm=temporal(ys,xs,d['flow'][start:start+4],d['valid'][start:start+4]);gs=torch.autograd.grad(sp,ys);gt=torch.autograd.grad(tm,ys)
  ns=sum(float(t.double().square().sum()) for t in gs)**.5;nt=sum(float(t.double().square().sum()) for t in gt)**.5
  assert math.isfinite(ns) and math.isfinite(nt) and ns>0 and nt>0,'INVALID_LAMBDA_CALIBRATION'
  rows.append(dict(sample=s,g_sp=ns,g_temp=nt,ratio=ns/nt));heartbeat('aligner','CALIBRATING',completed=len(rows))
 lam=.1*statistics.median(r['ratio'] for r in rows);assert math.isfinite(lam) and lam>0
 r=dict(lambda_temp=lam,rule='0.1*median(g_sp/g_temp)',differentiated_wrt='four reconstructed RGB output tensors, all permitted temporal endpoints',B300_unupdated=True,train_only=True,samples=rows);dump(p,r);return r

def initialize(stats):
 p=ROOT/'checkpoints/common_fusion.pt'
 if not p.with_suffix('.pt.json').exists():
  seed();m=AlignedLatent(stats,'S_current');save(p,dict(fusion=m.fusion.state_dict(),seed=20261112))
 return readtensor(p)['fusion']
def state_save(path,m,opt,step,phase,extra=None):
 rec=dict(model=m.state_dict(),optimizer=opt.state_dict(),committed_updates=step,phase=phase,RNG=rng(),plan_cursor=step,protocol_sha256=check_protocol(),training_plan_sha256=sha(ROOT/'training_plan.json'),statistics_sha256=sha(ROOT/'cache/statistics.pt'),pretrained_aligner_sha256=None if phase=='aligner' else sha(ROOT/'checkpoints/aligner_0500.pt'),implementation={n:sha(ROOT/n) for n in ['model.py','objective.py','restore.py','train.py','util.py']},**(extra or {}));save(path,rec)

def aligner_evaluate(m,step):
 rows=[];m.eval()
 with torch.no_grad():
  for split in ['train','dev']:
   if split=='train':samples=[dict(s) for s in plan()['pretrain'][0]]
   else:samples=[dict(id=r['id'],frame=8,qp=5) for r in manifest()['dev']]
   for s in samples:
    d=waitcache(s['id'],'aligner',s.get('qp'));i=s['frame'];_,_,flows=m(d['features'][i].cuda().float(),d['features'][i-1].cuda().float());flow=d['flow'][i].cuda();valid=d['valid'][i].cuda();ph,pw=d['padded_shape'];h,w=d['native_shape'];flow=F.pad(flow,(0,pw-w,0,ph-h));valid=F.pad(valid,(0,pw-w,0,ph-h));loss,epe=align_loss(flows,flow,valid);rows.append(dict(step=step,split=split,id=s['id'],frame=i,Charbonnier=float(loss),**{f'EPE_{k}':v for k,v in epe.items()}))
 csvwrite(ROOT/'metrics'/f'aligner_{step:04d}.csv',rows);m.train()

def pretrain():
 setup();seed();from restore import restore_b300
 c=restore_b300();torch.use_deterministic_algorithms(True,warn_only=True);v=versionguard(c);fhash=frozen(c);stats=prepare(c);initialize(stats);q=quality_models();calibrate(c,q);assertguard(c,v)
 del q;g=c['models']['generator'];g.reset_temporal_buffer();del c;gc.collect();torch.cuda.empty_cache()
 m=MotionAligner(stats['f_rms']).cuda();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=1e-4);latest=ROOT/'checkpoints/aligner_latest.pt';step=0;elapsed=0.
 if latest.with_suffix('.pt.json').exists():
  ck=readtensor(latest);assert ck['protocol_sha256']==check_protocol() and ck['phase']=='aligner';m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);setrng(ck['RNG']);step=ck['committed_updates'];elapsed=ck.get('training_seconds',0.)
 else:state_save(ROOT/'checkpoints/aligner_0000.pt',m,opt,0,'aligner');aligner_evaluate(m,0)
 while step<500:
  batch=plan()['pretrain'][step];ds=[(s,waitcache(s['id'],'aligner')) for s in batch];heartbeat('aligner','PRETRAIN',committed_updates=step);torch.cuda.synchronize();t=time.time();opt.zero_grad(set_to_none=True);lv=0.
  for s,d in ds:
   i=s['frame'];_,_,flows=m(d['features'][i].cuda().float(),d['features'][i-1].cuda().float());loss,epe=align_loss(flows,d['flow'][i].cuda(),d['valid'][i].cuda());assert torch.isfinite(loss);(loss/4).backward();lv+=float(loss.detach())/4
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());opt.step();torch.cuda.synchronize();elapsed+=time.time()-t;step+=1
  state_save(latest,m,opt,step,'aligner',dict(training_seconds=elapsed));heartbeat('aligner','PRETRAIN',committed_updates=step,training_seconds=elapsed);event('ALIGNER_UPDATE_COMMITTED',step=step,loss=lv,training_seconds=elapsed)
  if step in [100,250,500]:state_save(ROOT/'checkpoints'/f'aligner_{step:04d}.pt',m,opt,step,'aligner',dict(training_seconds=elapsed));aligner_evaluate(m,step)
 dump(ROOT/'audits/aligner_frozen_reference.json',dict(pretrain_updates=500,initialization_sha256=sha(ROOT/'checkpoints/aligner_0500.pt'),base_frozen_sha256=fhash,training_seconds=elapsed));heartbeat('aligner','COMPLETE',committed_updates=500,training_seconds=elapsed)

def gradients(m):
 r={}
 for name,p in m.named_parameters():
  if name.startswith('aligner'):group='MotionAligner'
  elif name.startswith('fusion.reliability'):group='Reliability'
  elif name.startswith('fusion.adapter.4'):group='OutputProjection'
  else:group='AdapterUpstream'
  r[group]=r.get(group,0.)+(0 if p.grad is None else float(p.grad.detach().double().square().sum()))
 return {k:v**.5 for k,v in r.items()}
def diagnose(c,m,q,step,folder):
 state=rng();m.zero_grad(set_to_none=True);rs=[segment(c,m,q,s) for s in plan()['joint'][0]];gn=gradients(m);dump(folder/f'diagnostic_{step:04d}.json',dict(step=step,fixed_segments=plan()['joint'][0],gradient_norms=gn,samples=rs));m.zero_grad(set_to_none=True);setrng(state)

def joint(group):
 setup();seed();from restore import restore_b300
 c=restore_b300();torch.use_deterministic_algorithms(True,warn_only=True);fhash=frozen(c);guard=versionguard(c);stats=readtensor(ROOT/'cache/statistics.pt');m=AlignedLatent(stats,group).cuda();m.fusion.load_state_dict(readtensor(ROOT/'checkpoints/common_fusion.pt')['fusion'])
 if m.aligner is not None:m.aligner.load_state_dict(readtensor(ROOT/'checkpoints/aligner_0500.pt')['model'])
 params=[dict(params=m.fusion.parameters(),lr=5e-5)]
 if m.aligner is not None:params.append(dict(params=m.aligner.parameters(),lr=1e-5))
 opt=torch.optim.AdamW(params,betas=(.9,.999),eps=1e-8,weight_decay=1e-4);q=quality_models();folder=ROOT/'checkpoints'/group;folder.mkdir(exist_ok=True);latest=folder/'latest.pt';step=0;elapsed=0.
 if latest.with_suffix('.pt.json').exists():
  ck=readtensor(latest);assert ck['protocol_sha256']==check_protocol() and ck['phase']==group and ck['pretrained_aligner_sha256']==sha(ROOT/'checkpoints/aligner_0500.pt');m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);setrng(ck['RNG']);step=ck['committed_updates'];elapsed=ck['training_seconds']
 else:
  state_save(folder/'step_0000.pt',m,opt,0,group,dict(training_seconds=0.));dump(folder/'initialization.json',dict(fusion_sha256=thash(m.fusion.state_dict().items()),aligner_sha256=None if m.aligner is None else thash(m.aligner.state_dict().items()),frozen_model_sha256=fhash,trainable_parameters=[n for n,p in m.named_parameters()]));diagnose(c,m,q,0,folder)
 while step<300:
  batch=plan()['joint'][step]
  for s in batch:waitcache(s['id'],group)
  heartbeat(group,'TRAINING',committed_updates=step);torch.cuda.synchronize();t=time.time();opt.zero_grad(set_to_none=True);rs=[segment(c,m,q,s) for s in batch];gn=gradients(m)
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()),'NONFINITE_GRADIENT';norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step();assertguard(c,guard);torch.cuda.synchronize();elapsed+=time.time()-t;step+=1
  state_save(latest,m,opt,step,group,dict(training_seconds=elapsed));heartbeat(group,'TRAINING',committed_updates=step,training_seconds=elapsed,remaining_training_seconds=(300-step)*elapsed/step if step>=20 else None)
  event('JOINT_UPDATE_COMMITTED',group=group,step=step,loss=sum(r['loss'] for r in rs)/4,L_sp=sum(r['L_sp'] for r in rs)/4,L_temp=sum(r['L_temp'] for r in rs)/4,L_align=sum(r['L_align'] for r in rs)/4,gradient_norms=gn,clip_before=norm,training_seconds=elapsed)
  if step in [1,10,50,100,300]:diagnose(c,m,q,step,folder)
  if step in [50,100,200,300]:
   assert frozen(c)==fhash,'FROZEN_HASH_CHANGED';state_save(folder/f'step_{step:04d}.pt',m,opt,step,group,dict(training_seconds=elapsed))
 assert frozen(c)==fhash;assert all(p.grad is None and not p.requires_grad for model in q for p in model.parameters());dump(folder/'freeze_audit.json',dict(status='PASS',before_after_sha256=fhash,updates=step,training_seconds=elapsed));heartbeat(group,'COMPLETE',committed_updates=300,training_seconds=elapsed)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['aligner','joint'],required=True);a.add_argument('--group',choices=['S_current','U_unaligned','T_aligned']);p=a.parse_args()
 with lock((p.group or 'aligner')+'.lock'):
  if p.phase=='aligner':pretrain()
  else:joint(p.group)
