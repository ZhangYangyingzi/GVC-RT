from util import *
import argparse,gc,torch
from model import DeformableHistory,bridge,decode
from objective import quality_models,spatial,temporal,LAMBDA_TEMP
GROUPS=['A_zero_base','B_flow_guided'];MILESTONES=[0,50,100,200,300]

def frozen(c):return thash(c['pm'].state_dict().items())+thash(c['im'].state_dict().items())
def versions(c):return {n:p._version for n,p in c['pm'].named_parameters()}
def guard(c,v):assert all(not p.requires_grad and p.grad is None and p._version==v[n] for n,p in c['pm'].named_parameters()),'FROZEN_MUTATION'
def state_save(folder,m,opt,step,seconds):
 rec=dict(model=m.state_dict(),optimizer=opt.state_dict(),committed_updates=step,group=m.group,RNG=rng(),plan_cursor=step,protocol_sha256=check_protocol(),implementation={n:sha(ROOT/n) for n in ['model.py','train.py','objective.py','restore.py','reference.py','util.py']},initialization_sha256=sha(ROOT/'checkpoints/common_initialization.pt'),statistics_sha256=sha(ROOT/'cache/evidence_stats.pt'),training_plan_sha256=sha(ROOT/'training_plan.json'),training_seconds=seconds)
 save(folder/'latest.pt',rec)
 if step in [0,1,10,50,100,200,300]:save(folder/f'step_{step:04d}.pt',rec)
 heartbeat(m.group,'TRAINING',committed_updates=step,training_seconds=seconds)

def run_segment(c,m,quality,s,backward=True):
 d=oldtrain(s['id']);r=sidewait(s['id'],m.group);start=s['start'];g=c['models']['generator'];g.reset_temporal_buffer();m.reset();ys=[];xs=[];acts=[]
 for i in range(1,start+4):
  enabled=i>=start
  with torch.set_grad_enabled(enabled):
   with torch.no_grad():l=bridge(c,d['features'][i].cuda().float())
   back=torch.from_numpy(r['decoded_ref_backward'][i]).cuda();e=torch.from_numpy(r['decoded_ref_evidence'][i]).cuda();lp,a=m(l,back,e,d['actual_qps'][i],d['id'],i);y=decode(c,lp,d['actual_qps'][i],enabled)
   if enabled:ys.append(y);xs.append(d['GT'][i].permute(2,0,1)[None].cuda().float()/255);acts.append(a)
 sp=torch.stack([spatial(quality,y,x) for y,x in zip(ys,xs)]).mean();temp=temporal(ys,xs,d['flow'][start:start+4],d['valid'][start:start+4]);loss=sp+LAMBDA_TEMP*temp;assert torch.isfinite(loss),'NONFINITE_LOSS'
 if backward:(loss/4).backward()
 with torch.no_grad():
  a=acts[-1];off=a['offset'];mod=a['modulation'];q=a['q'];act=dict(offset_RMS=float(off.square().mean().sqrt()),offset_min=float(off.min()),offset_max=float(off.max()),offset_p95=float(off.abs().quantile(.95)),mask_mean=float(mod.mean()),mask_p05=float(mod.quantile(.05)),mask_p95=float(mod.quantile(.95)),q_mean=float(q.mean()),q_p05=float(q.quantile(.05)),q_p50=float(q.quantile(.5)),q_p95=float(q.quantile(.95)),point_valid=float(a['point_validity'].float().mean()),center_valid=float(a['v'].mean()),delta_RMS=float(a['delta'].square().mean().sqrt()),gated_correction_RMS=float((a['corrected_l']-a['original_l']).square().mean().sqrt()))
 for n in ['tcg_low','tcg_mid','tcg_out']:
  z=getattr(g,n)
  if z._previous is not None:z._previous=z._previous.detach()
 g.set_temporal_cascade_grad(False);return dict(L_sp=float(sp.detach()),L_temp=float(temp.detach()),loss=float(loss.detach()),activation=act)

def gradient_groups(m):
 out={}
 for n,p in m.named_parameters():
  k='offset_head' if n.startswith('offset.4') else 'offset_upstream' if n.startswith('offset') else 'DCN' if n.startswith('dcn') else 'q_head' if n.startswith('fusion.reliability.2') else 'q_upstream' if n.startswith('fusion.reliability') else 'output_projection' if n.startswith('fusion.adapter.4') else 'fusion_upstream'
  out[k]=out.get(k,0.)+(0 if p.grad is None else float(p.grad.detach().double().square().sum()))
 return {k:v**.5 for k,v in out.items()}
def diagnose(c,m,q,step,folder):
 state=rng();m.zero_grad(set_to_none=True);rows=[run_segment(c,m,q,s) for s in plan()['joint'][0]];gr=gradient_groups(m);assert all(__import__('math').isfinite(v) for v in gr.values())
 if step==0:assert gr['output_projection']>0 and all(v==0 for k,v in gr.items() if k!='output_projection'),'ZERO_HEAD_INITIAL_GRADIENT_CHECK'
 if step>=10:assert all(gr[k]>0 for k in ['offset_head','offset_upstream','DCN','fusion_upstream','q_head','q_upstream']),'UPSTREAM_GRADIENT_MISSING'
 dump(folder/f'diagnostic_{step:04d}.json',dict(step=step,gradient_groups=gr,samples=rows,fixed_samples=plan()['joint'][0]));m.zero_grad(set_to_none=True);setrng(state)

def wait_evaluation(group,step):
 p=ROOT/'evaluation/milestones'/group/f'{step:04d}.json'
 while not p.exists():heartbeat(group,'WAITING_FOR_EVALUATION',committed_updates=step,waiting_for=str(p));time.sleep(3)
 r=load(p);assert r['status']=='PASS';return r

def run(group):
 setup();from restore import restore_b300
 c=restore_b300();torch.use_deterministic_algorithms(True,warn_only=True);base=frozen(c);v=versions(c);seed();st=stats();m=DeformableHistory(st,group).cuda();folder=ROOT/'checkpoints'/group;folder.mkdir(exist_ok=True)
 with lock('common_initialization.lock',True):
  p=ROOT/'checkpoints/common_initialization.pt'
  if p.with_suffix('.pt.json').exists():assert thash(readtensor(p)['model'].items())==thash(m.state_dict().items()),'AB_INITIALIZATION_MISMATCH'
  else:save(p,dict(model={n:t.cpu() for n,t in m.state_dict().items()},seed=20261113))
 opt=torch.optim.AdamW(m.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=1e-4);quality=quality_models();step=0;seconds=0.;latest=folder/'latest.pt';stop=None
 if latest.with_suffix('.pt.json').exists():
  ck=readtensor(latest);assert ck['protocol_sha256']==check_protocol() and ck['group']==group;assert ck['implementation']=={n:sha(ROOT/n) for n in ck['implementation']},'IMPLEMENTATION_CHANGED';m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);setrng(ck['RNG']);step=ck['committed_updates'];seconds=ck['training_seconds']
 else:
  dump(folder/'initialization.json',dict(parameter_count=sum(p.numel() for p in m.parameters()),state_sha256=thash(m.state_dict().items()),frozen_sha256=base,pretrained_T_checkpoint_loaded=False));state_save(folder,m,opt,0,0.);diagnose(c,m,quality,0,folder)
 if step in MILESTONES and step<300:
  decision=wait_evaluation(group,step)
  if decision.get('stop'):stop=decision
 while step<300 and stop is None:
  batch=plan()['joint'][step]
  for s in batch:sidewait(s['id'],group)
  heartbeat(group,'TRAINING',committed_updates=step);torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True);rows=[run_segment(c,m,quality,s) for s in batch];grad=gradient_groups(m)
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()),'NONFINITE_GRADIENT';clip=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step();guard(c,v);torch.cuda.synchronize();seconds+=time.perf_counter()-t;step+=1;state_save(folder,m,opt,step,seconds)
  event('UPDATE_COMMITTED',group=group,step=step,L_sp=sum(r['L_sp'] for r in rows)/4,L_temp=sum(r['L_temp'] for r in rows)/4,loss=sum(r['loss'] for r in rows)/4,gradient_groups=grad,clip_before=clip,training_seconds=seconds,activation=rows[-1]['activation'])
  if step in [1,10,50,100,200,300]:diagnose(c,m,quality,step,folder)
  if step in MILESTONES:
   assert frozen(c)==base,'FROZEN_HASH_CHANGED';decision=wait_evaluation(group,step)
   if decision.get('stop'):stop=decision
 assert frozen(c)==base;reason='PARETO_STOP' if stop is not None else 'BUDGET_300';dump(folder/'completion.json',dict(status='COMPLETE',committed_updates=step,stop_reason=reason,stop_evidence=stop,frozen_before_after_sha256=base,training_seconds=seconds));heartbeat(group,'COMPLETE',committed_updates=step,stop_reason=reason,training_seconds=seconds)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--group',choices=GROUPS,required=True);p=a.parse_args()
 with lock(p.group+'.lock'):run(p.group)
