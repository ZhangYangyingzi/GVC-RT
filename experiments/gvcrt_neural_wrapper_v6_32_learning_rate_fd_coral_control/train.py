"""A/B exact C1 independent continuation; auxiliary statistics never alter crop updates."""
from io32 import *
import argparse,fcntl,resource,math,traceback
def main(branch,gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch
 from training32 import setup,seed,crop_c1,full_pass,official_model,rng_state,restore_rng,buffer_state,restore_buffers,norm,groups,check_frozen
 from frames32 import PlanFrames,full_eligible
 assert branch in BRANCHES[:2], 'C/D blocked until complete fixed full-frame reference and calibration pass'
 assert load(ROOT/'audits/common.json')['status']=='PASS'
 b=ROOT/'branches'/branch;b.mkdir(exist_ok=True);lock=(b/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ctx=setup();v4,im,pm,models,quality,common,params,frozen=ctx;official=official_model();oh=tensor_hash(official.state_dict());cfg=load(ROOT/'config.json');sig=signature('train')
 seed(cfg['seed_global']);global_rng=rng_state();seed(cfg['seed']);opt=torch.optim.AdamW([dict(name=k,params=list(models[k].parameters()),lr=lr) for k,lr in zip(('bridge','generator'),cfg['branch_lrs'][branch])],betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
 assert not opt.state;latest=b/'checkpoints/latest.pt';index=load(b/'checkpoint_index.json') if (b/'checkpoint_index.json').exists() else {};start=0;visits={f'{d}:{q}':0 for d in ('ulong','uvg') for q in range(10)};observed=dict(visits)
 if latest.exists():
  s=torch.load(latest,map_location='cpu',weights_only=True);assert s['signature']==sig and s['branch']==branch and s['plan_sha256']==sha(ROOT/'plans/shared.json')
  for k,m in models.items():m.load_state_dict(s[k])
  opt.load_state_dict(s['optimizer']);restore_rng(s['crop_rng']);global_rng=s['global_rng'];start=s['adaptation_step'];visits=s['visits'];observed=s['observed'];del s
 def save(step,row=None):
  check_frozen(ctx);assert oh==tensor_hash(official.state_dict())
  s=dict(schema='v632',branch=branch,source_step=1000,adaptation_step=step,committed_optimizer_updates=step,replay_cursor=step,signature=sig,config_sha256=sha(ROOT/'config.json'),protocol_sha256=sha(ROOT/'protocol.json'),plan_sha256=sha(ROOT/'plans/shared.json'),source_checkpoint_sha256=cfg['source_checkpoint']['sha256'],**{k:m.state_dict() for k,m in models.items()},optimizer=opt.state_dict(),crop_rng=rng_state(),global_rng=global_rng,visits=visits,observed=observed,EMA_status='BLOCKED_SHARED_REFERENCE',EMA_reason='Four fixed plan windows violate full-resolution view; diagnostic pre-update features preserved, no substitute seeded statistics',frozen_hashes=frozen,last_receipt=row)
  atomic_torch(latest,s)
  if step in (0,1,250,500,1000):
   p=b/'checkpoints'/f'adaptation_step_{step:04d}.pt';atomic_torch(p,s)
   e=ROOT/'evaluation/checkpoints'/f'{branch}_{step}.pt';atomic_torch(e,{k:m.state_dict() for k,m in models.items()})
   hashes={k:v4.module_hash(m) for k,m in models.items()};deploy={k:tensor_hash({n:t.half() for n,t in models[k].state_dict().items()}) for k in ('bridge','generator')}
   index[str(step)]=dict(path=str(p),sha256=sha(p),source_step=1000,adaptation_step=step,module_hashes=hashes,compression_hash=frozen['compression'],deployment_receiver_hashes=deploy,inference=dict(path=str(e),sha256=sha(e),module_hashes=hashes));dump(b/'checkpoint_index.json',index)
 if start==0:save(0)
 elif srow:=torch.load(latest,map_location='cpu',weights_only=True)['last_receipt']:
  dump(b/'receipts'/f'{start:04d}.json',srow)
 plan=load(ROOT/'plans/shared.json')['updates'];torch.cuda.reset_peak_memory_stats();tic=time.time()
 for step in range(start+1,1001):
  r=plan[step-1];t=time.time();assert r['adaptation_step']==step;crop_rng=rng_state();buf=buffer_state(models);opt.zero_grad(set_to_none=True)
  frames=PlanFrames(r,'cuda:0');gr,sums,trace=crop_c1(ctx,frames,r['external_qp']);frames.close();after_crop_rng=rng_state();after_crop_buf=buffer_state(models)
  feature_path=None;fullstatus='BLOCKED_FULL_RESOLUTION_MISMATCH';key=f'{r["domain"]}:{r["external_qp"]}'
  if full_eligible(r):
   restore_rng(global_rng);restore_buffers(models,buf);ff=PlanFrames(r,'cuda:0',full=True);z,gt,_,_=full_pass(ctx,ff,r['external_qp'],official);ff.close();global_rng=rng_state()
   feature_path=ROOT/'features/training'/branch/f'{step:04d}.npz';save_npz(feature_path,reconstruction=z.numpy(),real=gt.numpy());observed[key]+=1;fullstatus='OBSERVED_PRE_UPDATE_EMA_BLOCKED_REFERENCE'
  restore_rng(after_crop_rng);restore_buffers(models,after_crop_buf)
  assert tensor_hash({k:t for k,t in models['wrapper'].state_dict().items()})==frozen['wrapper']
  grad=groups(models,gr);assert all(g['finite'] and g['norm']>0 and g['nonzero']>0 for g in grad.values())
  for p,g in zip(params,gr):p.grad=g
  originals=[p.detach().clone() for p in params];pre=norm(gr);reported=float(torch.nn.utils.clip_grad_norm_(params,1.));assert math.isfinite(reported) and reported>0;clip=min(1.,1./(reported+1e-6))
  # Physical call journal survives a crash before the model/checkpoint commit.
  append(b/'optimizer_calls.jsonl',dict(event='STARTED',step=step,time=time.time(),PID=os.getpid()))
  opt.step();assert all(p.isfinite().all() for p in params);changes=[p.detach()-o for p,o in zip(params,originals)]
  assert all(x['norm']>0 for x in groups(models,changes).values());visits[key]+=1
  receipt=dict(r,branch=branch,source_step=1000,adaptation_step=step,committed_optimizer_updates=step,**sums,actual_qps=[x['actual_qp'] for x in trace],frame_trace=trace,gradient_modules=grad,parameter_delta=groups(models,changes),global_preclip_norm=pre,global_postclip_norm=norm([p.grad for p in params]),clip_coefficient=clip,learning_rates={g['name']:g['lr'] for g in opt.param_groups},full_frame_status=fullstatus,feature_path=None if feature_path is None else str(feature_path),feature_sha256=None if feature_path is None else sha(feature_path),stratum=key,stratum_visits=visits[key],EMA_commit_count=0,auxiliary_forward_did_not_change_crop_rng_or_buffers=True,supervised_P_frames=15,gpu=gpu,seconds=time.time()-t,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,finite=True)
  assert receipt['actual_qps']==r['actual_qps'];save(step,receipt);dump(b/'receipts'/f'{step:04d}.json',receipt)
  append(b/'optimizer_calls.jsonl',dict(event='COMMITTED',step=step,time=time.time(),PID=os.getpid()))
  dump(b/'training_status.json',dict(status='RUNNING' if step<1000 else 'PASS_CROP_TRAINING',PID=os.getpid(),gpu=gpu,adaptation_step=step,committed_updates=step,updated_unix=time.time(),seconds_per_update=(time.time()-tic)/(step-start),peak_memory_MiB=receipt['peak_memory_MiB'],peak_RSS_MiB=receipt['peak_RSS_MiB'],diagnostic_EMA='BLOCKED_SHARED_REFERENCE'))
  if step==1 or step%25==0:print('TRAIN',branch,step,'/1000',flush=True)
 rows=[load(b/'receipts'/f'{i:04d}.json') for i in range(1,1001)];write(b/'train.csv',rows);textfile(b/'train.jsonl',''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
 check_frozen(ctx);dump(b/'training_integrity.json',dict(status='PASS_CROP_TRAINING',updates=1000,frozen=frozen,source_step=1000,supervised_P_frames=15000,D_updates=0,optimizer_fresh=True,exact_plan=True,EMA_status='BLOCKED_SHARED_REFERENCE',full_view_completed=sum(r['feature_path'] is not None for r in rows),full_view_expected=1000,missing_full_view_steps=[r['adaptation_step'] for r in rows if r['feature_path'] is None]))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=BRANCHES,required=True);ap.add_argument('--gpu',choices=(4,5,6,7),type=int,required=True);a=ap.parse_args()
 try:main(a.branch,a.gpu)
 except Exception:error(dict(stage='train',branch=a.branch,error=traceback.format_exc()));raise
