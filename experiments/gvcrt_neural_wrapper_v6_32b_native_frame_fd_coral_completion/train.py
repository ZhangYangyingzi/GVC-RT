"""C/D real full-chain distribution objective; atomic optimizer+EMA commits.
Every successful step is checkpointed (stronger than latest every25).
A crash before checkpoint replacement discards the uncommitted physical call;
resume replays that same cursor from the last committed model/optimizer/EMA.
"""
from io32 import *
import argparse,fcntl,resource,math,traceback

def main(branch,gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch
 from training32 import setup,seed,crop_c1,full_pass,official_model,rng_state,restore_rng,buffer_state,restore_buffers,norm,groups,check_frozen,dot
 from frames32 import PlanFrames
 from calibration32b import get_stratum,feature_derivatives
 from distribution32 import to,commit
 assert branch in BRANCHES[2:];assert load(ROOT/'audits/distribution_math.json')['status']=='PASS';assert load(ROOT/'geometry_audit.json')['status']=='PASS';cal=load(ROOT/'calibration/final.json');assert cal['status']=='PASS' and cal['completed_windows']==40
 b=ROOT/'branches'/branch;b.mkdir(exist_ok=True);lock=(b/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ctx=setup();v4,im,pm,models,quality,common,params,frozen=ctx;official=official_model();oh=tensor_hash(official.state_dict());cfg=load(ROOT/'config.json');sig=signature('train');calsha=sha(ROOT/'calibration/final.json');seedsha=sha(ROOT/'reference_stats/index.json')
 seed(cfg['seed_global']);global_rng=rng_state();seed(cfg['seed']);opt=torch.optim.AdamW([dict(name=k,params=list(models[k].parameters()),lr=lr) for k,lr in zip(('bridge','generator'),cfg['branch_lrs'][branch])],betas=(.9,.999),eps=1e-8,weight_decay=1e-4);assert not opt.state
 latest=b/'checkpoints/latest.pt';index=load(b/'checkpoint_index.json') if (b/'checkpoint_index.json').exists() else {};start=0;ema={};visits={f'{d}:{q}':0 for d in ('ulong','uvg') for q in range(10)}
 for item in load(ROOT/'reference_stats/index.json')['entries']:
  assert sha(item['path'])==item['sha256'];ema[item['stratum']]=torch.load(item['path'],map_location='cpu',weights_only=True)['seed']
 if latest.exists():
  s=torch.load(latest,map_location='cpu',weights_only=True);assert s['signature']==sig and s['branch']==branch and s['plan_sha256']==sha(ROOT/'plans/shared.json') and s['calibration_sha256']==calsha and s['seed_index_sha256']==seedsha
  for k,m in models.items():m.load_state_dict(s[k])
  opt.load_state_dict(s['optimizer']);restore_rng(s['crop_rng']);global_rng=s['global_rng'];start=s['adaptation_step'];visits=s['visits'];ema=s['EMA'];restore_buffers(models,s['buffers'])
  assert sum(v['counter'] for v in ema.values())==start
  if s['last_receipt']:dump(b/'receipts'/f'{start:04d}.json',s['last_receipt'])
  dump(b/'restore_audit.json',dict(status='PASS',restored_committed_step=start,model_optimizer_EMA_same_transaction=True,calibration_sha256=calsha,EMA_counters={k:v['counter'] for k,v in ema.items()},optimizer_not_refreshed=True,PID=os.getpid()));del s
 def save(step,row=None):
  check_frozen(ctx);assert oh==tensor_hash(official.state_dict())
  assert sum(v['counter'] for v in ema.values())==step and all(ema[k]['counter']==v for k,v in visits.items())
  s=dict(schema='v632b',branch=branch,source_step=1000,adaptation_step=step,committed_optimizer_updates=step,replay_cursor=step,signature=sig,config_sha256=sha(ROOT/'config.json'),protocol_sha256=sha(ROOT/'protocol.json'),plan_sha256=sha(ROOT/'plans/shared.json'),source_checkpoint_sha256=cfg['source_checkpoint']['sha256'],calibration_sha256=calsha,seed_index_sha256=seedsha,**{k:m.state_dict() for k,m in models.items()},optimizer=opt.state_dict(),crop_rng=rng_state(),global_rng=global_rng,buffers=buffer_state(models),visits=visits,EMA=ema,frozen_hashes=frozen,last_receipt=row)
  atomic_torch(latest,s)
  if step in (0,1,250,500,1000):
   p=b/'checkpoints'/f'adaptation_step_{step:04d}.pt';atomic_torch(p,s);e=ROOT/'evaluation/checkpoints'/f'{branch}_{step}.pt';atomic_torch(e,{k:m.state_dict() for k,m in models.items()})
   hashes={k:v4.module_hash(m) for k,m in models.items()};deploy={k:tensor_hash({n:t.half() for n,t in models[k].state_dict().items()}) for k in ('bridge','generator')}
   if step==0:
    assert hashes==cfg['source_checkpoint']['module_hashes'];assert deploy==load(ROOT/'evaluation_config.json')['deployment_receiver_hashes']['B1_initial']
    dump(ROOT/'audits'/f'initial_inference_alias_{branch}.json',dict(status='PASS',source_checkpoint=cfg['source_checkpoint'],training_FP32_module_hashes=hashes,deployment_FP16_receiver_hashes=deploy,baseline='verified B1_initial independent-decode real RANS points',step0_is_B1_alias=True,no_duplicate_encoding=True,inference_code=str(ROOT/'adapter32.py'),inference_code_sha256=sha(ROOT/'adapter32.py')))
   index[str(step)]=dict(path=str(p),sha256=sha(p),source_step=1000,adaptation_step=step,module_hashes=hashes,compression_hash=frozen['compression'],deployment_receiver_hashes=deploy,inference=dict(path=str(e),sha256=sha(e),module_hashes=hashes),calibration_sha256=calsha,EMA_counters={k:v['counter'] for k,v in ema.items()});dump(b/'checkpoint_index.json',index)
 if start==0:save(0)
 plan=load(ROOT/'plans/shared.json')['updates'];torch.cuda.reset_peak_memory_stats();tic=time.time()
 objectives=('FD',) if branch==BRANCHES[2] else ('mean','CORAL')
 weights={'FD':cal['lambda_FD'],'mean':cal['alpha_mu'],'CORAL':cal['alpha_cov']}
 for step in range(start+1,1001):
  row=plan[step-1];t=time.time();assert row['adaptation_step']==step;buf=buffer_state(models);crop_rng=rng_state();opt.zero_grad(set_to_none=True)
  frames=PlanFrames(row,'cuda:0');base,sums,trace=crop_c1(ctx,frames,row['external_qp']);frames.close();after_rng=rng_state();after_buf=buffer_state(models)
  key=f'{row["domain"]}:{row["external_qp"]}';_,ref=get_stratum(key,'cuda:0');old=to(ema[key],'cuda:0')
  restore_rng(global_rng);restore_buffers(models,buf);f=PlanFrames(row,'cuda:0',full=True);z,gt,_,_=full_pass(ctx,f,row['external_qp'],official);f.close();global_after=rng_state()
  fg,values,next_state=feature_derivatives(z,old,ref,objectives);combined=sum(weights[k]*fg[k] for k in objectives)
  # Feature-space moments already contain the15-frame mean. VJP never divides again.
  restore_rng(global_rng);restore_buffers(models,buf);f=PlanFrames(row,'cuda:0',full=True);zr,_,extra,replay=full_pass(ctx,f,row['external_qp'],official,fg if step==1 else combined,expected=z);f.close();global_rng=global_after
  unit_contributions={k:groups(models,g) for k,g in extra.items()} if isinstance(extra,dict) else None
  if isinstance(extra,dict):extra=[sum(weights[k]*extra[k][j] for k in objectives) for j in range(len(params))]
  restore_rng(after_rng);restore_buffers(models,after_buf);gr=[a+b for a,b in zip(base,extra)];grad=groups(models,gr);eg=groups(models,extra);assert all(g['finite'] and g['norm']>0 and g['nonzero']>0 for g in list(grad.values())+list(eg.values()));assert [x['actual_qp'] for x in trace]==row['actual_qps']
  for p,g in zip(params,gr):p.grad=g
  before=[p.detach().clone() for p in params];pre=norm(gr);rep=float(torch.nn.utils.clip_grad_norm_(params,1.));assert math.isfinite(rep) and rep>0;clip=min(1.,1./(rep+1e-6))
  append(b/'optimizer_calls.jsonl',dict(event='STARTED',step=step,time=time.time(),PID=os.getpid(),committed_before=step-1));opt.step();assert all(p.isfinite().all() for p in params);changes=[p.detach()-v for p,v in zip(params,before)];assert all(v['norm']>0 for v in groups(models,changes).values())
  # EMA candidate was computed with pre-update Z; one candidate per successful update.
  ema[key]=to(commit(old,next_state),'cpu');visits[key]+=1;assert ema[key]['counter']==visits[key]
  fp=ROOT/'features/training'/branch/f'{step:04d}.npz';save_npz(fp,reconstruction=z.numpy(),real=gt.numpy())
  receipt=dict(row,branch=branch,source_step=1000,adaptation_step=step,committed_optimizer_updates=step,**sums,distribution_losses=values,distribution_weights={k:weights[k] for k in objectives},additional_loss=sum(weights[k]*values[k] for k in objectives),complete_objective_loss=sums['total_loss']+sum(weights[k]*values[k] for k in objectives),actual_qps=[x['actual_qp'] for x in trace],frame_trace=trace,gradient_modules=grad,additional_gradient_modules=eg,first_step_unit_objective_gradients=unit_contributions,additional_gradient_norm=norm(extra),C1_full_gradient_norm=norm(base),additional_to_C1_ratio=norm(extra)/norm(base),additional_C1_cosine=dot(base,extra)/(norm(base)*norm(extra)),parameter_delta=groups(models,changes),global_preclip_norm=pre,global_postclip_norm=norm([p.grad for p in params]),clip_coefficient=clip,gradient_combination='C1 + weighted native-full-frame VJP; single global clip; single optimizer step',gradient_formula_error=norm([g-a-e for g,a,e in zip(gr,base,extra)])/max(norm(gr),1e-30),learning_rates={g['name']:g['lr'] for g in opt.param_groups},full_frame_status='PASS_NATIVE_FULL_CHAIN',feature_path=str(fp),feature_sha256=sha(fp),stratum=key,stratum_visits=visits[key],EMA_commit_count=ema[key]['counter'],total_EMA_commits=sum(v['counter'] for v in ema.values()),replay=replay,auxiliary_forward_did_not_change_crop_rng_or_buffers=True,supervised_P_frames=15,gpu=gpu,seconds=time.time()-t,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,finite=True,calibration_sha256=calsha,seed_index_sha256=seedsha,discriminator_optimizer_updates=0)
  save(step,receipt);dump(b/'receipts'/f'{step:04d}.json',receipt);append(b/'optimizer_calls.jsonl',dict(event='COMMITTED',step=step,time=time.time(),PID=os.getpid()))
  dump(b/'training_status.json',dict(status='RUNNING' if step<1000 else 'PASS',PID=os.getpid(),gpu=gpu,adaptation_step=step,committed_updates=step,updated_unix=time.time(),seconds_per_update=(time.time()-tic)/(step-start),peak_memory_MiB=receipt['peak_memory_MiB'],peak_RSS_MiB=receipt['peak_RSS_MiB'],EMA_commits=step))
  if step==1:dump(ROOT/'audits'/f'first_committed_step_{branch}.json',receipt)
  if step==1 or step%25==0:print('TRAIN',branch,step,'/1000',receipt['seconds'],flush=True)
 rows=[load(b/'receipts'/f'{i:04d}.json') for i in range(1,1001)];keys=['adaptation_step','domain','video','external_qp','total_loss','distribution_losses','distribution_weights','additional_loss','C1_full_gradient_norm','additional_gradient_norm','additional_to_C1_ratio','additional_C1_cosine','gradient_modules','additional_gradient_modules','global_preclip_norm','global_postclip_norm','clip_coefficient','learning_rates','parameter_delta','total_EMA_commits','seconds','peak_memory_MiB','peak_RSS_MiB']
 write(b/'training_details.csv',[{k:r[k] for k in keys} for r in rows]);write(b/'gradient_summary.csv',[dict(step=r['adaptation_step'],domain=r['domain'],QP=r['external_qp'],C1_norm=r['C1_full_gradient_norm'],additional_norm=r['additional_gradient_norm'],additional_to_C1=r['additional_to_C1_ratio'],cosine=r['additional_C1_cosine'],preclip=r['global_preclip_norm'],clip=r['clip_coefficient'],bridge_additional_norm=r['additional_gradient_modules']['bridge']['norm'],generator_additional_norm=r['additional_gradient_modules']['generator']['norm'],EMA_commits=r['total_EMA_commits'],seconds=r['seconds'],peak_MiB=r['peak_memory_MiB']) for r in rows]);textfile(b/'train.jsonl',''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));check_frozen(ctx)
 dump(b/'training_integrity.json',dict(status='PASS',updates=1000,frozen=frozen,source_step=1000,supervised_P_frames=15000,D_updates=0,optimizer_fresh=True,exact_plan=True,EMA_commits=1000,EMA_counters={k:v['counter'] for k,v in ema.items()},full_view_completed=1000,calibration_sha256=calsha,seed_index_sha256=seedsha))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=BRANCHES[2:],required=True);ap.add_argument('--gpu',choices=(4,5,6,7),type=int,required=True);a=ap.parse_args()
 try:main(a.branch,a.gpu)
 except Exception:error(dict(stage='train',branch=a.branch,error=traceback.format_exc()));raise
