"""Four synchronized replicas; explicit mean gradient all-reduce, no BPTT."""
import datetime,inspect,math,os,sys,time,traceback
from common_expanded import *
def train():
 import torch,torch.distributed as dist
 rank=int(os.environ['LOCAL_RANK']);assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'
 assert int(os.environ['WORLD_SIZE'])==4
 torch.cuda.set_device(rank);device=torch.device('cuda',rank)
 dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=300))
 sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'experiments/gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100'))
 import training32 as tr,frames32
 from loss import basic_crop
 tr.backend()
 v4=tr.module('expanded_v4',REPO/'experiments/gvcrt_neural_wrapper_v4_progressive_joint/train.py')
 im,pm=v4.load_models(device,force_zero_thres=.12)
 models=dict(wrapper=torch.nn.Identity().to(device),bridge=pm.recon_generation_net.mlp.float(),generator=pm.recon_generation_net.decoder.float())
 source=torch.load(SOURCE,map_location='cpu',weights_only=True)
 assert sha(SOURCE)==load(ROOT/'protocol.json')['source_checkpoint']['sha256']
 for k,m in models.items():m.load_state_dict(source['models'][k],strict=True)
 quality=v4.quality_models(device)
 allmods=dict(models,im=im,pm=pm,lp=quality[0],dists=quality[1])
 tr.restore_buffers(allmods,source['buffers'])
 from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
 raft=raft_large(weights=None,progress=False)
 raft.load_state_dict(torch.load(RAFT,map_location='cpu',weights_only=True),strict=True)
 raft=raft.to(device).eval().requires_grad_(False)
 transform=Raft_Large_Weights.C_T_SKHT_V2.transforms()
 for m in (im,pm,*quality,raft,*models.values()):m.eval().requires_grad_(False)
 gen=models['generator'];gen.train();gen.set_temporal_cascade_grad(False)
 for n,p in gen.named_parameters():p.requires_grad_(n.startswith(PREFIXES))
 named=[(n,p) for n,p in gen.named_parameters() if p.requires_grad];params=[p for n,p in named]
 assert sum(p.numel() for p in params)==137760
 assert all(p.dtype==torch.float32 for p in params)
 for k in ('tcg_low','tcg_mid','tcg_out'):
  assert torch.count_nonzero(getattr(gen,k).fusion[-1].weight)==0
  assert torch.count_nonzero(getattr(gen,k).fusion[-1].bias)==0
 for d in ('uvg','ulong'):
  frames32._SOURCES[d]={v['filename' if d=='ulong' else 'name']:v for v in load(ROOT/f'plans/train_manifest_{d}.local.json')['videos']}
 common=tr.module('expanded_qp',REPO/'experiments/gvcrt_neural_wrapper_v6_2_objective_calibration/fullqp_common.py')
 ctx=(v4,im,pm,models,quality,common,params,None)
 optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
 tr.seed(SEED+rank)
 def frozen_hash():
  return tensor_hash((prefix+'.'+n,t) for prefix,m in [('im',im),('pm',pm),('lp',quality[0]),('dists',quality[1]),('raft',raft)]
     for n,t in m.state_dict().items() if not(prefix=='pm' and n.startswith(tuple('recon_generation_net.decoder.'+p for p in PREFIXES))))
 initial_frozen=frozen_hash()
 def tcg_hash():return tensor_hash((n,p) for n,p in named)
 def check():
  assert frozen_hash()==initial_frozen,'Frozen state changed'
  for n,p in pm.named_parameters():
   expected=n.startswith(tuple('recon_generation_net.decoder.'+x for x in PREFIXES))
   assert p.requires_grad==expected
   if not expected:assert p.grad is None
  assert all(not p.requires_grad and p.grad is None for m in (im,*quality,raft) for p in m.parameters())
 def buffers():return {k:None if getattr(gen,k)._previous is None else getattr(gen,k)._previous.detach().cpu() for k in ('tcg_low','tcg_mid','tcg_out')}
 def collect(local):
  out=[None]*4;dist.all_gather_object(out,local);return out
 protocol_sha=sha(ROOT/'protocol.json');code=load(ROOT/'source_manifest.json')['training_sources']
 for name,expected in code.items():assert sha(REPO/name)==expected,('Training source changed',name)
 plan_index=load(ROOT/'data_plan.json')
 assert sha(ROOT/'plans/schedule.json')==plan_index['schedule_sha256']
 for shard in plan_index['shards']:assert sha(ROOT/'plans'/shard['file'])==shard['sha256']
 latest=ROOT/'checkpoints/latest.pt';start=0
 def checkpoint(step,receipt,all_rng,all_buffers):
  tcg={n:t.detach().cpu() for n,t in gen.state_dict().items() if n.startswith(PREFIXES)}
  state=dict(committed_global_updates=step,plan_cursor=step,rank_sample_cursor=4*step,
       tcg_model_state=tcg,optimizer=optimizer.state_dict(),all_rank_RNG_states=all_rng,
       TCG_buffers=all_buffers,protocol_sha256=protocol_sha,
       source_checkpoint_sha256=sha(SOURCE),source_hashes=code,
       frozen_state_hash=initial_frozen,last_committed_receipt=receipt,
       reconstruction='Reload fixed pilot step0 models and frozen codec, then TCG-only state; cleared history is rebuilt by planned warmup')
  atomic_torch(latest,state)
  if step in STEPS:atomic_torch(ROOT/f'checkpoints/step_{step}.pt',state)
 if latest.exists():
  st=torch.load(latest,map_location='cpu',weights_only=True)
  assert st['protocol_sha256']==protocol_sha and st['frozen_state_hash']==initial_frozen
  assert st['source_checkpoint_sha256']==sha(SOURCE) and st['source_hashes']==code
  missing,unexpected=gen.load_state_dict(st['tcg_model_state'],strict=False)
  assert not unexpected and all(not n.startswith(PREFIXES) for n in missing)
  optimizer.load_state_dict(st['optimizer']);tr.restore_rng(st['all_rank_RNG_states'][rank])
  start=st['committed_global_updates'];assert st['plan_cursor']==start
  assert all(x is None for b in st['TCG_buffers'] for x in b.values())
  if rank==0:
   r=st['last_committed_receipt']
   if start and not(ROOT/f'receipts/{start:05d}.json').exists():dump(ROOT/f'receipts/{start:05d}.json',r)
   if start in STEPS and not(ROOT/f'checkpoints/step_{start}.pt').exists():atomic_torch(ROOT/f'checkpoints/step_{start}.pt',st)
  del st
 else:
  all_rng=collect(tr.rng_state());all_buffers=collect(buffers())
  init=collect(dict(rank=rank,TCG_hash=tcg_hash(),frozen_hash=initial_frozen))
  assert len({r['TCG_hash'] for r in init})==len({r['frozen_hash'] for r in init})==1
  if rank==0:
   checkpoint(0,dict(event='INITIALIZED',rank_audits=init),all_rng,all_buffers)
   event('INITIALIZED',world_size=4,protocol_sha256=protocol_sha)
 dist.barrier();del source
 schedule=load(ROOT/'plans/schedule.json')['updates'];rows=[r for p in sorted((ROOT/'plans').glob('base_rows_*.json')) for r in load(p)]
 comparisons=[];hook=None
 if start==0:
  def compare(module,args,kwargs,output):
   bound=inspect.signature(module.forward).bind(*args,**kwargs)
   bound.arguments.update(use_tcg=False,update_buffer=False)
   with torch.no_grad():original=module.forward(*bound.args,**bound.kwargs)
   assert torch.equal(output.detach(),original),'Zero-init decoder mismatch'
   comparisons.append(0.0)
  hook=gen.register_forward_hook(compare,with_kwargs=True)
 for step in range(start+1,5001):
  torch.cuda.reset_peak_memory_stats();begun=time.monotonic()
  s=schedule[step-1];assert s['global_update']==step
  item=s['ranks'][rank];row=dict(rows[item['base_row']]);row['external_qp']=item['external_qp']
  optimizer.zero_grad(set_to_none=True);before=[p.detach().clone() for p in params]
  frames=frames32.PlanFrames(row,device)
  try:gs,summary=basic_crop(ctx,frames,row['external_qp'],dists_weight=.5,raft=raft,raft_transform=transform)
  finally:frames.close()
  if hook is not None:
   hook.remove();hook=None;assert len(comparisons)==63
  assert all(math.isfinite(x) for x in summary['losses'].values())
  assert all(g.dtype==torch.float32 and torch.isfinite(g).all() for g in gs)
  flat=torch.cat([g.reshape(-1) for g in gs]);local_norm=float(flat.norm());assert math.isfinite(local_norm) and local_norm>0
  dist.all_reduce(flat,op=dist.ReduceOp.SUM);flat.div_(4)
  offset=0
  for p in params:p.grad=flat[offset:offset+p.numel()].view_as(p).clone();offset+=p.numel()
  raw=float(torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True));assert math.isfinite(raw) and raw>0
  optimizer.step()
  assert all(torch.isfinite(p).all() for p in params)
  delta=tr.norm([p.detach()-b for p,b in zip(params,before)]);assert delta>0
  check()
  assert all(getattr(gen,k)._previous is not None and not getattr(gen,k)._previous.requires_grad for k in ('tcg_low','tcg_mid','tcg_out'))
  pm.clear_dpb();assert not pm.dpb and all(x is None for x in buffers().values())
  assert {int(s['step']) for s in optimizer.state.values()}=={step}
  torch.cuda.synchronize()
  local=dict(rank=rank,physical_GPU=rank+4,sample_sha256=item['sample_sha256'],base_row=item['base_row'],
    domain=row['domain'],video=row['video'],external_qp=row['external_qp'],losses=summary['losses'],
    local_gradient_norm=local_norm,reduced_preclip_gradient_norm=raw,TCG_delta_norm=delta,
    non_TCG_delta_norm=0.,TCG_hash=tcg_hash(),frozen_hash=initial_frozen,buffer_reset=True,
    actual_qps=summary['actual_frame_qps'],free_memory_bytes=torch.cuda.mem_get_info()[0],
    peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
    seconds=time.monotonic()-begun,step0_equivalence_frames=63 if step==1 else None)
  audits=collect(local);rngs=collect(tr.rng_state());bufs=collect(buffers())
  assert len({a['sample_sha256'] for a in audits})==4
  assert len({a['TCG_hash'] for a in audits})==1,'Rank model desynchronization'
  assert len({a['reduced_preclip_gradient_norm'] for a in audits})==1
  if rank==0:
   receipt=dict(global_update=step,plan_cursor=step,world_size=4,effective_batch_size=4,
      all_reduce='NCCL SUM / 4 before clipping and optimizer.step',rank_sync='PASS',
      rank_audits=audits,mean_losses={k:sum(a['losses'][k] for a in audits)/4 for k in summary['losses']},
      finite_loss=True,finite_nonzero_gradient=True,only_TCG_changed=True,frozen_check='PASS',buffer_reset=True)
   if (ROOT/f'receipts/{step:05d}.json').exists():raise RuntimeError('Refusing duplicate committed receipt')
   checkpoint(step,receipt,rngs,bufs)
   dump(ROOT/f'receipts/{step:05d}.json',receipt)
   dump(ROOT/'training_summary.json',dict(status='RUNNING',committed_global_updates=step,planned_global_updates=5000,
        world_size=4,rank_samples=step*4,checkpoint_cursor=step))
   event('GLOBAL_UPDATE_COMMITTED',step=step,mean_losses=receipt['mean_losses'],gradient_norm=raw,
        TCG_delta_norm=delta,non_TCG_delta_norm=0.,rank_sync='PASS',seconds=max(a['seconds'] for a in audits))
   if step==2:dump(ROOT/'audits/DDP_smoke.json',dict(status='PASS',global_updates=2,rank_sync=True,all_reduce=True,
       finite_loss=True,finite_nonzero_gradient=True,only_TCG_changed=True,buffer_reset=True,included_in_budget=True))
  dist.barrier()
  if step==2:assert load(ROOT/'audits/DDP_smoke.json')['status']=='PASS'
 check()
 st=torch.load(latest,map_location='cpu',weights_only=True)
 assert st['committed_global_updates']==st['plan_cursor']==5000
 expected={n:p.detach().cpu() for n,p in named}
 assert tensor_hash(expected.items())==tensor_hash(st['tcg_model_state'].items())
 assert len(st['all_rank_RNG_states'])==4
 assert all((ROOT/f'checkpoints/step_{s}.pt').is_file() for s in STEPS)
 if rank==0:
  dump(ROOT/'training_summary.json',dict(status='PASS',committed_global_updates=5000,planned_global_updates=5000,
       world_size=4,rank_samples=20000,checkpoint_reload='PASS',rank_sync='PASS',frozen_state='PASS'))
  dump(ROOT/'audits/training_integrity.json',dict(status='PASS',committed_global_updates=5000,
       finite_loss=True,finite_nonzero_gradient=True,only_TCG_changed=True,frozen_state_hash=initial_frozen,
       buffer_reset=True,DDP_rank_sync=True,checkpoint_reload=True))
  event('TRAINING_PASS',committed_global_updates=5000)
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 try:train()
 except Exception as e:
  rank=int(os.environ.get('LOCAL_RANK','0'))
  dump(ROOT/f'logs/rank_{rank}_error.json',dict(status='FAILED',error=str(e),traceback=traceback.format_exc()))
  traceback.print_exc();sys.exit(1)
