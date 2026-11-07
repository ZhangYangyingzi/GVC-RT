"""One model per arm; four logical RNG/sample streams on two or four physical ranks."""
from v3_common import *
from ab_model_context import build,module,tensor_hash,frozen_items
from v3_runtime import Pipeline,optimized_loss
import argparse,datetime,gzip,traceback
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--source',required=True);ap.add_argument('--target',type=int,required=True);ap.add_argument('--cards',required=True);ap.add_argument('--probe-output');args=ap.parse_args()
 import torch,torch.distributed as dist
 verify_sources();cards=list(map(int,args.cards.split(',')));rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
 assert world==len(cards) and world in (2,4) and set(cards)<=set((4,5,6,7)) and os.environ['CUDA_VISIBLE_DEVICES']==args.cards
 torch.cuda.set_device(rank);dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=600))
 state=torch.load(args.source,map_location='cpu',weights_only=True);start=state['additional_step'];target=args.target;assert start<target<=12710
 assert state['parent_checkpoint_sha256']==PARENT_SHA and state['plan_index_sha256']==load(ROOT/'protocol.json')['plan_index_sha256'][args.arm]
 if state.get('active_protocol_revision')==V3:
  assert state['protocol_sha256']==sha(ROOT/'protocol_v3.json') and state['source_hashes_v3']==load(ROOT/'source_manifest_v3.json')['code']
 else:
  takeover=load(MIG/'takeover.json')['frozen_checkpoints'][args.arm];assert sha(args.source)==takeover['sha256'] and start==takeover['additional_step']
 if not args.probe_output:assert cards==load(MIG/'runtime_selection.json')['physical_GPUs'][args.arm]
 c=build(rank);tr=c['tr'];pm=c['pm'];gen=c['models']['generator'];params=c['params'];named=c['named'];optimizer=c['optimizer'];mapping=gpu_mapping(rank,cards)
 gen.load_state_dict(state['tcg_model_state'],strict=False);optimizer.load_state_dict(state['optimizer']);assert {int(s['step']) for s in optimizer.state.values()}=={5000+start}
 assert all(g['lr']==1e-4 and g['betas']==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==1e-4 for g in optimizer.param_groups)
 logical_rng=list(state['all_rank_RNG_states']);assert len(logical_rng)==4 and all(v is None for b in state['TCG_buffers'] for v in b.values())
 frozen=module('v3_main_frozen',P/'runtime_acceleration.py');initial=frozen.frozen_hash(frozen_items(c));assert initial==state['frozen_state_hash']
 plan=rows(args.arm);logicals=list(range(rank,4,world));sequence=[plan[4*(s-1)+lr] for s in range(start+1,target+1) for lr in logicals]
 pipeline=Pipeline(sequence,rank,cards);lossfn=optimized_loss(module('v3_fixed_loss',ROOT/'loss.py').basic_crop)
 def gather(obj):
  out=[None]*world;dist.all_gather_object(out,obj);return out
 def model():return {n:t.detach().cpu() for n,t in gen.state_dict().items() if n.startswith(PREFIXES)}
 branch=ROOT/args.arm;probe=bool(args.probe_output);last=start;begun_segment=time.monotonic()
 if rank==0 and not probe:
  found=[int(p.name.split('.')[0]) for p in (branch/'receipts_v3').glob('*.json.gz')]
  assert not any(n>start for n in found),'RECEIPT_AHEAD_OF_ATOMIC_LATEST'
  prefix=load(MIG/'bootstrap.json')['arms'][args.arm]['legacy_prefix_additional_step']
  assert all(n in found for n in range(prefix+1,start))
  if start>prefix and start not in found:dump(branch/f'receipts_v3/{start:06d}.json.gz',state['last_committed_receipt'])
 dist.barrier()
 for step in range(start+1,target+1):
  disk();torch.cuda.reset_peak_memory_stats(rank);begun=time.monotonic();before=[p.detach().clone() for p in params];optimizer.zero_grad(set_to_none=True);sample_audits=[];local_rng={};local_sum=None
  for lr in logicals:
   row=plan[4*(step-1)+lr];tr.restore_rng(logical_rng[lr]);sample_start=time.monotonic();frames=pipeline.next(row)
   try:gs,summary=lossfn(c['ctx'],frames,row['external_qp'],dists_weight=.5,raft=c['raft'],raft_transform=c['transform'])
   finally:frames.close()
   summary['losses']['frame_loss']=summary['losses']['total_loss']-summary['losses']['weighted_motion_loss']
   for frame in summary['trace']:frame['frame_loss']=frame['total_loss']-frame['weighted_motion_loss']
   assert summary['actual_frame_qps']==row['actual_frame_qps'] and all(math.isfinite(v) for v in summary['losses'].values())
   flat=torch.cat([g.reshape(-1) for g in gs]);norm=float(flat.norm());assert bool(torch.isfinite(flat).all()) and math.isfinite(norm) and norm>0
   local_sum=flat.clone() if local_sum is None else local_sum+flat
   assert all(getattr(gen,k)._previous is not None and not getattr(gen,k)._previous.requires_grad for k in ('tcg_low','tcg_mid','tcg_out'))
   pm.clear_dpb();assert not pm.dpb and all(getattr(gen,k)._previous is None for k in ('tcg_low','tcg_mid','tcg_out'))
   assert frozen.frozen_hash(frozen_items(c))==initial
   local_rng[lr]=tr.rng_state();sample_audits.append(dict(rank=lr,logical_rank=lr,physical_rank=rank,physical_GPU=cards[rank],sample_sha256=digest(row),source_sha256=row['source_sha256'],source_group=row['source_group'],video=row['video'],domain=row['domain'],external_qp=row['external_qp'],actual_qps=summary['actual_frame_qps'],losses=summary['losses'],frame_loss_trace=summary['trace'],local_gradient_norm=norm,buffer_reset=True,frozen_check='EXACT_DEVICE_BYTES_PASS',frozen_hash=initial,auxiliary_GT_flow=pipeline.last_target_audit,seconds=time.monotonic()-sample_start))
  # Each logical sample contributes exactly once; there is no division by
  # physical world size elsewhere. Mean batch=4 precedes a single clipping.
  dist.all_reduce(local_sum,op=dist.ReduceOp.SUM);local_sum.div_(4);reduced=local_sum.detach().cpu().clone();offset=0
  for p in params:p.grad=local_sum[offset:offset+p.numel()].view_as(p).clone();offset+=p.numel()
  preclip=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));assert math.isfinite(preclip) and preclip>0
  clipped=torch.cat([p.grad.reshape(-1) for p in params]).detach().cpu().clone()
  optimizer.step();assert all(bool(torch.isfinite(p).all()) for p in params)
  delta=tr.norm([p.detach()-b for p,b in zip(params,before)]);assert delta>0 and frozen.frozen_hash(frozen_items(c))==initial
  for n,p in pm.named_parameters():
   expected=n.startswith(tuple('recon_generation_net.decoder.'+x for x in PREFIXES));assert p.requires_grad==expected
   if not expected:assert p.grad is None
  assert all(not p.requires_grad and p.grad is None for m in (c['im'],*c['quality'],c['raft']) for p in m.parameters())
  assert {int(s['step']) for s in optimizer.state.values()}=={5000+step}
  h=tensor_hash(named);hs=gather(h);assert len(set(hs))==1
  audits=[a for batch in gather(sample_audits) for a in batch];audits.sort(key=lambda a:a['logical_rank']);assert [a['logical_rank'] for a in audits]==list(range(4))
  assert collections.Counter(a['domain'] for a in audits)=={'uvg':2,'ulong':2}
  rng_map={k:v for batch in gather(local_rng) for k,v in batch.items()};logical_rng=[rng_map[k] for k in range(4)]
  torch.cuda.synchronize(rank);elapsed=time.monotonic()-begun;performance=gather(dict(physical_rank=rank,physical_GPU=cards[rank],GPU_mapping=mapping,seconds=elapsed,peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(rank),peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(rank),free_memory_bytes=torch.cuda.mem_get_info(rank)[0]))
  for a in audits:a.update(reduced_preclip_gradient_norm=preclip,TCG_delta_norm=delta,TCG_hash=h,non_TCG_delta_norm=0.)
  receipt=dict(protocol_revision=V3,arm=args.arm,additional_step=step,absolute_optimizer_step=5000+step,rank_sample_cursor=4*step,physical_world_size=world,logical_world_size=4,effective_batch_size=4,rank_audits=audits,physical_rank_performance=performance,gradient_reduction='sum of four logical sample gradients /4; one global clip1 then optimizer.step',rank_sync='PASS',finite_loss=True,finite_nonzero_gradient=True,only_TCG_changed=True,buffer_reset=True,TCG_hash=h,mean_losses={k:sum(a['losses'][k] for a in audits)/4 for k in audits[0]['losses']})
  if rank==0:
   snapshot=dict(additional_step=step,committed_global_updates=5000+step,absolute_optimizer_step=5000+step,plan_cursor=step,rank_sample_cursor=4*step,tcg_model_state=model(),optimizer=optimizer.state_dict(),all_rank_RNG_states=logical_rng,TCG_buffers=[{k:None for k in ('tcg_low','tcg_mid','tcg_out')} for _ in range(4)],protocol_sha256=sha(ROOT/'protocol_v3.json'),predecessor_protocol_sha256=sha(ROOT/'protocol.json'),active_protocol_revision=V3,source_hashes_v3=load(ROOT/'source_manifest_v3.json')['code'],base_source_hashes=load(ROOT/'source_manifest.json')['code'],parent_checkpoint_sha256=PARENT_SHA,plan_index_sha256=load(ROOT/'protocol.json')['plan_index_sha256'][args.arm],frozen_state_hash=initial,last_committed_receipt=receipt,arm=args.arm,physical_world_size=world,logical_world_size=4)
   if probe:
    snapshot.update(probe_only=True,formal_updates_committed=0,reduced_gradient=reduced,clipped_gradient=clipped,parameter_before=torch.cat([b.reshape(-1) for b in before]).cpu(),parameter_after=torch.cat([p.detach().reshape(-1) for p in params]).cpu(),receipt=receipt)
    atomic_torch(Path(args.probe_output),snapshot)
   else:
    dest=branch/f'receipts_v3/{step:06d}.json.gz';assert not dest.exists(),'DUPLICATE_COMMITTED_UPDATE'
    atomic_torch(latest_path(args.arm),snapshot)
    if step in (2,target) or step%500==0 or step==12710:atomic_torch(branch/f'checkpoints_v3/step_{step}.pt',snapshot);frozen.full_frozen_hash_audit()
    dump(dest,receipt);dump(branch/'training_status_v3.json',dict(status='RUNNING',committed_additional_updates=step,absolute_optimizer_step=5000+step,target_segment_step=target,physical_GPUs=cards,effective_batch_size=4))
    event('V3_GLOBAL_UPDATE_COMMITTED',arm=args.arm,additional_step=step,absolute_step=5000+step,physical_GPUs=cards,logical_batch=4,seconds=max(x['seconds'] for x in performance),mean_losses=receipt['mean_losses'],gradient_norm=preclip)
    if step==2:dump(branch/'smoke_v3.json',dict(status='PASS',additional_updates=2,included_in_budget=True,four_logical_samples=True,finite_loss=True,finite_nonzero_gradient=True,only_TCG_changed=True,buffer_reset=True))
  last=step;dist.barrier()
  pause=[bool(load(ROOT/'control_v3.json').get('pause_requested',False)) if rank==0 else None];dist.broadcast_object_list(pause,src=0)
  if pause[0]:break
 frozen.full_frozen_hash_audit();pipeline.close();dist.barrier()
 if rank==0 and not probe:
  if last!=target:atomic_torch(branch/f'checkpoints_v3/step_{last}.pt',snapshot)
  dump(branch/'segment_status_v3.json',dict(status='SEGMENT_COMPLETE' if last==target else 'CONTROL_PAUSED',committed_additional_updates=last,target_step=target,elapsed_seconds=time.monotonic()-begun_segment,finished_unix=time.time(),physical_GPUs=cards,physical_world_size=world,effective_batch_size=4))
 dist.destroy_process_group()
if __name__=='__main__':
 try:
  arm=sys.argv[sys.argv.index('--arm')+1];(ROOT/arm).mkdir(exist_ok=True)
  if int(os.environ['LOCAL_RANK'])==0:
   with lock(f'{arm}/v3_rank0_active.lock'):main()
  else:main()
 except Exception:
  dump(ROOT/f'logs/v3_train_error_{os.environ.get("TCG_ARM","unknown")}_{os.environ.get("LOCAL_RANK","0")}_{time.time_ns()}.json',dict(status='FAILED',traceback=traceback.format_exc()));traceback.print_exc();sys.exit(1)
