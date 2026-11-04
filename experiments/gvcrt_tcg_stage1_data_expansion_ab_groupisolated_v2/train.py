"""Four-rank paired continuation. Atomic latest is the optimizer commit authority."""
from ab_common import *
from ab_model_context import build,tensor_hash,frozen_items,module
import datetime,gzip,traceback
def receipt_write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:f.write(gzip.compress(json.dumps(data,allow_nan=False,separators=(',',':')).encode(),mtime=0));f.flush();os.fsync(f.fileno())
 tmp.replace(path)
def training(arm):
 import torch,torch.distributed as dist
 assert arm in ('A_clean_old_pool','B_expanded_ulong') and os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'
 rank=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==4
 protocol=load(ROOT/'protocol.json');S=protocol['additional_global_updates'];branch=ROOT/arm
 def update_training_summary(committed,state='RUNNING',**details):
  dump(branch/'training_summary.json',dict(status=state,committed_additional_updates=committed,absolute_optimizer_step=5000+committed,planned_additional_updates=S,rank_samples=4*committed,world_size=4,**details))
  arms={}
  for name in ('A_clean_old_pool','B_expanded_ulong'):
   p=ROOT/name/'training_summary.json'
   arms[name]=load(p) if p.exists() else dict(status='NOT_STARTED',committed_additional_updates=0,absolute_optimizer_step=5000,rank_samples=0)
  dump(ROOT/'training_summary.json',dict(status='RUNNING',protocol_revision=REVISION,S=S,active_arm=arm,arms=arms,paired_parent_initialization=True))
 assert load(ROOT/'audits/plans_frozen.json')['protocol_sha256']==sha(ROOT/'protocol.json')
 assert load(ROOT/'audits/runtime_equivalence.json')['status']=='PASS'
 source_manifest=load(ROOT/'source_manifest.json');code=source_manifest['code'];runtime_code=source_manifest['read_only_runtime']
 for n,h in code.items():assert sha(ROOT/n)==h,('CODE_CHANGED_AFTER_PLAN_FREEZE',n)
 for n,h in runtime_code.items():assert sha(REPO/n)==h,('READ_ONLY_RUNTIME_CHANGED_AFTER_PLAN_FREEZE',n)
 for n,h in source_manifest['original_helpers'].items():assert sha(ORIGINAL/n)==h,('FIXED_PARENT_LOADING_HELPER_CHANGED',n)
 index=load(ROOT/f'plans/{arm}/index.json');assert sha(ROOT/f'plans/{arm}/index.json')==protocol['plan_index_sha256'][arm]
 rows=[]
 for shard in index['shards']:
  assert sha(ROOT/shard['file'])==shard['sha256'];rows.extend(load(ROOT/shard['file']))
 assert len(rows)==4*S
 torch.cuda.set_device(rank);dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=600))
 c=build(rank);tr=c['tr'];gen=c['models']['generator'];pm=c['pm'];params=c['params'];named=c['named'];optimizer=c['optimizer']
 frozen=module('ab_exact_frozen',P/'runtime_acceleration.py');initial=frozen.frozen_hash(frozen_items(c))
 assert initial==c['parent']['frozen_state_hash'],'FROZEN_RECONSTRUCTION_MISMATCH'
 basic_crop=module('ab_fixed_objective',ROOT/'loss.py').basic_crop
 from ab_runtime import Pipeline,optimized_loss
 lossfn=optimized_loss(basic_crop)
 def collect(x):
  out=[None]*4;dist.all_gather_object(out,x);return out
 def buffers():return {k:None if getattr(gen,k)._previous is None else getattr(gen,k)._previous.detach().cpu() for k in ('tcg_low','tcg_mid','tcg_out')}
 def model_state():return {n:t.detach().cpu() for n,t in gen.state_dict().items() if n.startswith(PREFIXES)}
 def check():
  assert frozen.frozen_hash(frozen_items(c))==initial
  for n,p in pm.named_parameters():
   expected=n.startswith(tuple('recon_generation_net.decoder.'+x for x in PREFIXES));assert p.requires_grad==expected
   if not expected:assert p.grad is None
  assert all(not p.requires_grad and p.grad is None for m in (c['im'],*c['quality'],c['raft']) for p in m.parameters())
 def commit(step,receipt,rngs,bufs):
  if step in protocol['checkpoint_additional_steps']:frozen.full_frozen_hash_audit()
  state=dict(additional_step=step,committed_global_updates=5000+step,absolute_optimizer_step=5000+step,plan_cursor=step,rank_sample_cursor=4*step,tcg_model_state=model_state(),optimizer=optimizer.state_dict(),all_rank_RNG_states=rngs,TCG_buffers=bufs,protocol_sha256=sha(ROOT/'protocol.json'),parent_checkpoint_sha256=PARENT_SHA,plan_index_sha256=protocol['plan_index_sha256'][arm],source_hashes=code,read_only_runtime_hashes=runtime_code,frozen_state_hash=initial,last_committed_receipt=receipt,arm=arm)
  atomic_torch(branch/'checkpoints/latest.pt',state)
  if step in protocol['checkpoint_additional_steps']:atomic_torch(branch/f'checkpoints/step_{step}.pt',state)
 latest=branch/'checkpoints/latest.pt';start=0
 def verified_geometry_source_transition(st):
  path=ROOT/'repairs/native_geometry_v1/transition.json'
  assert sha(path)=='9b5c1529bbedeb4e73dc92e43d2cc6bd211c8c75e35481d56e758090ae3c9a0f'
  transition=load(path);predecessor=ROOT/transition['predecessor_source_manifest_file'];authority=transition['resume_authority']
  assert sha(predecessor)==transition['predecessor_source_manifest_sha256']
  old=load(predecessor);assert st['source_hashes']==old['code'] and set(old['code'])==set(code)
  assert {n for n in code if code[n]!=old['code'][n]}==set(transition['allowed_changed_original_sources'])
  assert old['read_only_runtime']==runtime_code and old['original_helpers']==source_manifest['original_helpers']
  assert transition['protocol_sha256']==st['protocol_sha256']==sha(ROOT/'protocol.json')
  assert transition['plan_index_sha256_unchanged']==protocol['plan_index_sha256']
  assert arm==authority['arm'] and sha(latest)==authority['checkpoint_sha256']
  assert st['additional_step']==st['plan_cursor']==authority['additional_step']==authority['next_plan_cursor']==10
  assert st['absolute_optimizer_step']==authority['absolute_optimizer_step']==5010 and st['rank_sample_cursor']==authority['next_rank_sample_cursor']==40
  receipt_path=branch/'receipts/000010.json.gz';assert sha(receipt_path)==authority['last_committed_receipt_sha256']
  assert st['last_committed_receipt']==load(receipt_path)
  if rank==0:dump(branch/'geometry_source_transition_resume.json',dict(status='PASS',scope='Exact documented step10 source-revision acceptance only; checkpoint tensors/RNG/optimizer unchanged',checkpoint_sha256=sha(latest),transition_sha256=sha(path),predecessor_source_manifest_sha256=sha(predecessor),current_source_manifest_sha256=sha(ROOT/'source_manifest.json'),next_additional_step=11,committed_additional_updates=10,no_optimizer_update_replayed=True))
  return True
 if latest.exists():
  st=torch.load(latest,map_location='cpu',weights_only=True)
  assert st['protocol_sha256']==sha(ROOT/'protocol.json') and st['parent_checkpoint_sha256']==PARENT_SHA
  assert st['source_hashes']==code or verified_geometry_source_transition(st)
  assert st['read_only_runtime_hashes']==runtime_code
  assert st['arm']==arm and st['frozen_state_hash']==initial and st['plan_index_sha256']==protocol['plan_index_sha256'][arm]
  start=st['additional_step'];assert st['plan_cursor']==start and st['absolute_optimizer_step']==5000+start
  assert all(x is None for b in st['TCG_buffers'] for x in b.values())
  gen.load_state_dict(st['tcg_model_state'],strict=False);optimizer.load_state_dict(st['optimizer']);tr.restore_rng(st['all_rank_RNG_states'][rank])
  assert {int(s['step']) for s in optimizer.state.values()}=={5000+start}
  if rank==0:
   present=[int(p.name.split('.')[0]) for p in (branch/'receipts').glob('*.json.gz')]
   assert not any(x>start for x in present),'RECEIPT_AHEAD_OF_COMMITTED_CHECKPOINT'
   assert all(x in present for x in range(1,start)),'MISSING_PREVIOUS_COMMITTED_RECEIPT'
   if start and start not in present:receipt_write(branch/f'receipts/{start:06d}.json.gz',st['last_committed_receipt'])
   if start in protocol['checkpoint_additional_steps'] and not(branch/f'checkpoints/step_{start}.pt').exists():atomic_torch(branch/f'checkpoints/step_{start}.pt',st)
 else:
  init=collect(dict(rank=rank,TCG_hash=tensor_hash(named),RNG_hash=tensor_hash([('torch',tr.rng_state()['torch'])]) if 'torch' in tr.rng_state() else digest(str(tr.rng_state())),frozen_hash=initial))
  assert len({r['TCG_hash'] for r in init})==1
  rngs=collect(tr.rng_state());bufs=collect(buffers())
  if rank==0:
   commit(0,dict(event='PARENT_INITIALIZATION',rank_audits=init),rngs,bufs)
   dump(branch/'initialization.json',dict(status='PASS',parent_checkpoint_sha256=PARENT_SHA,TCG_tensors_exactly_parent=True,optimizer_absolute_step=5000,optimizer_state_preserved=True,all_rank_RNG_preserved=True,zero_head_assertion='NOT_APPLICABLE_TRAINED_PARENT',same_input_output_audit='audits/runtime_equivalence.json'))
 if rank==0 and start>=2:
  smoke=branch/'smoke.json'
  if not smoke.exists():
   for n in (1,2):
    r=load(branch/f'receipts/{n:06d}.json.gz');assert r['additional_step']==n and r['rank_sync']=='PASS' and r['finite_loss'] and r['finite_nonzero_gradient'] and r['only_TCG_changed'] and r['buffer_reset']
   dump(smoke,dict(status='PASS',additional_updates=2,included_in_budget=True,rank_sync=True,finite_loss=True,nonzero_gradient=True,only_TCG_changed=True,buffer_reset=True,recovered_from_committed_receipts=True))
  else:assert load(smoke)['status']=='PASS'
 if rank==0:update_training_summary(start)
 dist.barrier();pipeline=Pipeline(rows,rank,start)
 # Pipeline constructors use only helper-process RNG. Reapply model RNG after
 # startup to explicitly preserve the four parent rank initial RNG states.
 if start==0:tr.restore_rng(c['parent']['all_rank_RNG_states'][rank])
 for step in range(start+1,S+1):
  disk();torch.cuda.reset_peak_memory_stats(rank);begun=time.monotonic();row=rows[4*(step-1)+rank];before=[p.detach().clone() for p in params]
  optimizer.zero_grad(set_to_none=True);frames=pipeline.next(row)
  try:gs,summary=lossfn(c['ctx'],frames,row['external_qp'],dists_weight=.5,raft=c['raft'],raft_transform=c['transform'])
  finally:frames.close()
  assert summary['actual_frame_qps']==row['actual_frame_qps'],'ACTUAL_FRAME_QP_PLAN_MISMATCH'
  summary['losses']['frame_loss']=summary['losses']['total_loss']-summary['losses']['weighted_motion_loss']
  for frame in summary['trace']:frame['frame_loss']=frame['total_loss']-frame['weighted_motion_loss']
  assert all(math.isfinite(x) for x in summary['losses'].values()) and all(torch.isfinite(g).all() for g in gs)
  flat=torch.cat([g.reshape(-1) for g in gs]);local=float(flat.norm());assert math.isfinite(local) and local>0
  dist.all_reduce(flat,op=dist.ReduceOp.SUM);flat.div_(4);offset=0
  for p in params:p.grad=flat[offset:offset+p.numel()].view_as(p).clone();offset+=p.numel()
  norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));assert math.isfinite(norm) and norm>0
  optimizer.step();assert all(torch.isfinite(p).all() for p in params)
  delta=tr.norm([p.detach()-b for p,b in zip(params,before)]);assert delta>0;check()
  assert all(getattr(gen,k)._previous is not None and not getattr(gen,k)._previous.requires_grad for k in ('tcg_low','tcg_mid','tcg_out'))
  pm.clear_dpb();assert not pm.dpb and all(x is None for x in buffers().values())
  assert {int(s['step']) for s in optimizer.state.values()}=={5000+step}
  torch.cuda.synchronize(rank)
  local=dict(rank=rank,physical_GPU=4+rank,sample_sha256=digest(row),source_sha256=row['source_sha256'],source_group=row['source_group'],video=row['video'],domain=row['domain'],external_qp=row['external_qp'],actual_qps=summary['actual_frame_qps'],losses=summary['losses'],frame_loss_trace=summary['trace'],local_gradient_norm=local,reduced_preclip_gradient_norm=norm,TCG_delta_norm=delta,TCG_hash=tensor_hash(named),non_TCG_delta_norm=0.,frozen_check='EXACT_DEVICE_BYTES_PASS',frozen_hash=initial,buffer_reset=True,free_memory_bytes=torch.cuda.mem_get_info(rank)[0],peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(rank),peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(rank),seconds=time.monotonic()-begun,coverage_pass=row.get('coverage_pass'),padding=row.get('padding',False),auxiliary_GT_flow=pipeline.last_target_audit)
  local['exact_content_sha256']=digest([row['domain'],row['source_sha256'],row['source_frame_indices'],row['crop_x'],row['crop_y'],row['external_qp']])
  audits=collect(local);rngs=collect(tr.rng_state());bufs=collect(buffers())
  assert len({r['TCG_hash'] for r in audits})==1 and len({r['reduced_preclip_gradient_norm'] for r in audits})==1
  assert len({r['sample_sha256'] for r in audits})==4
  if rank==0:
   receipt=dict(arm=arm,additional_step=step,absolute_optimizer_step=5000+step,rank_sample_cursor=4*step,rank_audits=audits,unique_exact_rank_samples=len({r['exact_content_sha256'] for r in audits}),rank_sync='PASS',finite_loss=True,finite_nonzero_gradient=True,only_TCG_changed=True,buffer_reset=True,mean_losses={k:sum(r['losses'][k] for r in audits)/4 for k in summary['losses']})
   target=branch/f'receipts/{step:06d}.json.gz';assert not target.exists(),'DUPLICATE_UPDATE_COMMIT'
   commit(step,receipt,rngs,bufs);receipt_write(target,receipt)
   update_training_summary(step)
   event('GLOBAL_UPDATE_COMMITTED',arm=arm,additional_step=step,absolute_step=5000+step,mean_losses=receipt['mean_losses'],gradient_norm=norm,seconds=max(r['seconds'] for r in audits))
   if step==2:dump(branch/'smoke.json',dict(status='PASS',additional_updates=2,included_in_budget=True,rank_sync=True,finite_loss=True,nonzero_gradient=True,only_TCG_changed=True,buffer_reset=True))
  dist.barrier()
 check();frozen.full_frozen_hash_audit();pipeline.close()
 st=torch.load(latest,map_location='cpu',weights_only=True)
 assert st['additional_step']==st['plan_cursor']==S and st['absolute_optimizer_step']==5000+S and st['rank_sample_cursor']==4*S
 assert tensor_hash(model_state().items())==tensor_hash(st['tcg_model_state'].items())
 assert all((branch/f'checkpoints/step_{s}.pt').exists() for s in protocol['checkpoint_additional_steps'])
 if rank==0:
  manifest=[dict(filename=f.name,size_bytes=f.stat().st_size,sha256=sha(f)) for f in sorted((branch/'checkpoints').glob('*.pt'))]
  dump(branch/'checkpoints_manifest.json',dict(checkpoints=manifest));update_training_summary(S,'PASS',checkpoint_reload=True,frozen_state=True,rank_sync=True,only_TCG_changed=True,finite_nonzero_gradient=True)
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 try:
  arm=os.environ['TCG_ARM'];(ROOT/arm).mkdir(exist_ok=True)
  if int(os.environ['LOCAL_RANK'])==0:
   with lock(f'{arm}/rank0_active.lock'):training(arm)
  else:training(arm)
 except Exception:
  rank=os.environ.get('LOCAL_RANK','0');arm=os.environ.get('TCG_ARM','unknown');error=dict(status='FAILED',arm=arm,rank=rank,traceback=traceback.format_exc());dump(ROOT/f'logs/{arm}_rank{rank}_error_{time.time_ns()}.json',error);traceback.print_exc();sys.exit(1)
