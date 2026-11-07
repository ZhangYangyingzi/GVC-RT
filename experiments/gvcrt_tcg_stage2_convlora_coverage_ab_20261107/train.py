"""Four logical samples, two or four physical ranks, one SUM/4 and optimizer update."""
from common_s2 import *
from model_context import build,assert_reset,frozen_items,lora_state,load_lora_state,trainable_hash
from pipeline import Pipeline,optimized_loss
import argparse,datetime,traceback
import loss

def cpu_state(x):
    import torch
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_state(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_state(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_state(v) for v in x)
    return x

def norm(gs):return math.sqrt(sum(float(g.double().square().sum()) for g in gs))

def rng_hash(s):
    h=hashlib.sha256(repr(s['python']).encode())
    h.update(repr((s['numpy'][0],*s['numpy'][2:])).encode());h.update(s['numpy'][1].numpy().tobytes())
    h.update(s['torch'].numpy().tobytes());h.update(s['cuda'].numpy().tobytes());return h.hexdigest()

def run(group,until,engineering_tag,reference_anchor=None):
    assert reference_anchor is None or engineering_tag is not None
    import torch,torch.distributed as dist
    env();world=int(os.environ['WORLD_SIZE']);rank=int(os.environ['LOCAL_RANK']);assert world in (2,4)
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==world
    assert set(os.environ['CUDA_VISIBLE_DEVICES'].split(','))<=set(['4','5','6','7'])
    torch.cuda.set_device(rank);dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
    code_hash=digest({n:sha(ROOT/n) for n in ['train.py','model_context.py','convlora.py','loss.py','pipeline.py','data_reader.py']})
    if engineering_tag is None:verify_sources()
    c=build(rank,train=True,losses=True,group=group);frozen=FrozenGuard(frozen_items(c));plan=plan_rows()
    logical_ids=list(range(rank,4,world));logical_rng={}
    for logical in logical_ids:seed(SEED+logical);logical_rng[logical]=rng_state()
    home=group_dir(group) if engineering_tag is None else ROOT/'large_artifacts/aggregation'/engineering_tag/group
    cpdir=home/'checkpoints';rcdir=home/'receipts';cpdir.mkdir(parents=True,exist_ok=True);rcdir.mkdir(parents=True,exist_ok=True)
    train_lock=None
    if rank==0:
        train_lock=(home/'training.lock').open('a');fcntl.flock(train_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    latest=cpdir/'latest.pt';cursor=0
    source_sha=sha(ROOT/'source_manifest.json') if (ROOT/'source_manifest.json').exists() else None
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        assert saved['group']==group and saved['protocol_sha256']==sha(ROOT/'protocol.json')
        assert saved['source_manifest_sha256']==source_sha and saved['engineering_code_sha256']==code_hash
        assert saved['physical_world_size']==world and saved['baseline_sha256']==BASELINE_SHA
        load_lora_state(c,saved);c['optimizer'].load_state_dict(saved['optimizer'])
        cursor=saved['committed_optimizer_updates'];assert saved['next_cursor']==1500+cursor
        assert saved['frozen_sha256']==frozen.digest and saved['boundary_buffers_cleared']
        assert saved['common_initialization_sha256']==sha(ROOT/'common_initialization/untrained_common.pt')
        logical_rng={i:saved['all_logical_rank_RNG'][str(i)] for i in logical_ids}
        if cursor:assert all(int(s['step'])==cursor for s in c['optimizer'].state.values())
        if rank==0 and cursor and not (rcdir/f'update_{cursor:05d}.json.gz').exists():dump(rcdir/f'update_{cursor:05d}.json.gz',saved['last_committed_receipt'])
        if rank==0 and (cursor in (0,2) or cursor%250==0) and not (cpdir/f'step_{cursor}.pt').exists():atomic_torch(cpdir/f'step_{cursor}.pt',saved)
        assert not (rcdir/f'update_{cursor+1:05d}.json.gz').exists(),'RECEIPT_AHEAD_OF_MODEL_COMMIT'
    assert cursor<=until<=3000
    replicas=[None]*world;dist.all_gather_object(replicas,trainable_hash(c));assert len(set(replicas))==1
    groups={k:[i for i,(n,p) in enumerate(c['named']) if token in n] for k,token in [('stage3','.stage3.'),('head','.head.'),('upsample','.upsample.')]}
    groups={k:v for k,v in groups.items() if v}
    dist.barrier();assert_reset(c)

    def all_rng_states():
        gathered=[None]*world;dist.all_gather_object(gathered,logical_rng)
        result={str(k):v for d in gathered for k,v in d.items()};assert set(result)==set(map(str,range(4)));return result

    def commit(step,receipt):
        states=all_rng_states()
        if rank==0:
            control=group_dir(group)/'stopping_state.json'
            state=dict(format='coverage_AB_logical4_v1',group=group,lora_state=cpu_state(lora_state(c)),optimizer=cpu_state(c['optimizer'].state_dict()),
                committed_optimizer_updates=step,optimizer_step=step,next_cursor=1500+step,all_logical_rank_RNG=states,
                physical_world_size=world,logical_world_size=4,physical_to_logical={str(r):list(range(r,4,world)) for r in range(world)},
                physical_GPUs=[int(v) for v in os.environ['CUDA_VISIBLE_DEVICES'].split(',')],baseline_sha256=BASELINE_SHA,pilot_sha256=PILOT_SHA,
                TCG_cumulative_step_frozen_at=6500,protocol_sha256=sha(ROOT/'protocol.json'),source_manifest_sha256=source_sha,
                engineering_code_sha256=code_hash,common_initialization_sha256=sha(ROOT/'common_initialization/untrained_common.pt'),
                plan_index_sha256=sha(ROOT/'plans/index.json'),last_committed_receipt=receipt,boundary_buffers_cleared=True,frozen_sha256=frozen.digest,
                validation_state=load(control) if control.exists() and engineering_tag is None else None,engineering_only=engineering_tag is not None,time=time.time())
            atomic_torch(latest,state)
            if engineering_tag or step in (0,2) or step%250==0 or step==until:atomic_torch(cpdir/f'step_{step}.pt',state)
            if step:dump(rcdir/f'update_{step:05d}.json.gz',receipt)
            if engineering_tag is None:dump(home/'training_status.json',dict(status='RUNNING',committed_optimizer_updates=step,next_cursor=1500+step,
                latest_sha256=sha(latest),effective_batch=4,physical_world_size=world,logical_world_size=4,time=time.time()))
        dist.barrier()
    if not latest.exists():commit(0,None)
    if rank==0 and engineering_tag is None:
        dump(home/'baseline_provenance.json',c['provenance'])
        dump(home/'trainable_parameters.json',dict(group=group,count=sum(p.numel() for p in c['params']),configuration=c['provenance']['LoRA_configuration']))
    if cursor==until:
        if rank==0 and engineering_tag is None:dump(home/'training_status.json',dict(status='SEGMENT_COMPLETE',committed_optimizer_updates=cursor,next_cursor=1500+cursor,time=time.time()))
        frozen.full();dist.destroy_process_group();return
    sequence=[plan[4*i+logical] for i in range(cursor,until) for logical in logical_ids]
    pipeline=Pipeline(sequence,rank);fn=optimized_loss(loss.basic_crop)
    try:
        for step in range(cursor+1,until+1):
            anchor_sha=None
            if reference_anchor is not None:
                anchor_path=Path(reference_anchor)/f'step_{step-1}.pt'
                anchor=torch.load(anchor_path,map_location='cpu',weights_only=True)
                assert anchor['group']==group and anchor['committed_optimizer_updates']==step-1
                assert anchor['boundary_buffers_cleared'] and anchor['frozen_sha256']==frozen.digest
                load_lora_state(c,anchor);c['optimizer'].load_state_dict(anchor['optimizer'])
                logical_rng={i:anchor['all_logical_rank_RNG'][str(i)] for i in logical_ids}
                assert_reset(c);anchor_sha=sha(anchor_path)
            optimizer_before=cpu_state(c['optimizer'].state_dict()) if engineering_tag else None
            begin=time.time();torch.cuda.reset_peak_memory_stats(rank);c['optimizer'].zero_grad(set_to_none=True)
            before=[p.detach().clone() for p in c['params']];local_sum=[torch.zeros_like(p) for p in c['params']];samples=[];sample_gradients={}
            for logical in logical_ids:
                row=plan[4*(step-1)+logical];restore_rng(logical_rng[logical]);frames=pipeline.next(row)
                grad,diag=fn(c['ctx'],frames,row['external_qp'],raft=c['raft'],raft_transform=c['transform']);frames.close()
                logical_rng[logical]=rng_state()
                if engineering_tag:sample_gradients[logical]=cpu_state(grad)
                assert all(torch.isfinite(g).all() for g in grad) and norm(grad)>0
                assert all(math.isfinite(v) for v in diag['losses'].values()) and diag['actual_frame_qps']==row['actual_frame_qps']
                assert all(p.grad is None for n,p in c['pm'].named_parameters() if '.lora.' not in n)
                frozen.check()
                from evaluate import lora_statistics
                runtime=lora_statistics(c)
                assert all(not getattr(c['models']['generator'],n)._previous.requires_grad for n in ['tcg_low','tcg_mid','tcg_out'])
                assert_reset(c)
                for dst,g in zip(local_sum,grad):dst.add_(g)
                samples.append(dict(logical_rank=logical,physical_rank=rank,physical_GPU=physical_gpu(rank),sample_key=digest(row),video=row['video'],domain=row['domain'],
                    external_qp=row['external_qp'],source_frame_indices=row['source_frame_indices'],source_sha256=row['source_sha256'],crop=[row['crop_x'],row['crop_y'],256,256],
                    actual_frame_qps=diag['actual_frame_qps'],losses=diag['losses'],frame_losses=diag['trace'],sample_gradient_norm=norm(grad),
                    logical_RNG_sha256=rng_hash(logical_rng[logical]),frozen_byte_exact=True,buffers_cleared=True,runtime_statistics=runtime,GT_auxiliary=pipeline.last_target_audit))
            physical_sum_norm=norm(local_sum)
            for g in local_sum:dist.all_reduce(g,op=dist.ReduceOp.SUM);g.div_(4)
            mean_norm=norm(local_sum);assert mean_norm>0
            mean_group_norms={k:norm([local_sum[i] for i in indices]) for k,indices in groups.items()}
            assert all(v>0 for v in mean_group_norms.values())
            preclip=cpu_state(local_sum) if engineering_tag else None
            for p,g in zip(c['params'],local_sum):p.grad=g
            clip=float(torch.nn.utils.clip_grad_norm_(c['params'],1.,error_if_nonfinite=True))
            assert abs(clip-mean_norm)<=max(1e-7,1e-5*mean_norm)
            postclip=cpu_state([p.grad for p in c['params']]) if engineering_tag else None
            c['optimizer'].step();frozen.check()
            delta=norm([p-old for p,old in zip(c['params'],before)]);assert delta>0
            changes={k:norm([c['params'][i]-before[i] for i in indices]) for k,indices in groups.items()};assert all(v>0 for v in changes.values())
            hashes=[None]*world;dist.all_gather_object(hashes,trainable_hash(c));assert len(set(hashes))==1
            opt_hash=tensor_hash([(f'{i}_{n}',t) for i,s in c['optimizer'].state_dict()['state'].items() for n,t in s.items() if torch.is_tensor(t)])
            oh=[None]*world;dist.all_gather_object(oh,opt_hash);assert len(set(oh))==1
            local=dict(physical_rank=rank,physical_GPU=physical_gpu(rank),PID=os.getpid(),logical_ids=logical_ids,
                physical_accumulated_gradient_norm=physical_sum_norm,mean_gradient_norm=mean_norm,clip_input_norm=clip,clip_count=1,optimizer_step_count=1,
                mean_group_gradient_norms=mean_group_norms,group_parameter_changes_L2=changes,parameter_change_L2=delta,
                frozen_sha256=frozen.digest,seconds_before_commit=time.time()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(rank),free_memory_bytes=torch.cuda.mem_get_info(rank)[0],logical_samples=samples)
            gathered=[None]*world;dist.all_gather_object(gathered,local)
            logical_samples=sorted([s for x in gathered for s in x['logical_samples']],key=lambda x:x['logical_rank'])
            assert [s['logical_rank'] for s in logical_samples]==list(range(4))
            assert sorted(s['domain'] for s in logical_samples)==['ulong','ulong','uvg','uvg']
            receipt=dict(step=step,group=group,source_plan_cursor=1500+step,cumulative_samples=4*step,physical_world_size=world,logical_world_size=4,
                physical_ranks=gathered,logical_samples=logical_samples,gradient_normalization_denominator=4,global_clip_count=1,optimizer_step_count=1,
                rank_synchronized=True,optimizer_synchronized=True,only_LoRA_changed=True,frozen_unchanged=True,group_parameter_changes_L2=changes,reference_anchor_checkpoint_sha256=anchor_sha,
                lora_sha256=hashes[0],optimizer_sha256=opt_hash,time=time.time())
            commit(step,receipt)
            gathered_gradients=None
            if engineering_tag:
                gathered_gradients=[None]*world;dist.all_gather_object(gathered_gradients,sample_gradients)
            if rank==0 and engineering_tag:
                atomic_torch(home/f'comparison_step_{step}.pt',dict(gradient_preclip=preclip,gradient_postclip=postclip,sample_gradients={k:v for part in gathered_gradients for k,v in part.items()},
                    parameters_before=cpu_state(before),parameters_after=cpu_state(c['params']),optimizer=cpu_state(c['optimizer'].state_dict()),
                    optimizer_before=optimizer_before,reference_anchor_checkpoint_sha256=anchor_sha,losses=[x['losses'] for x in logical_samples],sample_keys=[x['sample_key'] for x in logical_samples],
                    logical_RNG_sha256=[x['logical_RNG_sha256'] for x in logical_samples],group=group,physical_world_size=world))
            if rank==0:event('LORA_AB_UPDATE_COMMITTED',group=group,step=step,engineering_only=engineering_tag is not None,mean_gradient_norm=mean_norm,seconds=time.time()-begin)
        frozen.full()
    finally:pipeline.close()
    dist.barrier()
    if rank==0:
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        assert saved['committed_optimizer_updates']==until and len(saved['all_logical_rank_RNG'])==4
        assert all(torch.equal(t.detach().cpu(),saved['lora_state'][n]) for n,t in lora_state(c).items())
        out=home/f'engineering_segment_{until:04d}.json' if engineering_tag else home/f'audits/training_segment_{until:04d}.json'
        dump(out,dict(status='PASS',group=group,committed_updates=until,formal_updates=0 if engineering_tag else until,
            effective_batch=4,physical_world_size=world,logical_world_size=4,all_logical_RNG_saved=True,checkpoint_reload=True,
            rank_and_optimizer_synchronized=True,frozen_unchanged=True,auxiliary_processes_terminated=True))
        if engineering_tag is None:dump(home/'training_status.json',dict(status='SEGMENT_COMPLETE',committed_optimizer_updates=until,next_cursor=1500+until,time=time.time()))
    dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=GROUPS,required=True);p.add_argument('--until',type=int,required=True);p.add_argument('--engineering-tag');p.add_argument('--reference-anchor');a=p.parse_args()
    try:run(a.group,a.until,a.engineering_tag,a.reference_anchor)
    except Exception:traceback.print_exc();raise

if __name__=='__main__':main()
