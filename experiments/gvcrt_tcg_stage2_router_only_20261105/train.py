"""Four GPU replicas, explicit SUM/4 gradients, one clipping and optimizer update."""
from common_s2 import *
import argparse,traceback,datetime
from model_context import build,assert_reset,frozen_items
from pipeline import Pipeline,optimized_loss
import loss

def cpu_state(state):
    import torch
    if torch.is_tensor(state):return state.detach().cpu().clone()
    if isinstance(state,dict):return {k:cpu_state(v) for k,v in state.items()}
    if isinstance(state,list):return [cpu_state(v) for v in state]
    if isinstance(state,tuple):return tuple(cpu_state(v) for v in state)
    return state

def run(until,smoke):
    import torch,torch.distributed as dist
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==4
    torch.cuda.set_device(rank);dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
    verify_sources();c=build(rank,train=True,losses=True);seed(SEED+rank)
    cpdir=ROOT/('large_artifacts/disposable_DDP/checkpoints' if smoke else 'checkpoints')
    rcdir=ROOT/('audits/disposable_DDP_receipts' if smoke else 'receipts');cpdir.mkdir(parents=True,exist_ok=True);rcdir.mkdir(parents=True,exist_ok=True)
    latest=cpdir/'latest.pt';rows=plan_rows();cursor=0
    frozen=FrozenGuard(frozen_items(c));protocol_sha=sha(ROOT/'protocol.json');source_sha=sha(ROOT/'source_manifest.json') if (ROOT/'source_manifest.json').exists() else None
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        assert saved['protocol_sha256']==protocol_sha and saved['source_manifest_sha256']==source_sha
        assert saved['baseline_sha256']==BASELINE_SHA and saved['plan_index_sha256']==sha(ROOT/'plans/index.json')
        c['router'].load_state_dict(saved['router_state'],strict=True);c['optimizer'].load_state_dict(saved['optimizer'])
        cursor=saved['committed_router_updates'];assert saved['next_cursor']==1500+cursor
        assert saved['frozen_sha256']==frozen.digest and saved['boundary_buffers_cleared']
        if cursor:
            assert all(int(s['step'])==cursor for s in c['optimizer'].state.values())
            receipt_path=rcdir/f'update_{cursor:05d}.json.gz'
            if rank==0 and not receipt_path.exists():dump(receipt_path,saved['last_committed_receipt'])
        assert not list(rcdir.glob(f'update_{cursor+1:05d}.json.gz')),'RECEIPT_AHEAD_OF_ATOMIC_MODEL_COMMIT'
        restore_rng(saved['all_rank_RNG'][rank])
    assert cursor<=until<=3000
    replicas=[None]*world;dist.all_gather_object(replicas,tensor_hash(c['router'].state_dict().items()));assert len(set(replicas))==1
    dist.barrier();assert_reset(c)

    def commit(step,receipt,always=False):
        all_rng=[None]*world;dist.all_gather_object(all_rng,rng_state())
        if rank==0:
            state=dict(format='router_only_spatial_gating_v1',router_state=cpu_state(c['router'].state_dict()),optimizer=cpu_state(c['optimizer'].state_dict()),
                all_rank_RNG=all_rng,next_cursor=1500+step,committed_router_updates=step,Router_optimizer_step=step,baseline_cumulative_TCG_step=6500,
                baseline_sha256=BASELINE_SHA,protocol_sha256=protocol_sha,source_manifest_sha256=source_sha,plan_index_sha256=sha(ROOT/'plans/index.json'),
                last_committed_receipt=receipt,boundary_buffers_cleared=True,frozen_sha256=frozen.digest,
                validation_state=load(ROOT/'stopping_state.json') if (ROOT/'stopping_state.json').exists() else None,
                world_size=4,physical_to_logical_rank={str(i):i for i in range(4)},time=time.time())
            atomic_torch(latest,state)
            if step in (0,2) or step%250==0 or always:atomic_torch(cpdir/f'step_{step}.pt',state)
            if step:dump(rcdir/f'update_{step:05d}.json.gz',receipt)
            if not smoke:dump(ROOT/'training_status.json',dict(status='RUNNING',committed_router_updates=step,next_cursor=1500+step,latest_sha256=sha(latest),last_receipt=None if receipt is None else digest(receipt),time=time.time(),process_group='four ranks SUM/4',physical_GPUs=[4,5,6,7]))
        dist.barrier()
    if cursor==0:commit(0,None,True)
    if rank==0:
        dump(ROOT/('audits/disposable_DDP_provenance.json' if smoke else 'baseline_provenance.json'),c['provenance'])
        dump(ROOT/'trainable_parameters.json',dict(modules=['recon_generation_net.router'],count=33889,parameters=[dict(name=n,shape=list(p.shape),dtype=str(p.dtype)) for n,p in c['named']],only_Router_requires_grad=True))
    if cursor==until:
        frozen.full();dist.destroy_process_group();return
    pipe=Pipeline([rows[4*i+rank] for i in range(cursor,until)],rank);fn=optimized_loss(loss.basic_crop)
    try:
        for step in range(cursor+1,until+1):
            start=time.time();row=rows[4*(step-1)+rank];torch.cuda.reset_peak_memory_stats(rank)
            frames=pipe.next(row);before=tensor_hash(c['router'].state_dict().items());snap=[p.detach().clone() for p in c['params']]
            grad,diag=fn(c['ctx'],frames,row['external_qp'],raft=c['raft'],raft_transform=c['transform']);frames.close()
            assert diag['actual_frame_qps']==row['actual_frame_qps']
            assert all(torch.isfinite(g).all() for g in grad) and all(math.isfinite(v) for v in diag['losses'].values())
            local_norm=math.sqrt(sum(float(g.double().square().sum()) for g in grad));assert local_norm>0
            local_smoke_grad=cpu_state(grad) if smoke else None
            for g in grad:dist.all_reduce(g,op=dist.ReduceOp.SUM);g.div_(4)
            reduced_norm=math.sqrt(sum(float(g.double().square().sum()) for g in grad));assert reduced_norm>0
            reduction_error=None
            if smoke:
                all_local=[None]*world;dist.all_gather_object(all_local,local_smoke_grad)
                reference=[sum(all_local[r][j].double() for r in range(4))/4 for j in range(len(grad))]
                diff=math.sqrt(sum(float((g.cpu().double()-v).square().sum()) for g,v in zip(grad,reference)))
                refnorm=math.sqrt(sum(float(v.square().sum()) for v in reference));reduction_error=diff/max(refnorm,1e-30)
                assert reduction_error<1e-3,('EXPLICIT_SUM_DIVIDE4_GRADIENT_CHECK',reduction_error)
            c['optimizer'].zero_grad(set_to_none=True)
            for p,g in zip(c['params'],grad):p.grad=g
            clip=float(torch.nn.utils.clip_grad_norm_(c['params'],1.,error_if_nonfinite=True));assert abs(clip-reduced_norm)<=max(1e-7,1e-5*reduced_norm)
            c['optimizer'].step();delta=math.sqrt(sum(float((p-b).double().square().sum()) for p,b in zip(c['params'],snap)));assert delta>0
            after=tensor_hash(c['router'].state_dict().items());assert after!=before
            frozen.check()
            from evaluate import mask_stats
            masks=mask_stats(c)
            assert c['router']._previous is not None and not c['router']._previous.requires_grad
            assert all(not getattr(c['models']['generator'],n)._previous.requires_grad for n in ['tcg_low','tcg_mid','tcg_out'])
            assert_reset(c)
            hashes=[None]*world;dist.all_gather_object(hashes,after);assert len(set(hashes))==1,'ROUTER_RANK_DESYNCHRONIZATION'
            optimizer_hash=tensor_hash([(f'{i}_{n}',t) for i,s in c['optimizer'].state_dict()['state'].items() for n,t in s.items() if torch.is_tensor(t)])
            oh=[None]*world;dist.all_gather_object(oh,optimizer_hash);assert len(set(oh))==1
            local=dict(rank=rank,physical_GPU=rank+4,PID=os.getpid(),sample_key=digest(row),video=row['video'],source_sha256=row['source_sha256'],
                source_group=row.get('source_group'),domain=row['domain'],external_qp=row['external_qp'],source_frame_indices=row['source_frame_indices'],
                crop=[row['crop_x'],row['crop_y'],256,256],actual_frame_qps=diag['actual_frame_qps'],losses=diag['losses'],frame_losses=diag['trace'],
                local_gradient_norm=local_norm,mean_gradient_norm=reduced_norm,clip_input_norm=clip,parameter_change_L2=delta,smoke_SUM_div4_relative_L2=reduction_error,
                Router_before_sha256=before,Router_after_sha256=after,optimizer_sha256=optimizer_hash,frozen_byte_exact=True,frozen_sha256=frozen.digest,
                buffers_cleared=True,seconds_before_commit=time.time()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),peak_reserved_bytes=torch.cuda.max_memory_reserved(rank),
                free_memory_bytes=torch.cuda.mem_get_info(rank)[0],GT_auxiliary=pipe.last_target_audit,mask_statistics=masks)
            gathered=[None]*world;dist.all_gather_object(gathered,local)
            assert sum(x['domain']=='uvg' for x in gathered)==2 and sum(x['domain']=='ulong' for x in gathered)==2
            receipt=dict(step=step,source_plan_cursor=1500+step,rank_samples=4*step,ranks=gathered,rank_synchronized=True,Router_sha256=after,
                frozen_unchanged=True,only_Router_changed=True,time=time.time(),protocol_sha256=protocol_sha)
            commit(step,receipt,always=step==until)
            if rank==0:event('ROUTER_UPDATE_COMMITTED',step=step,losses=[r['losses'] for r in gathered],gradient_norm=reduced_norm,seconds=time.time()-start)
        frozen.full()
    finally:pipe.close()
    dist.barrier()
    if rank==0:
        state=torch.load(latest,map_location='cpu',weights_only=True)
        assert state['committed_router_updates']==until and state['next_cursor']==1500+until and len(state['all_rank_RNG'])==4
        probe={n:t.detach().cpu() for n,t in c['router'].state_dict().items()}
        assert all(torch.equal(probe[n],state['router_state'][n]) for n in probe)
        dump(ROOT/('audits/DDP_smoke.json' if smoke else f'audits/training_segment_{until:04d}.json'),dict(status='PASS',committed_updates=until,
            formal_updates=0 if smoke else until,frozen_unchanged=True,only_Router_changed=True,effective_batch=4,
            rank_synchronized=True,optimizer_rank_synchronized=True,checkpoint_reload_equal=True,checkpoint_sha256=sha(latest),all_rank_RNG=4,
            auxiliary_processes_terminated=True,physical_GPUs=[4,5,6,7],world_size=4))
        if not smoke:dump(ROOT/'training_status.json',dict(status='SEGMENT_COMPLETE',committed_router_updates=until,next_cursor=1500+until,time=time.time()))
    dist.destroy_process_group()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--until',type=int,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();env()
    run(a.until,a.smoke)
if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
