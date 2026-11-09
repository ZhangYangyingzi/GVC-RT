"""Two ranks / two distinct windows each / one globally synchronized update."""
from common_s2 import *
import argparse,traceback,datetime

def checkpoint(c,step,receipt,states):
    from experiment_model import state
    names={id(p):n for n,p in c['named']}
    return dict(experiment_group=c['experiment_group'],B300_SHA256=B300_SHA,additional_updates=step,committed_updates=step,new_optimizer_step=step,next_cursor=4*step,trainable_state=state(c),optimizer=c['optimizer'].state_dict(),optimizer_name_groups={g['group_name']:[names[id(p)] for p in g['params']] for g in c['optimizer'].param_groups},RNG_by_rank=states,plan_SHA256=sha(ROOT/'plans/index.json'),protocol_SHA256=sha(ROOT/'protocol.json'),source_manifest_SHA256=sha(ROOT/'source_manifest.json') if (ROOT/'source_manifest.json').exists() else None,last_committed_receipt=receipt,provenance=c['provenance'],boundary_state='TCG and DPB reset',s_train=.5,s_eval=.5)

def rank_main(rank,group,smoke,rendezvous):
    import torch,torch.distributed as dist
    from experiment_model import construct,restore
    from model_context import assert_reset,frozen_items
    from cache_codec import cache_paths
    from pipeline import wait_rows
    from loss import backward_window,valid_frames,norm
    env();torch.cuda.set_device(rank)
    dist.init_process_group('nccl',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=datetime.timedelta(minutes=60))
    seed(PLAN_SEED);c=construct(group,True,rank);seed(PLAN_SEED)
    root=ROOT/group/('engineering' if smoke else 'training');root.mkdir(parents=True,exist_ok=True)
    latest=root/'latest.pt';step=0;rows=plan_rows();detail=load(ROOT/'audits/calibration.json')['lambda_detail'] if group=='F' else 0.
    assert load(ROOT/'audits/calibration.json')['status']=='PASS'
    if not smoke:
        assert load(ROOT/f'audits/engineering_{group}.json')['status']=='PASS'
        assert load(ROOT/f'audits/checkpoint_reload_{group}.json')['status']=='PASS'
        assert load(ROOT/'audits/step0_consistency.json')['status']=='PASS'
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        assert saved['plan_SHA256']==sha(ROOT/'plans/index.json') and saved['protocol_SHA256']==sha(ROOT/'protocol.json')
        restore(c,saved);restore_rng(saved['RNG_by_rank'][rank]);step=saved['committed_updates']
        if rank==0 and step:dump(root/f'receipts/{step:04d}.json',saved['last_committed_receipt'])
        del saved
    else:
        states=[None,None];dist.all_gather_object(states,rng_state())
        if rank==0:
            saved=checkpoint(c,0,None,states);atomic_torch(latest,saved);atomic_torch(root/'checkpoints/step_0000.pt',saved)
            dump(root/'initial_provenance.json',dict(status='PASS',additional_updates=0,full_trainable_SHA256=tensor_hash(c['named']),provenance=c['provenance'],parameters=[dict(name=n,shape=list(p.shape),count=p.numel(),dtype=str(p.dtype)) for n,p in c['named']]))
            del saved
    frozen=tensor_hash(frozen_items(c));cap=2 if smoke else 300
    dist.barrier()
    for update in range(step+1,cap+1):
        # Already-committed validation boundaries are completed by the controller
        # before invoking the next training chunk (cap is bounded below).
        if not smoke and update>int(os.environ['TRAIN_UNTIL']):break
        begun=time.time();torch.cuda.reset_peak_memory_stats(rank)
        batch=rows[4*(update-1):4*update];assert len({r['source_group'] for r in batch})==4
        wait_result=[None]
        dist.barrier() # every rank is at the SAME complete optimizer boundary
        if rank==0:
            try:wait_result[0]=dict(ok=True,seconds=wait_rows(batch,group=group))
            except Exception as e:wait_result[0]=dict(ok=False,error=repr(e))
        dist.broadcast_object_list(wait_result,src=0)
        assert wait_result[0]['ok'],wait_result[0]
        local=[]
        for r in batch[rank*2:rank*2+2]:
            key,p,rp=cache_paths(r,r['external_qp'],True);proof=load(rp);assert sha(p)==proof['cache_sha256']
            d=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
            assert d['info']['source_row']==r and d['info']['actual_qps']==r['actual_frame_qps']
            local.append(d)
        nonempty=torch.tensor(sum(valid_frames(d)>0 for d in local),device=c['device'],dtype=torch.int64)
        dist.all_reduce(nonempty,op=dist.ReduceOp.SUM);nnon=int(nonempty)
        c['optimizer'].zero_grad(set_to_none=True);records=[]
        for d in local:records.append(backward_window(c,d,nnon,detail))
        # Explicit SUM; losses already use GLOBAL4 and global nonempty count.
        # No DDP reducer and no world-size division anywhere else.
        used=[]
        for n,p in c['named']:
            flag=torch.tensor(int(p.grad is not None),device=c['device']);dist.all_reduce(flag)
            if p.grad is None:p.grad=torch.zeros_like(p)
            dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
            assert torch.isfinite(p.grad).all(),('NONFINITE_GLOBAL_GRADIENT',n)
            used.append(int(flag)>0)
        assert all(used),('TRAINABLE_FORWARD_BYPASS',[n for (n,p),u in zip(c['named'],used) if not u])
        before_norm=norm([p.grad for p in c['params']]);assert math.isfinite(before_norm) and before_norm>0
        if smoke and update==1:
            synced=[p.grad.detach().cpu().clone() for p in c['params']]
            if rank==0:
                # Same complete4-window objective on one device; autograd.grad
                # is retained only for this engineering reference, never formal.
                c['optimizer'].zero_grad(set_to_none=True);ref_records=[]
                for r in batch:
                    _,p,_=cache_paths(r,r['external_qp'],True)
                    d=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
                    ref_records.append(backward_window(c,d,nnon,detail,reference=True));del d
                refs=[p.grad.detach().cpu() for p in c['params']]
                difference=norm([a-b for a,b in zip(synced,refs)])/norm(refs)
                objective=sum(x['objective_contribution'] for x in ref_records)
                dump(ROOT/f'audits/gradient_equivalence_{group}.json',dict(status='PASS' if difference<1e-3 else 'FAIL',relative_L2=difference,fixed_tolerance=1e-3,reference='single-device4-window autograd.grad vs two-rank segment-total backward+SUM',reference_objective=objective,gradient_L2=before_norm,nonempty_windows=nnon,parameter_usage_all=True))
                assert difference<1e-3,('GRADIENT_EQUIVALENCE_FAILED',difference)
            for p,g in zip(c['params'],synced):p.grad=g.to(c['device'])
            del synced
            dist.barrier()
        clipped=float(torch.nn.utils.clip_grad_norm_(c['params'],1.,error_if_nonfinite=True))
        assert abs(clipped-before_norm)<max(1e-5,before_norm*1e-4)
        c['optimizer'].step()
        assert all(int(s['step'])==update for s in c['optimizer'].state.values())
        assert all(torch.isfinite(p).all() for p in c['params'])
        assert_reset(c)
        assert tensor_hash(frozen_items(c))==frozen,'FROZEN_CODEC_OR_METRIC_CHANGED'
        gathered=[None,None];dist.all_gather_object(gathered,dict(rank=rank,physical_GPU=physical_gpu(rank),pid=os.getpid(),records=records,peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),peak_reserved_bytes=torch.cuda.max_memory_reserved(rank)))
        states=[None,None];dist.all_gather_object(states,rng_state())
        if update==1 or update%50==0:
            hashes=[None,None];dist.all_gather_object(hashes,tensor_hash(c['named']));assert len(set(hashes))==1,('RANK_MODEL_DIVERGED',hashes)
        if rank==0:
            receipt=dict(group=group,additional_step=update,global_optimizer_updates=update,next_cursor=4*update,sample_keys=[r['sample_key'] for r in batch],loss=sum(r['objective_contribution'] for a in gathered for r in a['records']),gradient_norm=before_norm,post_clip_norm=norm([p.grad for p in c['params']]),nonempty_temporal_windows=nnon,global_batch=4,ranks=gathered,seconds=time.time()-begun,cache_wait_seconds=wait_result[0]['seconds'],normalization='SUM ranks; spatial/RAFT/detail /(15*4); temporal /(valid_frames_per_window*global_nonempty_windows)',frozen_SHA256=frozen,time=time.time())
            saved=checkpoint(c,update,receipt,states);atomic_torch(latest,saved)
            if update%50==0 or smoke:atomic_torch(root/f'checkpoints/step_{update:04d}.pt',saved)
            dump(root/f'receipts/{update:04d}.json',receipt)
            dump(root/'progress.json',dict(status='RUNNING',committed_updates=update,next_cursor=4*update,seconds=receipt['seconds'],time=time.time()))
            event('disposable_update' if smoke else 'formal_update_committed',group=group,step=update,loss=receipt['loss'],seconds=receipt['seconds'],gradient_norm=before_norm)
            if not smoke and update==1:
                launch=load(ROOT/'launch.json');dump(ROOT/f'audits/first_formal_step_{group}.json',dict(task_started=launch['time'],first_commit=time.time(),seconds=time.time()-launch['time']))
            del saved,receipt
        del local,records,gathered,states
        dist.barrier()
    if smoke and rank==0:dump(ROOT/f'audits/engineering_{group}.json',dict(status='PASS',discarded_updates=2,gradient_equivalence=load(ROOT/f'audits/gradient_equivalence_{group}.json'),fresh_B300_and_optimizer_required_for_formal=True,time=time.time()))
    dist.barrier();dist.destroy_process_group()

def main():
    import torch.multiprocessing as mp
    p=argparse.ArgumentParser();p.add_argument('--group',choices=GROUPS,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--until',type=int,default=300);a=p.parse_args();env()
    expected='0,1' if a.group=='E' else '2,3';assert os.environ['CUDA_VISIBLE_DEVICES']==expected
    os.environ['TRAIN_UNTIL']=str(a.until)
    rendezvous=ROOT/f'tmp/dist_{a.group}_{os.getpid()}'
    with lock(f'locks/train_{a.group}_{str(a.smoke)}.lock',blocking=True):
        mp.spawn(rank_main,args=(a.group,a.smoke,str(rendezvous)),nprocs=2,join=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        env();dump(ROOT/f'audits/train_failure_{os.getpid()}.json',dict(status='FAILED',exception=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
