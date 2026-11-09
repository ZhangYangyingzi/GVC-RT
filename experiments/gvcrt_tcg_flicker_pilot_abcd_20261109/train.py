"""One GPU per branch; four distinct windows per committed optimizer update."""
from common_s2 import *
from experiment_model import construct,state,restore
from model_context import frozen_items,assert_reset
from cache_codec import cache_paths
from loss import window_gradients,gradient_norm,grouped_norms,TERMS
import argparse,traceback

def average_components(component_sum,valid_temporal_windows):
    assert 0<=valid_temporal_windows<=4
    averaged={}
    for key,values in component_sum.items():
        denominator=valid_temporal_windows if key.startswith('temp_') else 4
        averaged[key]=[value/denominator for value in values] if denominator else values
    total=[sum(averaged[k][i] for k in TERMS) for i in range(len(next(iter(averaged.values()))))]
    return averaged,total

def checkpoint(c,step,receipt,protocol_hash,code_hash):
    return dict(experiment_group=c['experiment_group'],initial_A2000_sha256=INIT_SHA,baseline_TCG_sha256=BASELINE_SHA,TCG_cumulative_step_at_initialization=6500,TCG_cumulative_training_updates=6500+step,
        committed_updates=step,additional_updates=step,LoRA_absolute_update_count=2000+step,new_optimizer_step=step,trainable_state=state(c),optimizer=c['optimizer'].state_dict(),
        optimizer_name_groups={g['group_name']:[{id(p):n for n,p in c['named']}[id(p)] for p in g['params']] for g in c['optimizer'].param_groups},RNG=rng_state(),next_cursor=4*step,plan_index_SHA256=sha(ROOT/'plans/index.json'),protocol_SHA256=protocol_hash,source_manifest_SHA256=code_hash,last_committed_receipt=receipt,
        boundary_state=dict(TCG='all caches empty',DPB='empty',previous_reconstruction=None),s_train=.5,s_eval=.5)

def run(group,rank,smoke=False):
    import torch
    env();root=ROOT/group/('smoke' if smoke else 'training');root.mkdir(parents=True,exist_ok=True)
    with lock(f'locks/{group}_'+('smoke' if smoke else 'training')+'.lock'):
        if (root/'completion.json').exists():
            done=load(root/'completion.json');assert done['status']=='PASS';return done
        if not smoke:
            from freeze import verify
            verify();assert load(ROOT/'audits'/f'engineering_{group}.json')['status']=='PASS'
            assert load(ROOT/'audits/calibration.json')['status']=='PASS'
        seed(PLAN_SEED);c=construct(group,True,rank);seed(PLAN_SEED)
        lambda_temp=load(ROOT/'audits/calibration.json')['lambda_temp'];rows=plan_rows()
        frozen=FrozenGuard(frozen_items(c));protocol_hash=sha(ROOT/'protocol.json');code_hash=sha(ROOT/'source_manifest.json') if (ROOT/'source_manifest.json').exists() else None
        dump(ROOT/f'audits/trainable_{group}.json',dict(group=group,parameters=[dict(name=n,shape=list(p.shape),count=p.numel(),dtype=str(p.dtype)) for n,p in c['named']],provenance=c['provenance'],frozen_SHA256=frozen.digest))
        latest=root/'latest.pt';step=0
        if latest.exists():
            saved=torch.load(latest,map_location='cpu',weights_only=True)
            assert saved['protocol_SHA256']==protocol_hash and saved['source_manifest_SHA256']==code_hash and saved['plan_index_SHA256']==sha(ROOT/'plans/index.json')
            restore(c,saved);restore_rng(saved['RNG']);step=saved['committed_updates']
            if step:
                rp=root/f'receipts/{step:04d}.json'
                if rp.exists():assert load(rp)==saved['last_committed_receipt']
                else:dump(rp,saved['last_committed_receipt'])
            assert not any(int(p.stem)>step for p in (root/'receipts').glob('*.json')),'RECEIPT_AHEAD_OF_COMMITTED_CHECKPOINT'
        else:
            assert len(c['optimizer'].state)==0
            assert_reset(c);s=checkpoint(c,0,None,protocol_hash,code_hash);atomic_torch(latest,s);atomic_torch(root/'checkpoints/step_0000.pt',s)
        cap=20 if smoke else 300;stop_reason=None;update=step
        # A committed validation boundary must finish validation before any
        # subsequent update. Restarting at300 never executes update301.
        if not smoke and step>0 and step%100==0:
            from evaluation import evaluate_group
            from summarize import screening,should_stop
            del c,frozen;torch.cuda.empty_cache();evaluate_group(group,step,rank);screening()
            stopped=(step==200 and should_stop(group))
            if stopped or step==cap:
                done=dict(status='PASS',group=group,committed_updates=step,stop_reason='PRESET_NO_TEMPORAL_GAIN_AND_SPATIAL_REGRESSION_200' if stopped else 'MAX_300_UPDATES_REACHED',disposable=False,final_checkpoint=str(root/f'checkpoints/step_{step:04d}.pt'),time=time.time())
                dump(root/'completion.json',done);return done
            c=construct(group,True,rank);saved=torch.load(latest,map_location='cpu',weights_only=True);restore(c,saved);restore_rng(saved['RNG']);del saved
            frozen=FrozenGuard(frozen_items(c))
        for update in range(step+1,cap+1):
            begun=time.time();torch.cuda.reset_peak_memory_stats(rank);sample_rows=rows[4*(update-1):4*update]
            assert len(sample_rows)==4 and len({r['source_group'] for r in sample_rows})==4 and all(r['domain']=='ulong' for r in sample_rows)
            before=state(c);c['optimizer'].zero_grad(set_to_none=True);total=[torch.zeros_like(p) for p in c['params']];records=[];data_wait=0;component_total={k:[torch.zeros_like(p) for p in c['params']] for k in TERMS}
            for row in sample_rows:
                key,path,receipt=cache_paths(row,row['external_qp'],True)
                assert path.exists() and receipt.exists(),('REQUIRED_WINDOW_CACHE_MISSING',key)
                t=time.time();a=load(receipt);assert a['status']=='PASS' and a['cache_identity']==key and sha(path)==a['cache_sha256']
                data=torch.load(path,map_location='cpu',weights_only=True,mmap=True);data_wait+=time.time()-t
                assert data['info']['source_row']==row and data['info']['training'] and data['info']['actual_qps']==row['actual_frame_qps']
                gradients,stats,components=window_gradients(c,data,lambda_temp,return_components=True)
                for k,gg in components.items():
                    for dst,g in zip(component_total[k],gg):dst.add_(g)
                records.append(dict(actual_frame_qps=data['info']['actual_qps'],sample_key=row['sample_key'],source_group=row['source_group'],identity=row['identity'],external_QP=row['external_qp'],source_frame_indices=row['source_frame_indices'],crop=[row['crop_x'],row['crop_y'],256,256],codec_cache_identity=key,**stats))
                del data,gradients,components
            valid_temporal_windows=sum(r['frames_temporal_valid']>0 for r in records)
            component_total,total=average_components(component_total,valid_temporal_windows)
            norm=gradient_norm(total);assert math.isfinite(norm) and norm>0
            group_norm=grouped_norms(c,total)
            effective_batch_angles={}
            if update%50==0 or update in [1,20]:
                spatial=[sum(component_total[k][i] for k in TERMS[:3]) for i in range(len(total))]
                direct=[component_total['temp_RGB_weighted'][i]+component_total['temp_Sobel_weighted'][i] for i in range(len(total))]
                all_temporal=[direct[i]+component_total['RAFT_weighted'][i] for i in range(len(total))]
                sn=gradient_norm(spatial)
                for label,vector in [('direct_MC_temporal',direct),('RAFT_plus_MC_temporal',all_temporal)]:
                    vn=gradient_norm(vector);dot=float(torch.stack([(a.double()*b.double()).sum() for a,b in zip(spatial,vector)]).sum())
                    cosine=dot/(sn*vn) if sn*vn>0 else None
                    effective_batch_angles[label]=dict(spatial_L2=sn,temporal_L2=vn,cosine=cosine,degrees=math.degrees(math.acos(max(-1.,min(1.,cosine)))) if cosine is not None else None)
            for p,g in zip(c['params'],total):p.grad=g
            clipped=torch.nn.utils.clip_grad_norm_(c['params'],1.,error_if_nonfinite=True)
            assert abs(float(clipped)-norm)<=max(1e-5,norm*1e-4)
            c['optimizer'].step()
            assert all(int(s['step'])==update for s in c['optimizer'].state.values()),'NEW_OBJECTIVE_OPTIMIZER_STEP_MISMATCH'
            frozen.check();after=state(c);changes={kind:0. for kind in ['TCG','LoRA','Bridge_mlp3']};base_change={kind:0. for kind in changes};base_norm={kind:0. for kind in changes}
            for name,t in after.items():
                assert torch.isfinite(t).all()
                kind='LoRA' if '.lora.' in name else 'TCG' if '.decoder.tcg_' in name else 'Bridge_mlp3';changes[kind]+=float((t-before[name]).double().square().sum())
                initial=c['initial_trainable'][name].cpu();base_change[kind]+=float((t-initial).double().square().sum());base_norm[kind]+=float(initial.double().square().sum())
            changes={k:math.sqrt(v) for k,v in changes.items()}
            assert changes['TCG']>0 and changes['LoRA']>0
            if group in 'BD':assert changes['Bridge_mlp3']>0
            assert_reset(c);torch.cuda.synchronize(rank)
            receipt=dict(experiment_group=group,disposable_engineering_update=smoke,additional_step=update,source_plan_cursor=4*update,sample_keys=[r['sample_key'] for r in sample_rows],
                samples=records,batch_temporal_valid_windows=valid_temporal_windows,batch_temporal_empty_windows=4-valid_temporal_windows,gradient_norm=norm,group_gradient_norms=group_norm,global_component_gradient_norms={k:grouped_norms(c,v) for k,v in component_total.items()},post_clip_norm=gradient_norm([p.grad for p in c['params']]),parameter_change_L2=changes,parameter_change_from_A2000={k:dict(absolute_L2=math.sqrt(v),relative_L2=math.sqrt(v/base_norm[k]) if base_norm[k]>0 else None) for k,v in base_change.items()},effective_batch_spatial_temporal_gradient_angles=effective_batch_angles,trainable_SHA256=tensor_hash(c['named']),frozen_state_SHA256=frozen.digest,
                frozen_exact_bytes_checked=True,buffers_reset=True,optimizer_steps=[int(s['step']) for s in c['optimizer'].state.values()],loss_normalization='spatial/RAFT over15 frames per window then4 windows; temporal over valid frames and nonempty windows',
                seconds=time.time()-begun,data_wait_seconds=data_wait,samples_per_second=4/(time.time()-begun),peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),peak_reserved_bytes=torch.cuda.max_memory_reserved(rank),physical_GPU=physical_gpu(rank),logical_rank_samples=4,time=time.time())
            saved=checkpoint(c,update,receipt,protocol_hash,code_hash)
            # Commit point is this atomic latest checkpoint. A missing external
            # receipt after interruption is recovered from its identical payload.
            atomic_torch(latest,saved);dump(root/f'receipts/{update:04d}.json',receipt)
            if update%50==0 or update==cap:atomic_torch(root/f'checkpoints/step_{update:04d}.pt',saved)
            dump(root/'progress.json',dict(status='RUNNING',committed_updates=update,next_cursor=4*update,time=time.time(),physical_GPU=physical_gpu(rank),last_seconds=receipt['seconds']))
            total_losses=[r['total_loss'] for r in records if r['total_loss'] is not None]
            event('smoke_update_committed' if smoke else 'formal_update_committed',group=group,step=update,loss=sum(total_losses)/len(total_losses) if total_losses else None,gradient_norm=norm,seconds=receipt['seconds'])
            del saved,receipt,before,after,total,component_total
            if not smoke and update%100==0:
                frozen.full()
                from evaluation import evaluate_group
                # Release training and RAFT models before full-resolution scoring.
                del c,frozen;torch.cuda.empty_cache()
                evaluate_group(group,update,rank)
                from summarize import screening,should_stop
                screening();stop_reason='PRESET_NO_TEMPORAL_GAIN_AND_SPATIAL_REGRESSION_200' if update==200 and should_stop(group) else None
                if stop_reason:break
                if update==cap:break
                c=construct(group,True,rank);saved=torch.load(latest,map_location='cpu',weights_only=True);restore(c,saved);restore_rng(saved['RNG']);del saved
                frozen=FrozenGuard(frozen_items(c))
        if smoke:
            frozen.full();stop_reason='DISPOSABLE_20_UPDATES_COMPLETED'
        elif stop_reason is None:stop_reason='MAX_300_UPDATES_REACHED'
        done=dict(status='PASS',group=group,committed_updates=update,stop_reason=stop_reason,disposable=smoke,final_checkpoint=str(root/f'checkpoints/step_{update:04d}.pt'),time=time.time())
        dump(root/'completion.json',done)
        return done

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',choices=GROUPS,required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:run(a.group,a.rank,a.smoke)
    except Exception as e:
        env();dump(ROOT/a.group/('smoke' if a.smoke else 'training')/'failure.json',dict(status='FAILED',exception=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
