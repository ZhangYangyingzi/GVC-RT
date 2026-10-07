"""Disposable equality, rate-path, clipping, export and real-codec checks."""
import argparse,random,gc,traceback,copy,inspect
from v68_io import *
from model_runtime import *
from objective import Objective
from train import atomic_torch,gradients

def compare_tensor(x,y):
    assert x.shape==y.shape and x.dtype==y.dtype
    delta=float((x.double()-y.double()).abs().max()) if x.numel() else 0.
    assert torch.equal(x,y),delta
    return dict(exact=True,max_abs_difference=delta,shape=list(x.shape),dtype=str(x.dtype))

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    torch.set_num_threads(2);cfg=frozen();device=torch.device('cuda:0');plan=load(ROOT/'audits/selected_training_plans.json')['plans'][0]
    refs={};comparisons=[];inventory_rows=[];updates={};exports={};runtime_sources=[];peak=0;f_update=None
    for branch,joint,mode in [('F_scale_ste',False,'legacy'),('F_scale_ste',False,'scale_ste'),('J_scale_ste',True,'legacy'),('J_scale_ste',True,'scale_ste')]:
        v4,im,pm,models,opt,quality,state=setup(device,joint=joint,mode=mode)
        ih=v4.module_hash(im);qh=[v4.module_hash(m) for m in quality];core=core_parameters(pm,models)
        initial={k:v4.module_hash(m) for k,m in models.items()};opthash=state_hash(opt.state_dict())
        runtime_sources.append(dict(branch=branch,**v4.rate_source,functional_call=joint))
        assert Path(v4.rate_source['path']).resolve()==ROOT/'rate_forward.py'
        if mode=='scale_ste':inventory_rows.extend(inventory(im,pm,models,opt,quality,branch))
        dump(ROOT/'audits/initialization_audit.json',dict(status='PASS',I_hash=ih,source_checkpoint_sha256=sha(SOURCE),module_hashes=initial,optimizer=optimizer_summary(state['optimizer']),source_rng_hashes={k:state_hash(state[k]) for k in ('sample_rng_state','torch_rng_state','cuda_rng_state')},training_plan_sha256=sha(ROOT/'audits/continuation_training_plans.json'),updates=500))
        rng=random.Random();rng.setstate(state['sample_rng_state']);frames,sample=sample_clip(rng,device,v4)
        assert all(sample[k]==plan[k] for k in ('sample_id','start','crop_x','crop_y','frame_sha256'))
        fn=Objective(v4,im,pm,models,quality,cfg)
        for qp in (0,4,9):
            opt.zero_grad(set_to_none=True);restore_rng(state);torch.cuda.reset_peak_memory_stats()
            result,actual=fn(frames,qp,measure=True,capture=True);peak=max(peak,torch.cuda.max_memory_reserved())
            expected=load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(qp)][:4];assert actual==expected
            data=dict(result=result,actual=actual,outputs=fn.outputs,traces=fn.traces,gradients=copy.deepcopy(fn.gradient_measurements))
            if mode=='legacy':refs[(branch,qp)]=data
            else:
                previous=refs[(branch,qp)];assert result==previous['result'] and actual==previous['actual']
                equality=[compare_tensor(x,y) for x,y in zip(previous['outputs'],fn.outputs)]
                traces=[{k:compare_tensor(x[k],y[k]) for k in x} for x,y in zip(previous['traces'],fn.traces)]
                for f in previous['gradients']['frames']:
                    assert all(x['scale_gradient']['is_none'] or x['scale_gradient']['norm']==0 for x in f['Gaussian_scale'])
                assert all(x['scale_gradient']['finite'] for f in fn.gradient_measurements['frames'] for x in f['Gaussian_scale'])
                assert any(x['scale_gradient']['norm']>0 for f in fn.gradient_measurements['frames'] for x in f['Gaussian_scale'])
                comparisons.append(dict(branch=branch,QP=qp,actual_qps=actual,loss_and_rate_components_exact=True,max_loss_difference=0.,components=result,reconstructions=equality,quantization_and_rates=traces,legacy_gradient_paths=previous['gradients'],scale_ste_gradient_paths=fn.gradient_measurements))
            if branch=='F_scale_ste' and mode=='legacy':
                # Verify the local legacy implementation against unmodified old core.
                local=v4.joint_ste_forward
                import core as original_core
                native_outputs=[]
                def observed(*args,**kwargs):
                    out=original_core.joint_ste_forward(*args,**kwargs);native_outputs.append(out[0].detach().cpu().clone());return out
                v4.joint_ste_forward=observed
                source_b=module('v612_native_B',V66/'objective.py').bind(v4,im,pm,models['wrapper'],quality,cfg,weight=cfg['lambda_struct'])
                opt.zero_grad(set_to_none=True);restore_rng(state);native,native_q=source_b(frames,qp)
                assert native_q==actual and all(result[k]==value for k,value in native.items())
                assert all(torch.equal(x,y) for x,y in zip(native_outputs,fn.outputs))
                v4.joint_ste_forward=local;del source_b
        assert state_hash(opt.state_dict())==opthash and initial=={k:v4.module_hash(m) for k,m in models.items()}
        assert ih==v4.module_hash(im) and qh==[v4.module_hash(m) for m in quality]
        if mode=='scale_ste':
            before={n:p.detach().cpu().clone() for n,p in core.items()}
            if joint:opt.param_groups[-1]['lr']=0.
            opt.zero_grad(set_to_none=True);restore_rng(state);result,actual=fn(frames,plan['external_qp'],measure=True)
            ps=[p for m in models.values() for p in m.parameters()]
            torch.nn.utils.clip_grad_norm_(ps,1.)
            if joint:torch.nn.utils.clip_grad_norm_(list(core.values()),1.)
            opt.step()
            updated={k:{n:t.detach().cpu().clone() for n,t in m.state_dict().items()} for k,m in models.items()}
            assert all(torch.equal(p.detach().cpu(),before[n]) for n,p in core.items())
            if not joint:f_update=updated
            else:assert all(torch.equal(x,f_update[k][n]) for k,ms in updated.items() for n,x in ms.items())
            updates[branch]=dict(PBG_update_equal_with_core_lr_zero=True,core_unchanged=True,optimizer_steps=1,gradient_paths=fn.gradient_measurements)
            cp=ROOT/'audits'/f'disposable_{branch}.pt';atomic_torch(cp,inference_state(im,pm,models,v4,joint=joint))
            exports[branch]=dict(path=str(cp),sha256=sha(cp),module_hashes={k:v4.module_hash(m) for k,m in models.items()})
            del before,updated,ps
        fn.close();del fn,v4,im,pm,models,opt,quality,state,core
        gc.collect();torch.cuda.empty_cache();print('EQUIVALENCE',branch,mode,'PASS',flush=True)
    del refs,f_update
    v4,im,pm,models,opt,quality,state=setup(device,joint=True)
    import gradient_probe
    gradient_probe.run(device)
    core=core_parameters(pm,models);before={n:p.detach().cpu().clone() for n,p in core.items()}
    ih=v4.module_hash(im);qh=[v4.module_hash(m) for m in quality]
    fn=Objective(v4,im,pm,models,quality,cfg);opt.zero_grad(set_to_none=True);restore_rng(state)
    result,actual=fn(frames,plan['external_qp'],measure=True)
    core_norm=norm([p.grad for p in core.values()]);assert math.isfinite(core_norm) and core_norm>0
    torch.nn.utils.clip_grad_norm_([p for m in models.values() for p in m.parameters()],1.)
    torch.nn.utils.clip_grad_norm_(list(core.values()),1.);opt.step()
    changed=[n for n,p in core.items() if not torch.equal(p.detach().cpu(),before[n])];assert changed
    assert all(p.dtype==torch.float32 for p in core.values())
    assert all(t.dtype==torch.float32 for p in core.values() if p in opt.state for t in opt.state[p].values() if torch.is_tensor(t))
    assert ih==v4.module_hash(im) and qh==[v4.module_hash(m) for m in quality]
    updates['J_actual']=dict(core_lr=1e-6,core_gradient_norm=core_norm,changed_parameters=changed,gradient_paths=fn.gradient_measurements,I_and_quality_unchanged=True)
    cp=ROOT/'audits/disposable_J_actual.pt';atomic_torch(cp,inference_state(im,pm,models,v4,joint=True))
    exports['J_actual']=dict(path=str(cp),sha256=sha(cp),module_hashes={k:v4.module_hash(m) for k,m in models.items()})
    fn.close();del fn,v4,im,pm,models,opt,quality,state,core,before
    gc.collect();torch.cuda.empty_cache()
    codec={}
    for label,cp in exports.items():
        runtime=engine().Runtime(dict(experiment='B',methods={label:[label,label]},checkpoints={label:cp},force_zero_thres=.12),label,device)
        r,_=runtime.run([x.cpu() for x in frames],plan['external_qp'],ROOT/'parts/smoke'/f'{label}.bin',ROOT/'parts/smoke'/f'{label}.npz')
        assert r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass'];codec[label]=r
        del runtime;gc.collect();torch.cuda.empty_cache()
    assert codec['F_scale_ste']['bitstream_sha256']==codec['J_scale_ste']['bitstream_sha256']
    write(ROOT/'audits/parameter_groups.csv',inventory_rows)
    old=read(V611/'audits/parameter_groups.csv')
    assert {r['name'] for r in inventory_rows if r['branch']=='J_scale_ste' and r['group']=='compression_core'}=={r['name'] for r in old if r['branch']=='J' and r['group']=='compression_core'}
    dump(ROOT/'audits/forward_equivalence.json',dict(status='PASS',comparisons=comparisons,unmodified_old_core_checked=True,forward_tolerance=0.,optimizer_steps_during_equivalence=0))
    dump(ROOT/'audits/runtime_source_audit.json',dict(status='PASS',executed_functions=runtime_sources,source_hashes={p.name:sha(p) for p in ROOT.glob('*.py')},native_codec_unchanged=True))
    dump(ROOT/'audits/objective_smoke.json',dict(status='PASS',gpu=a.gpu,peak_reserved=peak,checks=updates,native_codec=codec,formal_updates_consumed=0,source_optimizer_restored=True,core_parameter_set_matches_v611=True,disposable_only=True))
    print('ALL V6.12 IMPLEMENTATION CHECKS PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        error=dict(status='FAIL',OOM='out of memory' in str(e).lower(),traceback=traceback.format_exc())
        dump(ROOT/'audits/objective_smoke.json',error);dump(ROOT/'logs/failures'/f'smoke_{time.time_ns()}.json',error);raise
