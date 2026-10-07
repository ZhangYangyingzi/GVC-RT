"""Disposable source/equivalence/autograd/AdamW/native-codec checks."""
import argparse,random,gc,traceback
from v68_io import *
from model_runtime import *
from objective import Objective
from train import atomic_torch,gradients

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    torch.set_num_threads(2);device=torch.device('cuda:0');cfg=frozen();expected=load(ROOT/'audits/continuation_training_plans.json')['plans'][0]
    inventory_rows=[];evidence={};reference=None;peak=0
    for label,joint,lr in [('F',False,0.),('J_zero',True,0.),('J',True,1e-6)]:
        v4,im,pm,models,opt,quality,state=setup(device,joint=joint)
        initial={k:v4.module_hash(m) for k,m in models.items()};ih=v4.module_hash(im);qh=[v4.module_hash(m) for m in quality]
        dump(ROOT/'audits/initialization_audit.json',dict(status='PASS',I_hash=ih,source_checkpoint_sha256=sha(SOURCE),initial_module_hashes=initial,optimizer=optimizer_summary(state['optimizer']),plans_sha256=sha(ROOT/'audits/continuation_training_plans.json')))
        core=core_parameters(pm,models);before={n:p.detach().cpu().clone() for n,p in core.items()}
        if joint:opt.param_groups[-1]['lr']=lr
        inventory_rows.extend(inventory(im,pm,models,opt,quality,label))
        rng=random.Random();rng.setstate(state['sample_rng_state']);frames,plan=sample_clip(rng,device,v4)
        assert all(plan[k]==expected[k] for k in ('sample_id','start','crop_x','crop_y','frame_sha256'))
        fn=Objective(v4,im,pm,models,quality,cfg)
        quant=[]
        h1=pm.dec.register_forward_pre_hook(lambda m,args:quant.append(('y',args[0].detach().cpu().clone())))
        h2=pm.hyper_enc.register_forward_hook(lambda m,args,out:quant.append(('z',out.detach().round().clamp(-128,127).cpu().clone())))
        opt.zero_grad(set_to_none=True);restore_rng(state);torch.cuda.reset_peak_memory_stats()
        result,actual=fn(frames,expected['external_qp'],measure=joint,capture=True)
        if label=='F':
            source_row=json.loads((V610/'training_logs/C_plain.jsonl').read_text().splitlines()[0])
            assert all(result[k]==source_row[k] for k in result),'F differs from recorded V6.10 C first update'
        h1.remove();h2.remove();assert actual==expected['actual_qps']
        ps=[p for m in models.values() for p in m.parameters()]
        gn=norm([p.grad for p in core.values()]);core_detail=gradients(core)
        if joint:
            assert math.isfinite(gn) and gn>0
            assert all(math.isfinite(v) and v>0 for term in fn.gradient_measurements.values() for v in term)
        torch.nn.utils.clip_grad_norm_(ps,1.)
        if joint:torch.nn.utils.clip_grad_norm_(list(core.values()),1.)
        opt.step();peak=max(peak,torch.cuda.max_memory_reserved())
        updated={k:{n:p.detach().cpu().clone() for n,p in m.state_dict().items()} for k,m in models.items()}
        equal_forward=True;maxdiff={};exact_updates=True
        if reference is None:
            reference=dict(result=result,actual=actual,quant=quant,outputs=fn.outputs,updated=updated)
        else:
            assert result==reference['result'] and actual==reference['actual']
            assert len(quant)==len(reference['quant']) and all(n==m and torch.equal(x,y) for (n,x),(m,y) in zip(quant,reference['quant']))
            assert all(torch.equal(x,y) for x,y in zip(fn.outputs,reference['outputs']))
            for k,ms in updated.items():
                maxdiff[k]=max(float((t-reference['updated'][k][n]).abs().max()) for n,t in ms.items())
                exact_updates &= all(torch.equal(t,reference['updated'][k][n]) for n,t in ms.items())
                assert all(torch.allclose(t,reference['updated'][k][n],atol=1e-7,rtol=1e-6) for n,t in ms.items()),(label,k,maxdiff[k])
        changed=[n for n,p in core.items() if not torch.equal(p.detach().cpu(),before[n])]
        if lr==0:assert not changed
        else:
            assert changed and all(p.dtype==torch.float32 for p in core.values())
            assert all(v.dtype==torch.float32 for p in core.values() if p in opt.state for v in opt.state[p].values() if torch.is_tensor(v))
        assert ih==v4.module_hash(im) and qh==[v4.module_hash(m) for m in quality]
        evidence[label]=dict(initial_reconstruction_quantization_loss_equal=equal_forward,loss_components=result,PBG_update_bitwise_equal=exact_updates,PBG_update_max_abs_difference=maxdiff,core_gradient_norm=gn,changed_core_parameters=changed,core_gradient_modules=core_detail,base_and_rate_gradient_norms=fn.gradient_measurements,I_and_quality_unchanged=True,precision=precision_metadata(im,pm,models,v4))
        if label=='J':
            cp=ROOT/'audits/disposable_joint.pt';payload=inference_state(im,pm,models,v4);atomic_torch(cp,payload)
            hashes={k:v4.module_hash(m) for k,m in models.items()}
        fn.close();del fn,v4,im,pm,models,opt,quality,state,core,ps,before,updated,quant
        gc.collect();torch.cuda.empty_cache();print('CHECK',label,'PASS',flush=True)
    write(ROOT/'audits/parameter_groups.csv',inventory_rows)
    # Exact old C math has no modified FP32 core operations. Half forward is identical.
    # Reuse requires exact P/B/G update equality, not only allclose.
    reuse=evidence['J_zero']['PBG_update_bitwise_equal']
    runtime=engine().Runtime(dict(experiment='B',methods={'smoke_J':['smoke_J','smoke_J']},checkpoints={'smoke_J':dict(path=str(cp),module_hashes=hashes)},force_zero_thres=.12),'smoke_J',device)
    r,_=runtime.run([x.cpu() for x in frames],expected['external_qp'],ROOT/'parts/smoke_joint.bin',ROOT/'parts/smoke_joint.npz')
    assert r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass'] and r['full_P_checkpoint_loaded']
    dump(ROOT/'audits/objective_smoke.json',dict(status='PASS',gpu=a.gpu,reuse_F=reuse,peak_reserved=peak,checks=evidence,native_codec=r,formal_updates_consumed=0,exact_loss_source=str(V66/'objective.py'),code_hashes={p.name:sha(p) for p in ROOT.glob('*.py')}))
    print('ALL IMPLEMENTATION CHECKS PASS; reuse_F',reuse,flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        record=dict(status='FAIL',OOM='out of memory' in str(e).lower(),traceback=traceback.format_exc())
        dump(ROOT/'audits/objective_smoke.json',record);dump(ROOT/'logs/failures'/f'smoke_{time.time_ns()}.json',record);raise
