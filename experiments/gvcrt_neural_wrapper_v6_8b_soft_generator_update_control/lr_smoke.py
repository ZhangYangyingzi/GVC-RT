"""Three independent source restores and real backward/update comparisons."""
import argparse,random,math,traceback,gc,copy
from v68_io import *
from model_runtime import *
from objective import Objective
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);cfg=frozen();device=torch.device('cuda:0');expected=load(ROOT/'audits/continuation_training_plans.json')['plans'][0]
    results={};reference_grads=None;reference_updated=None;reference_loss=None;memory={}
    for label,lr in [('C1_reference',5e-7),('G50',2.5e-7),('G25',1.25e-7)]:
        v4,im,pm,models,opt,quality,state=setup(device);teacher=Teacher(v4,device);restore_rng(state)
        assert optimizer_summary(opt.state_dict())==load(ROOT/'audits/source_B1000_audit.json')['optimizer']
        opt.param_groups[2]['lr']=lr
        assert all(p.requires_grad for m in models.values() for p in m.parameters())
        rng=random.Random();rng.setstate(state['sample_rng_state']);frames,plan=sample_clip(rng,device,v4)
        assert all(plan[k]==expected[k] for k in ('sample_id','start','crop_x','crop_y','frame_sha256'))
        torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True)
        fn=Objective(v4,im,pm,models,quality,cfg,teacher,load(ROOT/'audits/interface_alignment_calibration.json')['lambda_align'])
        sums,qs=fn(frames,expected['external_qp']);assert qs==expected['actual_qps']
        grads={k:[p.grad.detach().cpu().clone() if p.grad is not None else None for p in m.parameters()] for k,m in models.items()}
        norms={k:norm(g) for k,g in grads.items()};assert all(math.isfinite(n) and n>0 for n in norms.values())
        if reference_grads is None:reference_grads=grads;reference_loss=sums
        else:
            assert sums==reference_loss,'LR changed pre-step forward values'
            for k in grads:
                assert len(grads[k])==len(reference_grads[k])
                assert all((x is None and y is None) or (x is not None and y is not None and torch.equal(x,y)) for x,y in zip(grads[k],reference_grads[k])),'LR changed gradient tensors'
        before_params={k:[p.detach().cpu().clone() for p in m.parameters()] for k,m in models.items()}
        params=[p for m in models.values() for p in m.parameters()]
        torch.nn.utils.clip_grad_norm_(params,cfg['optimizer']['grad_clip']);opt.step()
        updated={k:[p.detach().cpu().clone() for p in m.parameters()] for k,m in models.items()}
        moves={}
        for k,m in models.items():
            moves[k]=norm([p.detach().cpu()-b for p,b in zip(m.parameters(),before_params[k])])
        if reference_updated is None:reference_updated=updated
        else:
            for k in ('wrapper','bridge'):assert all(torch.equal(x,y) for x,y in zip(updated[k],reference_updated[k]))
        assert v4.compression_hash(im,pm)==cfg['compression_hash'];teacher.assert_frozen();assert teacher.hash()==teacher.hashes
        memory[label]={k:int(getattr(torch.cuda,k)()) for k in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved')}
        results[label]=dict(generator_lr=lr,loss=sums,gradient_norms=norms,gradient_hash=state_hash(grads),module_update_norms=moves,teacher_frozen=True,compression_frozen=True)
        print('SMOKE',label,'update_norms',moves,flush=True)
        fn.close();teacher.capture.close()
        del fn,teacher,models,opt,quality,state,im,pm,frames,params,grads,updated
        gc.collect();torch.cuda.empty_cache()
    assert 0<results['G25']['module_update_norms']['generator']<results['G50']['module_update_norms']['generator']<results['C1_reference']['module_update_norms']['generator']
    dump(ROOT/'audits/gpu_memory_smoke.json',dict(status='PASS',gpu=a.gpu,**memory))
    dump(ROOT/'audits/lr_control_smoke.json',dict(status='PASS',gpu=a.gpu,pre_step_losses_exactly_identical=True,all_gradient_tensors_exactly_identical=True,wrapper_updates_exactly_identical=True,bridge_updates_exactly_identical=True,generator_update_order_verified=True,disposable_only=True,no_formal_checkpoint_written=True,results=results))
if __name__=='__main__':
    try:main()
    except Exception as e:
        dump(ROOT/'audits/lr_control_smoke.json',dict(status='FAIL',OOM='out of memory' in str(e).lower(),traceback=traceback.format_exc()));raise
