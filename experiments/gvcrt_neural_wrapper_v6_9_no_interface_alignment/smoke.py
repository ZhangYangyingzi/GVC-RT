"""Disposable backward check: direct L_B equals V6.8b with no teacher/term."""
import argparse,random,gc,math,traceback
from v68_io import *
from model_runtime import setup,restore_rng,norm
from objective import Objective

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True)
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    torch.set_num_threads(2);cfg=frozen();device=torch.device('cuda:0')
    plan=load(ROOT/'audits/continuation_training_plans.json')['plans'][0]
    reference=module('v69_v68b_zero_alignment_reference',V68B/'objective.py')
    measurements={};expected_grads=expected_loss=None
    for label in ('baseline_without_teacher','direct_L_B'):
        v4,im,pm,models,opt,quality,state=setup(device)
        assert all(p.requires_grad for m in models.values() for p in m.parameters())
        rng=random.Random();rng.setstate(state['sample_rng_state']);frames,actual_plan=sample_clip(rng,device,v4)
        assert all(actual_plan[k]==plan[k] for k in ('sample_id','start','crop_x','crop_y','frame_sha256','temporal_indices'))
        opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        fn=(reference.Objective(v4,im,pm,models,quality,cfg,teacher=None,weight=0.0)
            if label=='baseline_without_teacher' else Objective(v4,im,pm,models,quality,cfg))
        sums,qps=fn(frames,plan['external_qp']);assert qps==plan['actual_qps']
        clean={k:v for k,v in sums.items() if v is not None}
        gradients={k:[p.grad.detach().cpu().clone() if p.grad is not None else None
                      for p in m.parameters()] for k,m in models.items()}
        norms={k:norm(v) for k,v in gradients.items()}
        assert all(math.isfinite(v) and v>0 for v in norms.values())
        ids={id(p) for m in models.values() for p in m.parameters()}
        assert all(p.grad is None and not p.requires_grad for root in (im,pm)
                   for p in root.parameters() if id(p) not in ids)
        if expected_grads is None:expected_grads=gradients;expected_loss=clean
        else:
            assert clean==expected_loss
            for k in gradients:
                assert all((a is None and b is None) or
                     (a is not None and b is not None and torch.equal(a,b))
                     for a,b in zip(gradients[k],expected_grads[k]))
            assert not any('align' in k or 'interface' in k for k in clean)
        compression=v4.compression_hash(im,pm);params=[p for m in models.values() for p in m.parameters()]
        torch.nn.utils.clip_grad_norm_(params,cfg['optimizer']['grad_clip']);opt.step()
        assert v4.compression_hash(im,pm)==compression==cfg['compression_hash']
        measurements[label]=dict(loss=clean,gradient_norms=norms,gradient_hash=state_hash(gradients),
            max_memory_reserved=torch.cuda.max_memory_reserved(),compression_frozen=True)
        fn.close();del fn,models,opt,quality,state,im,pm,frames,params,gradients
        gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'audits/gpu_memory_smoke.json',dict(status='PASS',gpu=args.gpu,**measurements))
    dump(ROOT/'audits/objective_smoke.json',dict(status='PASS',gpu=args.gpu,
        exact_L_B_loss_equality=True,exact_L_B_gradient_equality=True,
        no_alignment_computation_in_new_objective=True,no_formal_checkpoint_written=True,
        wrapper_bridge_generator_gradients_verified=True,compression_frozen=True,measurements=measurements))
    print('OBJECTIVE SMOKE PASS',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump(ROOT/'audits/objective_smoke.json',dict(status='FAIL',OOM='out of memory' in str(exc).lower(),traceback=traceback.format_exc()));raise
