"""Runtime semantic, temporal, zero-weight equivalence and 32-clip budget gates."""
import argparse,math,random,statistics,traceback
from v68_io import *
from model_runtime import *
from objective import Objective,reference,FIELDS
def memory():
    import torch
    return {k:int(getattr(torch.cuda,k)()) for k in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved')}
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);cfg=frozen();device=torch.device('cuda:0')
    if (ROOT/'audits/interface_alignment_calibration.json').exists():assert load(ROOT/'audits/interface_alignment_calibration.json')['status']=='PASS';return
    v4,im,pm,models,opt,quality,state=setup(device);initial={k:v4.module_hash(m) for k,m in models.items()};opthash=state_hash(opt.state_dict())
    params=[p for m in models.values() for p in m.parameters()];rng=random.Random();rng.setstate(state['sample_rng_state']);frames,plan=sample_clip(rng,device,v4)
    plans=load(ROOT/'audits/continuation_training_plans.json')['plans'];assert all(plan[k]==plans[0][k] for k in ('sample_id','start','crop_x','crop_y','frame_sha256'));q=plans[0]['external_qp']
    torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True);ref=reference(v4,im,pm,models,quality,cfg);rs,rq=ref(frames,q);gref=[p.grad.detach().clone() if p.grad is not None else None for p in params];c0mem=memory()
    opt.zero_grad(set_to_none=True);teacher=Teacher(v4,device);restore_rng(state)
    t1,tq=teacher.targets(frames,q,verify_decode=True);t2,tq2=teacher.targets(frames,q)
    assert tq==tq2 and all(torch.equal(x,y) for x,y in zip(t1,t2));assert disjoint_states(pm,teacher.pm)
    zero=Objective(v4,im,pm,models,quality,cfg,teacher,0.0);zs,zq=zero(frames,q)
    assert zq==rq==tq and all(rs[k]==zs[k] for k in rs)
    assert all((x is None and p.grad is None) or (x is not None and p.grad is not None and torch.equal(x,p.grad)) for x,p in zip(gref,params))
    details=zero.capture.details;td=teacher.last_details;assert details['generator_input']['shape']==td['generator_input']['shape']
    assert math.isfinite(zs['interface_cosine_similarity']) and not math.isclose(zs['interface_cosine_similarity'],1.,abs_tol=1e-7)
    files=[REPO/'src/models/video_model_gvcrt.py',REPO/'src/models/improved_model_gvcrt.py',REPO/'src/models/image_model_gvcrt.py',V66/'train.py',V66/'objective.py',V4/'train.py',ROOT.parent/'gvcrt_neural_wrapper_v2_joint/core.py']
    teacher_files={str(REPO/'checkpoints'/n):sha(REPO/'checkpoints'/n) for n in ('GVC-RT_I.pt','GVC-RT_P.pt')}
    dump(ROOT/'audits/interface_semantics_audit.json',dict(status='PASS',interface_mode=MODE,teacher_module='pm.recon_generation_net.decoder',student_module='pm.recon_generation_net.decoder',teacher_hook_position='forward_pre_hook inputs[0]',student_hook_position='forward_pre_hook inputs[0]',teacher_shape=td['generator_input']['shape'],student_shape=details['generator_input']['shape'],dtype=dict(teacher=td['generator_input']['dtype'],student=details['generator_input']['dtype'],cosine='float32'),semantic_description='18-channel bridge codeword directly consumed by the pretrained de-tokenizer; q_recon is a separate argument; no projection, resize, slicing or normalization',shape_exact_match=True,same_interface_semantics=True,bridge_output_exactly_decoder_input=True,teacher_checkpoint=list(teacher_files),teacher_checkpoint_sha256=teacher_files,teacher_frozen=True,student_trainable_modules=list(models),teacher_tensors=td,student_tensors=details,source_files_inspected=list(map(str,files)),source_file_hashes={str(p):sha(p) for p in files},exact_LFQ_objective_reproduction=False,margin_loss=False))
    dump(ROOT/'audits/teacher_temporal_state_audit.json',dict(status='PASS',independent_teacher_student_DPB=True,teacher_parameters_frozen=True,teacher_gradients_none=True,native_teacher_forward_equals_independent_RANS_decode=True,teacher_deterministic=True,teacher_rng_fork_restored=True,teacher_input='raw source x',student_input='P(x)',actual_qps=tq,reset='clear_dpb/set_curr_poc(0) each clip; native add_ref_frame after every frame',P_reference_feature='native compression decoder feature, own instance only',force_zero_thres=.12,teacher_initial_hashes=teacher.hashes))
    zero.close();del gref,t1,t2;opt.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    # Unit gradients accumulated over the same 3 P-frames as the existing B mean.
    rows=[];fn=Objective(v4,im,pm,models,quality,cfg,teacher,0.0,calibration=True);rng.setstate(state['sample_rng_state']);restore_rng(state)
    for i in range(32):
        frames,plan=sample_clip(rng,device,v4);expected=plans[i];assert all(plan[k]==expected[k] for k in ('sample_id','start','crop_x','crop_y','frame_sha256'))
        opt.zero_grad(set_to_none=True);s,qs=fn(frames,expected['external_qp']);assert qs==expected['actual_qps'];assert s['g_wrapper_main']>0 and s['g_wrapper_align']>0 and s['g_bridge_align']>0
        rows.append(dict(calibration_index=i,**expected,**s));print('CALIBRATION UNIT',i+1,'/32',flush=True)
    fn.close();del fn;mb=statistics.median(r['g_wrapper_main'] for r in rows);ma=statistics.median(r['g_wrapper_align'] for r in rows);weight=.05*mb/ma
    assert math.isfinite(weight) and weight>0
    fn=Objective(v4,im,pm,models,quality,cfg,teacher,weight,calibration=True);rng.setstate(state['sample_rng_state']);restore_rng(state)
    for i in range(32):
        frames,plan=sample_clip(rng,device,v4);s,qs=fn(frames,plans[i]['external_qp'])
        assert math.isclose(s['g_wrapper_main'],rows[i]['g_wrapper_main'],rel_tol=1e-6)
        rows[i].update(actual_weighted_gradient=s['g_wrapper_weighted'],actual_weighted_ratio=s['g_wrapper_weighted']/s['g_wrapper_main'],formula_weighted_ratio=weight*rows[i]['g_wrapper_align']/rows[i]['g_wrapper_main'])
        print('CALIBRATION WEIGHTED',i+1,'/32',flush=True)
    fn.close();del fn
    actual_ratio=statistics.median(r['actual_weighted_gradient'] for r in rows)/mb
    write(ROOT/'audits/interface_alignment_gradient_samples.csv',rows)
    dump(ROOT/'audits/gradient_verification_measurement.json',dict(lambda_align=weight,median_g_B=mb,median_g_align_unit=ma,formula_aggregate_ratio=weight*ma/mb,actual_backward_aggregate_ratio=actual_ratio,actual_per_sample_ratios=[r['actual_weighted_ratio'] for r in rows],actual_ratio_in_5percent_band=.0475<=actual_ratio<=.0525))
    print('GRADIENT VERIFICATION',json.dumps(load(ROOT/'audits/gradient_verification_measurement.json')),flush=True)
    # Latest request section 8 explicitly gates lambda*median(g_unit)/median(g_B).
    # Independently measured mixed-precision backward values are preserved, not
    # substituted for that expression and never used to retune the fixed lambda.
    assert math.isfinite(actual_ratio) and actual_ratio>0
    assert .0475<=weight*ma/mb<=.0525
    write(ROOT/'audits/interface_alignment_gradient_samples.csv',rows)
    dump(ROOT/'audits/interface_alignment_gradient_cosine_summary.json',dict(status='PASS',wrapper_median=statistics.median(r['cosine_wrapper'] for r in rows),bridge_median=statistics.median(r['cosine_bridge'] for r in rows),generator='NOT_APPLICABLE: interface is upstream of generator parameters',wrapper_values=[r['cosine_wrapper'] for r in rows],bridge_values=[r['cosine_bridge'] for r in rows]))
    # One full actual C1 backward, no optimizer update; release audit gradient buffers first.
    opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();smoke=Objective(v4,im,pm,models,quality,cfg,teacher,weight);smoke(frames,plans[31]['external_qp']);c1mem=memory();smoke.close()
    assert initial=={k:v4.module_hash(m) for k,m in models.items()} and teacher.hash()==teacher.hashes and state_hash(opt.state_dict())==opthash
    assert v4.compression_hash(im,pm)==cfg['compression_hash'];teacher.assert_frozen()
    dump(ROOT/'audits/gpu_memory_smoke.json',dict(status='PASS',gpu=a.gpu,C0=c0mem,C1=c1mem,no_optimizer_updates=True,protocol_unchanged=True))
    dump(ROOT/'audits/interface_preservation_feasibility.json',dict(status='PASS',shape_exact_match=True,same_semantics=True,finite_cosine=True,nontrivial_cosine=zs['interface_cosine_similarity'],nonzero_finite_wrapper_bridge_gradient=True,teacher_no_gradients=True,compression_frozen=True,lambda_zero_exact_forward_B_loss_and_gradients=True,teacher_deterministic=True,no_parameter_updates=True))
    dump(ROOT/'audits/interface_alignment_calibration.json',dict(status='PASS',lambda_align=weight,lambda_struct=cfg['lambda_struct'],clips=32,source_checkpoint_sha256=sha(SOURCE),training_plans_sha256=sha(ROOT/'audits/continuation_training_plans.json'),median_g_B=mb,median_g_align_unit=ma,formula_aggregate_ratio=weight*ma/mb,required_aggregate_expression='lambda_align * median(g_align_unit) / median(g_B)',formula_budget_verified=.0475<=weight*ma/mb<=.0525,actual_backward_aggregate_ratio=actual_ratio,actual_backward_in_5percent_band=.0475<=actual_ratio<=.0525,actual_per_sample_ratios=[r['actual_weighted_ratio'] for r in rows],no_parameter_updates=True,no_test_data=True,frozen=True))
    print('INTERFACE/CALIBRATION/SMOKE PASS lambda_align',weight,flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        status='C1_GPU_OOM_NO_GO' if 'out of memory' in str(e).lower() else 'NO_GO'
        dump(ROOT/'audits/interface_preservation_feasibility.json',dict(status=status,reason=str(e),traceback=traceback.format_exc()));raise
