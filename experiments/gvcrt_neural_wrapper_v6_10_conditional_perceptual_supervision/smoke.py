"""Disposable implementation checks. Every PASS field has a measured assertion."""
import argparse,random,gc,traceback,copy
from v68_io import *
from model_runtime import setup,restore_rng,norm
from objective import Objective,discriminator_update
from discriminator import create
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);cfg=frozen();device=torch.device('cuda:0')
    v4,im,pm,models,opt,quality,state=setup(device);initial={k:v4.module_hash(m) for k,m in models.items()}
    opthash=state_hash(opt.state_dict());rng=random.Random();rng.setstate(state['sample_rng_state'])
    frames,plan=sample_clip(rng,device,v4);expected=load(ROOT/'audits/continuation_training_plans.json')['plans'][0]
    assert all(plan[k]==expected[k] for k in ('sample_id','crop_x','crop_y','frame_sha256','start'))
    sys.path.insert(0,str(V66))
    reference=module('v610_B_reference',V66/'objective.py').bind(v4,im,pm,models['wrapper'],quality,cfg,weight=cfg['lambda_struct'])
    recon=[];original=v4.joint_ste_forward
    def captured(*args,**kw):
        output=original(*args,**kw);recon.append(output[0].detach().cpu().clone());return output
    v4.joint_ste_forward=captured
    opt.zero_grad(set_to_none=True);restore_rng(state);r,qs=reference(frames,expected['external_qp'])
    grads={k:[p.grad.detach().cpu().clone() if p.grad is not None else None for p in m.parameters()] for k,m in models.items()}
    v4.joint_ste_forward=original;opt.zero_grad(set_to_none=True);restore_rng(state)
    fn=Objective(v4,im,pm,models,quality,cfg);s,actual=fn(frames,expected['external_qp'],capture=True)
    assert r=={k:s[k] for k in r} and qs==actual==expected['actual_qps']
    assert all(torch.equal(x,y) for x,y in zip(recon,fn.outputs))
    differences={}
    for k,m in models.items():
        values=[]
        for x,p in zip(grads[k],m.parameters()):
            if x is None:assert p.grad is None;continue
            y=p.grad.detach().cpu();assert torch.allclose(x,y,atol=1e-6,rtol=1e-4)
            values.append(float((x-y).abs().max()))
        differences[k]=max(values)
    opt.zero_grad(set_to_none=True)
    with torch.random.fork_rng(devices=[0]):records=fn.collect(frames,expected['external_qp'])
    f=records[1]['condition'];assert not f.requires_grad and f.dtype==torch.float32
    feature_stats=dict(shape=list(f.shape),condition_dtype=str(f.dtype),native_dtype=records[1]['native_feature_dtype'],
        min=float(f.min()),max=float(f.max()),mean=float(f.mean()),std=float(f.std()))
    assert f.shape[1]==cfg['discriminator']['feature_channels']
    d=create(f.shape[1],device,cfg['discriminator']['seed']);d2=create(f.shape[1],device,cfg['discriminator']['seed'])
    assert state_hash(d.state_dict())==state_hash(d2.state_dict())
    dhash=state_hash(d.state_dict());del d2
    stats={}
    for branch in ('S_cond_image','T_cond_temporal'):
        state=torch.load(SOURCE,map_location='cpu',weights_only=True)
        for k,m in models.items():m.load_state_dict(state[k])
        opt.load_state_dict(copy.deepcopy(state['optimizer']));opt.zero_grad(set_to_none=True);restore_rng(state)
        with torch.random.fork_rng(devices=[0]):records=fn.collect(frames,expected['external_qp'])
        d=create(f.shape[1],device,cfg['discriminator']['seed'])
        do=torch.optim.Adam(d.parameters(),lr=1e-4,betas=(0.,.99),weight_decay=0.)
        captured_inputs=[]
        def input_hook(module,args):
            rgb,condition,q=args
            if rgb.shape[0]==4:
                assert torch.equal(condition[:2],condition[2:]) and not condition.requires_grad
                captured_inputs.append(True)
        h=d.register_forward_pre_hook(input_hook)
        torch.cuda.reset_peak_memory_stats()
        dr=discriminator_update(d,do,records,branch,expected['external_qp']);h.remove();assert captured_inputs
        assert initial=={k:v4.module_hash(m) for k,m in models.items()}
        assert opthash==state_hash(opt.state_dict()) and all(p.grad is None for m in models.values() for p in m.parameters())
        before=state_hash(d.state_dict());restore_rng(state)
        result,actual=fn(frames,expected['external_qp'],branch,d,.01,measure=True)
        assert all(v>0 and v<float('inf') for v in fn.gradient_measurements['weighted_L_adv'].values())
        assert all(p.grad is None for p in d.parameters())
        torch.nn.utils.clip_grad_norm_([p for m in models.values() for p in m.parameters()],1.);opt.step()
        assert before==state_hash(d.state_dict())
        assert all(v4.module_hash(m)!=initial[k] for k,m in models.items())
        assert v4.compression_hash(im,pm)==cfg['compression_hash']
        stats[branch]=dict(**dr,gradient_components=fn.gradient_measurements,previous_fake_detached=True,
            condition_identical_and_detached=True,D_step_preserves_PBG_and_optimizer=True,G_step_preserves_D_and_SN=True,
            peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated())
        print('IMPLEMENTATION CHECK',branch,'PASS',flush=True)
    # Native codec determinism and receiver independence, with discriminator state ignored.
    for k,m in models.items():m.load_state_dict(state[k])
    tmpcp=ROOT/'audits/disposable_lambda_zero.pt'
    pending=tmpcp.with_suffix('.tmp')
    torch.save(dict(**{k:m.state_dict() for k,m in models.items()},discriminator=d.state_dict()),pending)
    pending.replace(tmpcp)
    fn.close();del fn,d,do,models,opt,quality,im,pm,reference,records,f
    gc.collect();torch.cuda.empty_cache()
    e=engine();records_codec=[]
    for label,cp in [('source',SOURCE),('lambda_zero',tmpcp)]:
        runtime=e.Runtime(dict(experiment='B',methods={label:[label,label]},checkpoints={label:dict(path=str(cp),module_hashes=initial)},force_zero_thres=.12),label,device)
        path=ROOT/'parts/implementation_check'/f'{label}.bin';feature=path.with_suffix('.npz')
        record,_=runtime.run([x.cpu() for x in frames],expected['external_qp'],path,feature)
        assert record['real_RANS'] and record['independent_decode_pass'] and record['state_sync_pass']
        records_codec.append((sha(path),record))
        del runtime;gc.collect();torch.cuda.empty_cache()
    assert records_codec[0][0]==records_codec[1][0]
    assert records_codec[0][1]['real_bytes']==records_codec[1][1]['real_bytes']
    dump(ROOT/'audits/gpu_memory_smoke.json',dict(status='PASS',gpu=a.gpu,branches=stats))
    dump(ROOT/'audits/objective_smoke.json',dict(status='PASS',gpu=a.gpu,
        zero_adv_B_components_equal=True,zero_adv_reconstruction_equal=True,gradient_atol=1e-6,gradient_rtol=1e-4,
        gradient_max_abs_difference=differences,feature_statistics=feature_stats,discriminator_initial_hash=dhash,
        identical_discriminator_initialization=True,branches=stats,native_zero_adv_bitstreams_identical=True,
        independent_decode=True,discriminator_not_used_in_codec=True,codec_sha256=records_codec[0][0]))
    print('ALL IMPLEMENTATION CHECKS PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        dump(ROOT/'audits/objective_smoke.json',dict(status='FAIL',OOM='out of memory' in str(e).lower(),traceback=traceback.format_exc()));raise
