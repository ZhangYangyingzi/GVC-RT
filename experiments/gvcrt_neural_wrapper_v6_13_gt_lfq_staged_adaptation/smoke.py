"""Disposable ori equality, B-weight-zero, frozen-generator and gradient gates."""
import argparse,gc,traceback
import torch
from v68_io import *
from model_runtime import *
from objective import Objective
from data import sample
from io_utils import command
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    command([PYTHON,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    cfg=frozen();torch.set_num_threads(2);device=torch.device('cuda:0');plan=load(ROOT/'manifests/training_plans.json')['plans'][0]
    refs={};exports={};reports={};inventories=[];peak=0
    for label,weight in [('A',0.),('B_zero',0.),('B_actual',.2)]:
        v4,im,pm,models,opt,quality,teacher=setup(device);core=core_parameters(pm,models)
        hashes={k:v4.module_hash(m) for k,m in models.items()};constants=dict(I=v4.module_hash(im),generator=hashes['generator'],teacher=v4.module_hash(teacher),quality=[v4.module_hash(m) for m in quality])
        if label=='A':
            initial=hashes;dump(ROOT/'audits/initialization_audit.json',dict(status='PASS',I_hash=constants['I'],native_checkpoints=cfg['init_checkpoints'],initial_model_hashes=hashes,optimizer_entries=len(opt.state),fresh_optimizer=True,teacher_checkpoint_sha256=cfg['teacher_checkpoint']['sha256']))
            write(ROOT/'audits/parameter_groups.csv',inventory(im,pm,models,opt,quality,teacher))
        else:assert hashes==initial
        frames,source_hashes=sample(plan,device)
        assert torch.equal(models['wrapper'](frames[0]),frames[0]) and not opt.state
        assert deployment_hash(im,pm)==cfg['compression_hash']
        if label!='B_actual':
            cp=ROOT/'audits'/f'disposable_{label}_step0.pt';atomic_torch(cp,inference_state(im,pm,models,v4));exports[label]=dict(path=str(cp),sha256=sha(cp),module_hashes=hashes)
        fn=Objective(v4,im,pm,models,quality,teacher,weight);observations={}
        for q in (0,4,9):
            opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
            values,actual=fn(frames,q,measure=True,capture=True);peak=max(peak,torch.cuda.max_memory_reserved())
            grad={g['name']:state_hash([p.grad for p in g['params']]) for g in opt.param_groups}
            assert actual==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][:4]
            assert all(o['reconstruction_latent_gradient']['norm']>0 and o['reconstruction_latent_gradient']['finite'] for o in fn.observations)
            assert all(x['finite'] for o in fn.observations for x in o['Gaussian_scale'])
            assert any(x['norm']>0 for o in fn.observations for x in o['Gaussian_scale'])
            assert all(o['cosine']['bridge']['norm']>0 and o['cosine']['compression_core']['norm']>0 for o in fn.observations)
            assert all(p.grad is None and not p.requires_grad for p in models['generator'].parameters())
            observations[str(q)]=fn.observations
            if label=='A':refs[q]=dict(values=values,actual=actual,grad=grad,outputs=fn.outputs)
            elif label=='B_zero':
                assert refs[q]['values']==values and refs[q]['actual']==actual and refs[q]['grad']==grad
                assert all(torch.equal(x,y) for x,y in zip(refs[q]['outputs'],fn.outputs))
        assert hashes=={k:v4.module_hash(m) for k,m in models.items()} and not opt.state
        opt.zero_grad(set_to_none=True);values,actual=fn(frames,plan['external_qp'])
        wb=[p for k in ('wrapper','bridge') for p in models[k].parameters()];torch.nn.utils.clip_grad_norm_(wb,1.);torch.nn.utils.clip_grad_norm_(list(core.values()),1.);opt.step()
        update_hash=state_hash([pm.state_dict(),models['wrapper'].state_dict(),opt.state_dict()])
        if label=='A':reference_update=update_hash
        elif label=='B_zero':assert update_hash==reference_update
        assert constants==dict(I=v4.module_hash(im),generator=v4.module_hash(models['generator']),teacher=v4.module_hash(teacher),quality=[v4.module_hash(m) for m in quality])
        reports[label]=dict(status='PASS',gradient_checks=observations,frozen_modules_unchanged=True,one_disposable_update=True,update_hash=update_hash)
        fn.close();del fn,v4,im,pm,models,opt,quality,teacher,core,wb;gc.collect();torch.cuda.empty_cache()
        print('SMOKE',label,'PASS',flush=True)
    import gradient_probe
    gradient_probe.run(device)
    codec={}
    for label in ('original','A','B_zero'):
        pair=[None,None] if label=='original' else [label,label]
        runtime=engine().Runtime(dict(experiment='B',methods={label:pair},checkpoints={} if label=='original' else {label:exports[label]},force_zero_thres=.12),label,device)
        results=[]
        for q in (0,4,9):
            r,_=runtime.run([x.cpu() for x in frames],q,ROOT/'bitstreams/smoke'/f'{label}_{q}.bin',ROOT/'features/smoke'/f'{label}_{q}.npz')
            assert r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass'];results.append(r)
        codec[label]=results;del runtime;gc.collect();torch.cuda.empty_cache()
    assert all(codec['original'][i]['bitstream_sha256']==codec['A'][i]['bitstream_sha256']==codec['B_zero'][i]['bitstream_sha256'] and codec['original'][i]['reconstruction_sha256']==codec['A'][i]['reconstruction_sha256']==codec['B_zero'][i]['reconstruction_sha256'] for i in range(3))
    dump(ROOT/'audits/objective_smoke.json',dict(status='PASS',gpu=a.gpu,peak_reserved=peak,checks=reports,step0_native_codec=codec,step0_matches_ori=True,B_zero_common_loss_gradients_update_exact=True,generator_frozen_but_latent_gradients=True,teacher_frozen_no_rng_change=True,formal_updates_consumed=0))
    print('ALL PREFLIGHT SMOKES PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        err=dict(status='FAIL',traceback=traceback.format_exc());dump(ROOT/'audits/objective_smoke.json',err);dump(ROOT/'logs/failures'/f'smoke_{time.time_ns()}.json',err);raise
