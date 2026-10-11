"""No-update fixed-weight and separate FP32 gradient-restoration checks."""
import argparse,random,traceback,math
from io23 import *
from window_training import replay,make_runner
from gradient_tools import stats,dists_details

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args()
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);cfg=load(ROOT/'config.json');seed=cfg['seed']
    random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);frozen()
    command([PYTHON,'-B',str(Path(__file__).resolve()),*sys.argv[1:]])
    v4,im,pm,models,quality=load_training(torch.device('cuda:0'))
    params=[p for m in models.values() for p in m.parameters()]
    initial={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm);qh=[v4.module_hash(m) for m in quality]
    assert core==cfg['source_checkpoint']['compression_hash']
    row=load(ROOT/'plans/shared.json')['updates'][0];frames=replay(row,BRANCHES[0],torch.device('cuda:0'));q=row['external_qp']
    run=make_runner(v4,im,pm,models,quality)
    def clear():
        for p in params:p.grad=None
    def copyg():return [None if p.grad is None else p.grad.detach().cpu().clone() for p in params]
    def summary(gs):
        out={};offset=0
        for k,m in models.items():
            ps=list(m.parameters());n=len(ps);out[k]=stats(gs[offset:offset+n],ps);offset+=n
        return out
    # Execute the frozen reference function in its own IO namespace, read-only.
    oldio=module('weight23_reference_io',V22/'io21.py');ns=dict(oldio.__dict__)
    exec(compile((V22/'window_training.py').read_text().replace('from io21 import *',''),str(V22/'window_training.py'),'exec'),ns)
    oldrun=ns['make_runner'](v4,im,pm,models,quality)
    clear();ref,qs,trace,warmup,_=oldrun(frames,q,'B1_reconstruction');refg=copyg()
    clear();base,bq,bt,bw,_=run(frames,q,BRANCHES[0],rec_weight=0);bg=copyg()
    assert qs==bq==row['actual_qps'] and warmup==bw
    clear();_,_,_,_,cg=run(frames,q,BRANCHES[0],rec_weight=0,mode='calibrate',rec_scale=4096.)
    rg=[g.detach().cpu().clone() for g in cg['rec']];del cg
    norms=summary(rg);assert all(g['finite'] and g['effective_parameters']>0 for g in norms.values())
    for g,b,r in zip(refg,bg,rg):assert torch.allclose(g,b+REFERENCE_WEIGHT*r,atol=2e-7,rtol=2e-4)
    records={}
    for branch in BRANCHES:
        clear();loss,aq,at,aw,_=run(frames,q,branch) # Default lookup: checks production branch dispatch.
        grad=copyg();weight=WEIGHTS[branch];assert loss['branch_lambda_rec']==loss['lambda_rec']==weight
        assert aq==qs and aw==warmup
        assert all(x['reconstruction_sha256']==y['reconstruction_sha256'] and x['feature_sha256']==y['feature_sha256'] for x,y in zip(at,trace))
        for key in ('LPIPS','DISTS','D','rate_bpp','beta_rate','proxy_L1','reconstruction_L1'):
            assert loss[key]==base[key]==ref[key],(branch,key)
        assert math.isclose(loss['loss']-base['loss'],weight*loss['reconstruction_L1'],rel_tol=2e-5,abs_tol=2e-7)
        errors=[]
        for actual,b,r in zip(grad,bg,rg):
            expected=b+weight*r;assert actual.dtype==torch.float32 and torch.isfinite(actual).all()
            assert torch.allclose(actual,expected,atol=2e-7,rtol=2e-4)
            errors.append(float((actual-expected).abs().max()))
        contributions=[r*weight for r in rg];reference=[r*REFERENCE_WEIGHT for r in rg];factor=weight/REFERENCE_WEIGHT
        assert factor in (.5,2.)
        assert all(torch.equal(c,r*factor) for c,r in zip(contributions,reference))
        sn=summary(contributions);assert all(g['finite'] and g['effective_parameters']>0 for g in sn.values())
        records[branch]=dict(lambda_rec=weight,losses=loss,restored_weighted_reconstruction_gradients=sn,expected_ratio_to_v622=factor,
            actual_norm_ratios={k:sn[k]['norm']/summary(reference)[k]['norm'] for k in sn},
            merged_gradient_max_absolute_error=max(errors),merged_gradient_checked=True,forward_reconstructions_identical=True,
            base_losses_and_rates_identical=True,production_default_branch_dispatch_checked=True)
        dump(ROOT/'audits/weight_check_progress.json',dict(status='RUNNING',completed_branch=branch,records=records))
        del grad,contributions,reference
    x=torch.rand_like(frames[0],requires_grad=True);y=torch.rand_like(x)
    _,layers,_=dists_details(quality,x,y,True)
    assert all(z['output_gradient']['finite'] and z['output_gradient']['effective_parameters']>0 for z in layers)
    assert initial=={k:v4.module_hash(m) for k,m in models.items()} and core==v4.compression_hash(im,pm) and qh==[v4.module_hash(m) for m in quality]
    cp=torch.load(cfg['source_checkpoint']['path'],map_location='cpu',weights_only=True)
    for k,m in models.items():m.load_state_dict(cp[k],strict=True)
    del cp;clear();pm.clear_dpb();pm.set_curr_poc(0);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    assert initial=={k:v4.module_hash(m) for k,m in models.items()}
    result=dict(status='PASS',source_checkpoint_sha256=EXPECTED,initial_modules=initial,compression_hash=core,quality_hashes=qh,
        sample=row,plan_sha256=sha(ROOT/'plans/shared.json'),records=records,DISTS_layers=layers,reconstruction_gradients=norms,
        reconstruction_unit_gradient_scale=4096.,parameter_dtypes=sorted({str(p.dtype) for p in params}),
        restored_and_accumulated_dtype='torch.float32',base_gradients_not_rescaled=True,models_unchanged=True,
        optimizer_updates=0,formal_plan_consumed=0,reloaded_initial_parameters=True,RNG_reset_seed=seed,
        fixed_weights_no_calibration=True,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20)
    dump(ROOT/'audits/implementation_check.json',result);dump(ROOT/'audits/weight_check.json',result)
    dump(ROOT/'audits/executed_source_hashes.json',{str(p.resolve()):sha(p) for p in ROOT.glob('*.py')})
    print('WEIGHT CHECK PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'audits/weight_check.json',dict(status='FAIL',error=traceback.format_exc()));raise
