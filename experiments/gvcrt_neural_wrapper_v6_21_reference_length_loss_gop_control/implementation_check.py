"""No P/B/G optimizer updates; objective equivalence and gradient isolation."""
import argparse,traceback
from io21 import *
from window_training import *
from gradient_tools import stats,dists_details
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args()
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    from patchgan import create,update
    torch.set_num_threads(2);torch.manual_seed(20261011);frozen()
    v4,im,pm,models,quality=load_training(torch.device('cuda:0'));params=[p for m in models.values() for p in m.parameters()]
    initial={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm);qh=[v4.module_hash(m) for m in quality]
    r=replay_rows('A')[0];frames=replay(r,'A',torch.device('cuda:0'));q=r['external_qp'];records=[]
    def clear():
        for p in params:p.grad=None
    def grads():return [None if p.grad is None else p.grad.detach().cpu().clone() for p in params]
    def match(g):
        for p,h in zip(params,g):
            assert (p.grad is None)==(h is None)
            if h is not None:assert torch.allclose(p.grad.cpu(),h,atol=2e-6,rtol=2e-4)
    oldns=dict(globals(),BRANCHES=('wrapped_i_control','native_i_adapt'))
    exec((V20/'window_training.py').read_text().replace('from io20 import *',''),oldns)
    old=oldns['make_runner'](v4,im,pm,models,quality);clear();ol,oq,ot=old(frames[48:],q,'native_i_adapt',True);og=grads()
    run=make_runner(v4,im,pm,models,quality)
    clear();al,aq,at,aw,_=run(frames,q,'A');match(og)
    assert all(abs(al[k]-ol[k])<=2e-6+1e-5*abs(ol[k]) for k in al if k in ol)
    assert aq==oq[1:]==r['actual_qps'] and not aw
    ag=grads()
    disc,opt,drng=create(torch.device('cuda:0'));withD=make_runner(v4,im,pm,models,quality,disc)
    clear();cl,cq,ct,cw,cf=withD(frames,q,'C',adv_weight=0);match(ag)
    assert cq==aq and not cw and all(x['reconstruction_sha256']==y['reconstruction_sha256'] for x,y in zip(at,ct))
    assert all(abs(cl[k]-al[k])<=2e-6+1e-5*abs(al[k]) for k in al if k!='L_adv')
    clear();adv,_,_,_,fake=withD(frames,q,'C',adv_only=True)
    gs={k:stats([p.grad for p in m.parameters()],list(m.parameters())) for k,m in models.items()}
    dump(ROOT/'audits/adversarial_gradient_details.json',dict(gradients=gs,loss=adv,sample=r,D_hash=v4.module_hash(disc),D_parameters_frozen=all(not p.requires_grad for p in disc.parameters())))
    assert all(g['finite'] and g['effective_parameters']>0 for g in gs.values()),gs
    assert all(p.grad is None for p in disc.parameters())
    high=grads();clear()
    lowloss,_,_,_,_=withD(frames,q,'C',adv_only=True,adv_scale=256.)
    low=grads();comparisons={};offset=0
    for k,m in models.items():
        n=len(list(m.parameters()));aa=high[offset:offset+n];bb=low[offset:offset+n];offset+=n
        error=sum(float((x.double()-y.double()).square().sum()) for x,y in zip(aa,bb) if x is not None)**.5
        norm=sum(float(x.double().square().sum()) for x in aa if x is not None)**.5
        cosine=sum(float((x.double()*y.double()).sum()) for x,y in zip(aa,bb) if x is not None)/(norm*sum(float(y.double().square().sum()) for y in bb if y is not None)**.5)
        comparisons[k]=dict(relative_L2=error/norm,cosine=cosine)
        assert error/norm<.25 and cosine>.97,comparisons[k]
    assert all(abs(lowloss[k]-adv[k])<1e-7 for k in adv)
    clear();merged,_,mt,_,_=withD(frames,q,'C')
    for p,base,added in zip(params,ag,high):
        expected=base+added
        assert torch.allclose(p.grad.cpu(),expected,rtol=2e-4,atol=2e-7)
    assert all(x['reconstruction_sha256']==y['reconstruction_sha256'] for x,y in zip(at,mt))
    dump(ROOT/'audits/adversarial_backward_scaling.json',dict(status='PASS',scale=4096.,compared_scales=[256.,4096.],parameter_dtypes=sorted(set(str(p.dtype) for p in params)),gradient_dtypes=sorted(set(str(p.grad.dtype) for p in params)),restored_adversarial_gradients=gs,comparisons=comparisons,formula='g_base + grad(S * 0.01 * L_adv / 15) / S',FP32_unscale=True,separate_gradients=True,forward_reconstruction_unchanged=True,unscaled_loss_unchanged=True,C_zero_adv_matches_A=True,merged_gradients_verified=True,formal_plan_updates_consumed=0,formal_optimizer_updates=0))
    pg=grads();beforeD=v4.module_hash(disc);dlog=update(disc,opt,frames[49:],fake);match(pg)
    assert beforeD!=v4.module_hash(disc) and initial=={k:v4.module_hash(m) for k,m in models.items()}
    clear();bl,bq,bt,bw,_=run(frames,q,'B')
    assert bq==aq and len(bw)==48 and all(x['no_grad'] and x['reference_detached'] for x in bw)
    assert [x['position'] for x in bw]==list(range(1,49))
    assert [x['reference_age'] for x in at]==list(range(1,16)) and [x['reference_age'] for x in bt]==list(range(49,64))
    x=torch.rand_like(frames[0],requires_grad=True);y=torch.rand_like(x);_,layers,_=dists_details(quality,x,y,True)
    assert all(t['output_gradient']['finite'] and t['output_gradient']['effective_parameters']>0 for t in layers)
    assert initial=={k:v4.module_hash(m) for k,m in models.items()} and core==v4.compression_hash(im,pm) and qh==[v4.module_hash(m) for m in quality]
    dump(ROOT/'audits/implementation_check.json',dict(status='PASS',PBG_optimizer_updates=0,probe_only_D_updates=1,initial_modules=initial,compression_hash=core,quality_hashes=qh,source_native16_loss_gradient_match=True,C_zero_adv_matches_A=True,C_adv_gradients=gs,GD_gradient_isolation=True,D_probe=dlog,B_no_grad_warmup=bw,target_QPs=aq,targets=list(range(49,64)),loss_denominator=15,A=al,B=bl,C_zero=cl,DISTS_layers=layers,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20))
    dump(ROOT/'audits/executed_source_hashes.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.glob('*.py')},repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()))
    print('IMPLEMENTATION PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'audits/implementation_check.json',dict(status='FAIL',error=traceback.format_exc()));raise
