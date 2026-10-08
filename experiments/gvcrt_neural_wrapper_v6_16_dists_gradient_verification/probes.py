"""Component-specific gradients; finite differences only on independent RGB."""
import argparse,math,types,traceback
from io16 import *
def gradients(loss,params):
    import torch
    if not loss.requires_grad:return [None for _ in params]
    return torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
def stats(gs,params):
    import torch
    defined=[g for g in gs if g is not None];s=sum(float(g.detach().double().square().sum()) for g in defined)
    return dict(norm=math.sqrt(s),finite=all(bool(torch.isfinite(g).all()) for g in defined),total_parameters=len(params),connected_parameters=len(defined),effective_parameters=sum(bool((g!=0).any()) for g in defined),effective_elements=sum(int((g!=0).sum()) for g in defined))
def dists_details(quality,x,y,fixed):
    import torch
    captures=[];ds=quality[1];original=ds.forward_once
    def traced(z):
        fs=original(z);captures.append(fs);return fs
    ds.forward_once=traced
    try:score=ds(x,y,require_grad=True).mean() if fixed else quality[1](x,y).mean()
    finally:ds.forward_once=original
    assert len(captures)==2
    terms=[];ws=ds.alpha.sum()+ds.beta.sum();alpha=torch.split(ds.alpha/ws,ds.chns,dim=1);beta=torch.split(ds.beta/ws,ds.chns,dim=1)
    for k,(a,b) in enumerate(zip(*captures)):
        am=a.mean([2,3],keepdim=True);bm=b.mean([2,3],keepdim=True)
        s1=(2*am*bm+1e-6)/(am.square()+bm.square()+1e-6)
        av=(a-am).square().mean([2,3],keepdim=True);bv=(b-bm).square().mean([2,3],keepdim=True);cov=(a*b).mean([2,3],keepdim=True)-am*bm
        s2=(2*cov+1e-6)/(av+bv+1e-6)
        terms.append(-((alpha[k]*s1).sum(1,keepdim=True)+(beta[k]*s2).sum(1,keepdim=True)).mean())
    assert torch.allclose(1+sum(terms),score,rtol=1e-5,atol=2e-6)
    layers=[]
    for k,(feature,term) in enumerate(zip(captures[0],terms)):
        g=gradients(term,[x]);layers.append(dict(layer=k,kind='RGB' if k==0 else 'deep',shape=list(feature.shape),requires_grad=feature.requires_grad,grad_fn=None if feature.grad_fn is None else type(feature.grad_fn).__name__,connected_to_output=g[0] is not None,output_gradient=stats(g,[x])))
    return score,layers,terms
def independent():
    import torch
    torch.set_num_threads(2);command([PYTHON,'-B','-u',str(Path(__file__).resolve()),'--cpu']);frozen()
    v4=module('dists16_independent_v4',V4/'train.py');quality=v4.quality_models(torch.device('cpu'));before=[v4.module_hash(m) for m in quality]
    callables=dict(perceptual=v4.perceptual,quality_models=v4.quality_models,DISTS_forward=type(quality[1]).forward,DISTS_forward_once=type(quality[1]).forward_once)
    runtime={name:dict(file=inspect.getfile(fn),sha256=sha(inspect.getfile(fn)),signature=str(inspect.signature(fn)),source=inspect.getsource(fn)) for name,fn in callables.items()}
    dump(ROOT/'audits/loaded_quality_runtime.json',dict(status='PASS',callables=runtime,quality_classes=[type(m).__module__+'.'+type(m).__name__ for m in quality],frozen_parameters=all(not p.requires_grad for m in quality for p in m.parameters()),eval_state=all(not m.training for m in quality)))
    gen=torch.Generator().manual_seed(20261008);x=(.15+.7*torch.rand((1,3,128,128),generator=gen)).requires_grad_();y=.15+.7*torch.rand((1,3,128,128),generator=gen)
    vals={};records=[];fd=[];grads={}
    for mode,fixed in [('A',False),('B',True)]:
        score,layers,terms=dists_details(quality,x,y,fixed);g=gradients(score,[x])[0];vals[mode]=float(score.detach());grads[mode]=g
        lp=quality[0](x,y,normalize=True).mean();records.append(dict(mode=mode,DISTS=vals[mode],DISTS_scalar_requires_grad=score.requires_grad,image_gradient=stats([g],[x]),LPIPS_image_gradient=stats(gradients(lp,[x]),[x]),layers=layers))
    torch.testing.assert_close(torch.tensor(vals['A']),torch.tensor(vals['B']),rtol=1e-6,atol=2e-6)
    directions=[torch.randn(x.shape,generator=gen),grads['B'].detach().clone()]
    for i,d in enumerate(directions):
        d=d/d.norm()
        for mode,fixed in [('A',False),('B',True)]:
            auto=float((grads[mode]*d).sum())
            for eps in (.01,.003,.001):
                with torch.no_grad():plus=quality[1](x+eps*d,y,require_grad=fixed).mean();minus=quality[1](x-eps*d,y,require_grad=fixed).mean()
                central=float((plus-minus)/(2*eps));error=abs(auto-central);match=error<=3e-5+.2*abs(auto)
                fd.append(dict(mode=mode,direction=i,direction_kind='seeded_random' if i==0 else 'B_gradient',direction_sha256=tensor_hash({'direction':d}),epsilon=eps,autograd=auto,finite_difference=central,absolute_error=error,tolerance=3e-5+.2*abs(auto),within_tolerance=match,input='independent RGB tensor; no codec or STE'))
    assert all(any(r['within_tolerance'] for r in fd if r['mode']=='B' and r['direction']==i) for i in range(2))
    assert before==[v4.module_hash(m) for m in quality]
    result=dict(status='PASS',values=vals,forward_difference=vals['B']-vals['A'],sampling=dict(seed=20261008,shape=list(x.shape),dtype=str(x.dtype),device=str(x.device),range=[.15,.85],rule='.15+.7*torch.rand; x then y; independent of all video pools'),input_sha256=tensor_hash({'x':x,'y':y}),x_sha256=tensor_hash({'x':x}),target_sha256=tensor_hash({'y':y}),records=records,finite_difference=fd,quality_hash_before=before,quality_hash_after=before,optimizer_step=False,probe_script_sha256=sha(Path(__file__)),runtime_audit_sha256=sha(ROOT/'audits/loaded_quality_runtime.json'))
    dump(ROOT/'probes/independent.json',result);write(ROOT/'probes/independent_finite_difference.csv',fd);print('INDEPENDENT PROBE PASS',flush=True)
def chain(gpu):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2);frozen();command([PYTHON,'-B','-u',str(Path(__file__).resolve()),'--gpu',str(gpu)])
    v4,im,pm,models,quality=load_training(torch.device('cuda:0'))
    before={k:v4.module_hash(m) for k,m in models.items()};qh=[v4.module_hash(m) for m in quality];core=v4.compression_hash(im,pm)
    common=module('dists16_probe_common',V62/'fullqp_common.py');records=[];plans=[]
    for domain in ('ulong','uvg'):
        row=next(r for r in replay_rows() if r['domain']==domain);frames,plan=replay(row,torch.device('cuda:0'));plans.append(plan)
        for q in (0,4,9):
            pm.clear_dpb();pm.set_curr_poc(0);actual=[common.frame_qp(pm,q,i) for i in range(4)]
            assert actual==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][:4]
            with torch.no_grad():pm.add_ref_frame(None,im.compress(v4.codec_input(models['wrapper'](frames[0])),actual[0])['x_hat'])
            capt=[];hook=pm.dec.register_forward_hook(lambda m,a,o:capt.append(o))
            try:
                for i,frame in enumerate(frames[1:],1):
                    proxy=models['wrapper'](frame);rec,rate,_=v4.joint_ste_forward(pm,proxy,actual[i]);vals={};item={}
                    for mode,fixed in [('A',False),('B',True)]:
                        ds,layers,terms=dists_details(quality,rec,frame,fixed);vals[mode]=float(ds.detach());pbg={}
                        for k,m in models.items():ps=list(m.parameters());pbg[k]=stats(gradients(ds,ps),ps)
                        item[mode]=dict(DISTS=vals[mode],layers=layers,reconstruction_gradient=stats(gradients(ds,[rec]),[rec]),PBG_gradients=pbg)
                    assert abs(vals['A']-vals['B'])<=2e-6+1e-5*abs(vals['A'])
                    lp=quality[0](rec,frame,normalize=True).mean();rb=rate/(frame.shape[-2]*frame.shape[-1]);pl=F.l1_loss(proxy,frame);other={}
                    for key,loss in [('LPIPS',lp),('rate_bpp',rb),('proxy_L1',pl)]:
                        other[key]={}
                        for k,m in models.items():ps=list(m.parameters());other[key][k]=stats(gradients(loss,ps),ps)
                    records.append(dict(domain=domain,plan=plan,external_QP=q,actual_QPs=actual,frame=i,values=vals,forward_difference=vals['B']-vals['A'],DISTS=item,other_components=other))
                    with torch.no_grad():pm.add_ref_frame(capt[-1].detach(),(rec.detach()*2-1).half())
                    capt.clear();del rec,proxy,lp,rb,pl,ds,terms
            finally:hook.remove()
            assert before=={k:v4.module_hash(m) for k,m in models.items()} and qh==[v4.module_hash(m) for m in quality] and core==v4.compression_hash(im,pm)
            dump(ROOT/'probes'/f'chain_{domain}_qp{q}.json',dict(status='PASS',records=[r for r in records if r['domain']==domain and r['external_QP']==q],optimizer_step=False,model_parameters_unchanged=True,quality_parameters_unchanged=True));print('CHAIN PROBE PASS',domain,q,flush=True)
    historical_complete=all(all(x['requires_grad'] and x['connected_to_output'] for x in r['DISTS']['A']['layers'][1:]) for r in records)
    fixed_complete=all(all(x['requires_grad'] and x['connected_to_output'] and x['output_gradient']['finite'] and x['output_gradient']['norm']>0 for x in r['DISTS']['B']['layers'][1:]) and all(g['finite'] and g['effective_parameters']>0 for g in r['DISTS']['B']['PBG_gradients'].values()) for r in records)
    gate='NO_FIX_REQUIRED' if historical_complete else 'FIX_REQUIRED' if fixed_complete else 'FAIL'
    flat=[]
    for r in records:
        for mode in ('A','B'):
            for module_name,g in r['DISTS'][mode]['PBG_gradients'].items():flat.append(dict(domain=r['domain'],external_QP=r['external_QP'],frame=r['frame'],mode=mode,component='DISTS',module=module_name,**g))
        for c,modules in r['other_components'].items():
            for m,g in modules.items():flat.append(dict(domain=r['domain'],external_QP=r['external_QP'],frame=r['frame'],mode='same_A_B',component=c,module=m,**g))
    write(ROOT/'probes/component_gradients.csv',flat)
    independent_result=load(ROOT/'probes/independent.json');assert independent_result['status']=='PASS'
    dump(ROOT/'gradient_audit.json',dict(status='PASS' if gate!='FAIL' else 'FAIL',gate=gate,trigger_repair=gate=='FIX_REQUIRED',historical_deep_gradients_complete=historical_complete,explicit_true_restores_all_deep_gradients=fixed_complete,plans=plans,chain_cases=6,reconstructed_frames=18,optimizer_steps=0,model_hashes_before=before,model_hashes_after=before,quality_hashes_before=qh,quality_hashes_after=qh,compression_hash=core,source_checkpoint_sha256=EXPECTED))
    print('GRADIENT AUDIT',gate,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu',action='store_true');p.add_argument('--gpu',type=int,choices=(4,5,6,7));a=p.parse_args()
    try:independent() if a.cpu else chain(a.gpu)
    except Exception:dump(ROOT/'probes/failure.json',dict(status='FAIL',error=traceback.format_exc()));raise
