"""Same reconstruction graph, two DISTS weights, no optimizer update."""
import argparse,math,traceback
from io17 import *
from gradient_tools import gradients,stats,dists_details
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2);frozen();command([PYTHON,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    v4,im,pm,models,quality=load_training(torch.device('cuda:0'));before={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm);qh=[v4.module_hash(m) for m in quality]
    row=replay_rows()[0];frames,plan=replay(row,torch.device('cuda:0'));q=row['external_qp'];common=module('weight17_common',V62/'fullqp_common.py')
    pm.clear_dpb();pm.set_curr_poc(0);actual=[common.frame_qp(pm,q,i) for i in range(4)];assert actual==row['actual_qps']
    with torch.no_grad():pm.add_ref_frame(None,im.compress(v4.codec_input(models['wrapper'](frames[0])),actual[0])['x_hat'])
    capture=[];hook=pm.dec.register_forward_hook(lambda m,args,out:capture.append(out));records=[];flat=[]
    try:
        for i,target in enumerate(frames[1:],1):
            proxy=models['wrapper'](target);rec,rate,_=v4.joint_ste_forward(pm,proxy,actual[i]);beta=common.beta_q('schedule_s1p0',q)
            lp=quality[0](rec,target,normalize=True).mean();ds,layers,_=dists_details(quality,rec,target,True);rb=rate/(256*256);pl=F.l1_loss(proxy,target)
            losses={w:lp+w*ds+beta*rb+.01*pl for w in (1.0,.5)}
            torch.testing.assert_close(losses[1.0]-losses[.5],.5*ds,rtol=1e-5,atol=2e-6)
            assert all(x['connected_to_output'] and x['output_gradient']['finite'] and x['output_gradient']['norm']>0 for x in layers[1:])
            scaled={}
            for name,m in models.items():
                ps=list(m.parameters());g1=gradients(ds,ps);g05=gradients(.5*ds,ps)
                assert stats(g1,ps)['finite'] and stats(g1,ps)['effective_parameters']>0
                assert all((x is None)==(y is None) for x,y in zip(g1,g05))
                diff=math.sqrt(sum(float((y.detach().double()-.5*x.detach().double()).square().sum()) for x,y in zip(g1,g05) if x is not None))
                norm=stats(g1,ps)['norm'];assert diff<=1e-7+.003*norm,('DISTS gradient scaling mismatch',name,diff,norm)
                scaled[name]=dict(weight1=stats(g1,ps),weight05=stats(g05,ps),half_gradient_residual_norm=diff,tolerance=1e-7+.003*norm)
                flat.append(dict(frame=i,module=name,weight1_norm=norm,weight05_norm=stats(g05,ps)['norm'],half_gradient_residual_norm=diff,status='PASS'))
            with torch.no_grad():
                values={str(w):dict(LPIPS=float(quality[0](rec,target,normalize=True).mean()),DISTS=float(quality[1](rec,target,require_grad=True).mean()),rate_bpp=float(rb),beta_rate=float(beta*rb),proxy_L1=float(pl),loss=float(losses[w])) for w in (1.,.5)}
            for key in ('LPIPS','DISTS','rate_bpp','beta_rate','proxy_L1'):assert values['1.0'][key]==values['0.5'][key]
            records.append(dict(frame=i,reconstruction_sha256=tensor_hash({'rec':rec}),same_reconstruction_graph=True,values=values,loss_difference=float((losses[1.]-losses[.5]).detach()),expected_difference=float((.5*ds).detach()),deep_features=layers,PBG_gradient_scaling=scaled))
            with torch.no_grad():pm.add_ref_frame(capture[-1].detach(),(rec.detach()*2-1).half())
            capture.clear()
    finally:hook.remove()
    assert before=={k:v4.module_hash(m) for k,m in models.items()} and core==v4.compression_hash(im,pm) and qh==[v4.module_hash(m) for m in quality]
    # Exercise the exact training objective for both weights, at identical model/input.
    totals={};grads={};recon_hashes={}
    for weight in (1.,.5):
        for m in models.values():m.zero_grad(set_to_none=True)
        run=objective(v4,im,pm,models,quality,True,weight);wrapper=run.__globals__['v4'];native=wrapper.joint_ste_forward;captured=[]
        def capture_forward(*args,**kwargs):
            result=native(*args,**kwargs);captured.append(tensor_hash({'rec':result[0]}));return result
        wrapper.joint_ste_forward=capture_forward
        totals[str(weight)],qp=run(frames,q);assert qp==actual
        recon_hashes[str(weight)]=captured
        grads[str(weight)]={k:stats([p.grad for p in m.parameters()],list(m.parameters())) for k,m in models.items()}
    assert recon_hashes['1.0']==recon_hashes['0.5'] and len(recon_hashes['1.0'])==3
    for key in ('LPIPS','DISTS','rate_bpp','beta_rate','proxy_L1'):assert math.isclose(totals['1.0'][key],totals['0.5'][key],rel_tol=1e-6,abs_tol=2e-6)
    assert math.isclose(totals['1.0']['loss']-totals['0.5']['loss'],.5*totals['1.0']['DISTS'],rel_tol=2e-5,abs_tol=3e-6)
    assert before=={k:v4.module_hash(m) for k,m in models.items()} and core==v4.compression_hash(im,pm) and qh==[v4.module_hash(m) for m in quality]
    write(ROOT/'audits/weight_gradient_scaling.csv',flat)
    dump(ROOT/'audits/weight_check.json',dict(status='PASS',optimizer_updates=0,sample=plan,external_qp=q,actual_qps=actual,frames=records,exact_training_objective_totals=totals,exact_training_reconstruction_hashes=recon_hashes,total_PBG_gradients=grads,model_hashes_before=before,model_hashes_after=before,quality_hashes_before=qh,quality_hashes_after=qh,compression_hash=core,inherited_gradient_audit_sha256=sha(V16/'gradient_audit.json')))
    print('WEIGHT CHECK PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'audits/weight_check.json',dict(status='FAIL',optimizer_updates=0,error=traceback.format_exc()));raise
