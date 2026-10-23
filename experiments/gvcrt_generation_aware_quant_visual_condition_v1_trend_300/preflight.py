"""Required identity, independent coding and gradient checks before long runs."""
import json, os, sys, time
from pathlib import Path
P = Path(__file__).resolve().parent
sys.path.insert(0,str(P))
os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','4')
import train
train.setup_path()
import torch
from core import codec_input


def near(a,b,tol=1e-5):
    value=float((a.detach().float()-b.detach().float()).abs().max())
    assert value <= tol, (value,tol)
    return value


def fresh(branch):
    im,pm=train.load_models(branch)
    pm.eval()
    return im,pm


def seed_reference(im,pm,frame):
    pm.clear_dpb();pm.set_curr_poc(0)
    with torch.no_grad():pm.add_ref_frame(None,im.compress(codec_input(frame),2)['x_hat'])


def check():
    assert torch.cuda.device_count()==1
    result={'status':'RUNNING','physical_gpu':int(os.environ['CUDA_VISIBLE_DEVICES']),
            'initialization_sha256':train.sha(train.INIT),
            'numeric_tolerances':{'identity_reconstruction_feature_qmap':1e-5,
                                  'independent_decode_main_latent':1e-2,
                                  'independent_decode_qmap':1e-4,
                                  'estimated_rate_identity_abs':1e-3},'checks':{}}
    row=train.frame_plan()[0]
    frames=train.load_frames(row,torch.device('cuda:0'))[:3]
    model={}
    with torch.no_grad():
        for branch in 'ABCD':
            im,pm=fresh(branch);seed_reference(im,pm,frames[0])
            rec,feature,rate,trace=train.causal_forward(pm,frames[1],2)
            model[branch]=(im,pm,rec.detach(),feature.detach(),rate.detach(),trace)
        for left,right in [('A','B'),('A','C'),('B','D')]:
            a,b=model[left],model[right]
            result['checks'][f'identity_{left}_{right}']={
                'reconstruction_maxabs':near(a[2],b[2]),
                'feature_maxabs':near(a[3],b[3]),
                'estimated_rate_abs':float((a[4]-b[4]).abs())}
            assert result['checks'][f'identity_{left}_{right}']['estimated_rate_abs']<1e-3
        streams={}
        for branch in 'ABCD':
            im,pm,_,_,_,_=model[branch]
            seed_reference(im,pm,frames[0])
            coded=[]
            for pos in (1,2):
                state={}
                hook=pm.dec.register_forward_pre_hook(lambda m,args:state.update(y=args[0].detach().clone()))
                try:stream=pm.compress(codec_input(frames[pos]),2)['bit_stream']
                finally:hook.remove()
                coded.append((stream,state['y'],pm.q_actual.detach().clone()))
            streams[branch]=coded
        for left,right in [('A','B'),('A','C'),('B','D')]:
            assert [x[0] for x in streams[left]]==[x[0] for x in streams[right]]
            for a,b in zip(streams[left],streams[right]):near(a[1],b[1]);near(a[2],b[2])
            result['checks'][f'bitstream_{left}_{right}']='identical_for_two_causal_P_frames'
        # Force a known nonidentity map to check decoder synchronization and real stream effect.
        for branch in 'BD':
            pm=model[branch][1]
            pm.quant_adjust.net[-1].bias.fill_(.25)
            seed_reference(model[branch][0],pm,frames[0])
            coded=[]
            for pos in (1,2):
                state={}
                hook=pm.dec.register_forward_pre_hook(lambda m,args:state.update(y=args[0].detach().clone()))
                try:stream=pm.compress(codec_input(frames[pos]),2)['bit_stream']
                finally:hook.remove()
                coded.append((stream,state['y'],pm.q_actual.detach().clone()))
            streams[branch+'_nonidentity']=coded
        assert streams['B_nonidentity'][0][0]!=streams['A'][0][0]
        assert [x[0] for x in streams['B_nonidentity']]==[x[0] for x in streams['D_nonidentity']]
        result['checks']['nonidentity_stream']='changed_vs_A_and_identical_B_D'
        for branch in 'BD':
            im,dec=fresh(branch)
            dec.quant_adjust.net[-1].bias.fill_(.25)
            seed_reference(im,dec,frames[0])
            for pos,coded in enumerate(streams[branch+'_nonidentity'],1):
                state={}
                hook=dec.dec.register_forward_pre_hook(lambda m,args:state.update(y=args[0].detach().clone()))
                try:dec.decompress(coded[0],{'height':256,'width':256,'ec_part':1},2)
                finally:hook.remove()
                near(state['y'],coded[1],tol=1e-2)
                near(dec.q_actual,coded[2],tol=1e-4)
            result['checks'][f'independent_decode_{branch}']='two_frames_y_and_qmap_synchronized'
    for branch in 'BCD':
        im,pm=fresh(branch)
        seed_reference(im,pm,frames[0])
        pm.recon_generation_net.decoder.float().requires_grad_(True)
        if branch in 'BD':pm.quant_adjust.float().requires_grad_(True)
        if branch in 'CD':pm.visual_condition.float().requires_grad_(True)
        rec,feature,rate,_=train.causal_forward(pm,frames[1],2)
        if branch in 'BD':
            qparams=list(pm.quant_adjust.parameters())
            recon_grad=torch.autograd.grad(rec.float().square().mean(),qparams,retain_graph=True,allow_unused=True)
            rate_grad=torch.autograd.grad(rate,qparams,retain_graph=True,allow_unused=True)
            rn=sum(float(g.float().norm()) for g in recon_grad if g is not None)
            bn=sum(float(g.float().norm()) for g in rate_grad if g is not None)
            assert rn>0 and bn>0 and all(g is None or torch.isfinite(g).all() for g in recon_grad+rate_grad)
            result['checks'][f'quant_gradient_{branch}']={'reconstruction':rn,'rate':bn}
        if branch in 'CD':
            params=list(pm.visual_condition.parameters())
            grad=torch.autograd.grad(rec.float().square().mean(),params,retain_graph=True,allow_unused=True)
            gn=sum(float(g.float().norm()) for g in grad if g is not None)
            assert gn>0
            before=[p.detach().clone() for p in params]
            opt=torch.optim.AdamW(params,lr=1e-4)
            opt.zero_grad(set_to_none=True)
            rec.float().square().mean().backward()
            opt.step()
            delta=sum(float((p.detach()-old).float().norm()) for p,old in zip(params,before))
            assert delta>0
            seed_reference(im,pm,frames[0])
            with torch.no_grad():after,_,_,_=train.causal_forward(pm,frames[1],2)
            response=float((after-rec.detach()).abs().max())
            assert response>0
            rec2,_,_,_=train.causal_forward(pm,frames[1],2)
            upstream=torch.autograd.grad(rec2.float().square().mean(),
                                          list(pm.visual_condition.trunk.parameters()),allow_unused=True)
            upstream_norm=sum(float(g.float().norm()) for g in upstream if g is not None)
            assert upstream_norm>0
            result['checks'][f'condition_gradient_{branch}']={'gradient_norm':gn,'parameter_delta':delta,
                                                                 'response_maxabs':response,'upstream_gradient_after_update':upstream_norm}
    result['status']='PASS';result['time_unix']=time.time()
    (P/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':check()
