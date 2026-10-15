"""Frozen paper-compatible proxy; no claim of original training teacher identity."""
import gc
import types
from audit_io import *

METHOD_KEYS = {'Original':'original', **{k:'B1_initial' for k in ('B1_alpha0','B1_alpha025','B1_alpha050','B1_alpha075','B1_alpha100','B1_alpha125')}}


def backend():
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(REPO),str(EXPS/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies')]
    import torch
    from src.utils.common import set_torch_env
    torch.set_num_threads(2)
    set_torch_env()


def frames_for(v, count=64):
    import numpy as np
    import torch
    from PIL import Image
    class Frames:
        def __len__(self): return count
        def __getitem__(self,i):
            if i>=count: raise IndexError(i)
            p=Path(v['input_dir'])/(f'im{i+1:05d}.png' if v['dataset']=='hevc_b' else f'frame_{i:06d}.png')
            assert sha(p)==v['frame_png_sha256' if v['dataset']=='hevc_b' else 'frame_file_sha256'][i]
            with Image.open(p) as image:
                arr=np.asarray(image,dtype=np.uint8).copy()
            assert hashlib.sha256(arr.tobytes()).hexdigest()==v['frame_rgb_sha256'][i]
            return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()/255
    return Frames()


def teacher_targets(v, count=64):
    """Official Open-MAGVIT2 encoder definition/EMA weights; isolated writes only."""
    import torch
    import torch.nn.functional as F
    import yaml
    path=ROOT/'cache/targets'/v['dataset']/(v['name']+f'_{count}.pt')
    source=load(V13/'audits/pretrain_teacher_checkpoint_source.json')['checkpoint']
    signature=dict(teacher_sha256=source['sha256'],RGB=v['frame_rgb_sha256'][:count],
                   preprocessing='FP32 RGB -> replicate bottom 8 -> 2*x-1; full frame, no resize',EMA=True)
    historical=EXPS/'gvcrt_neural_wrapper_v6_31a_latent_fidelity_vs_generative_utility_audit/cache/targets'/v['dataset']/(v['name']+f'_{count}.pt')
    if not path.exists() and historical.exists():
        state=torch.load(historical,map_location='cpu',weights_only=True)
        assert state['signature']==signature
        return state['targets'],state['teacher_hash']
    if path.exists():
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['signature']==signature
        return state['targets'],state['teacher_hash']
    cfg=yaml.safe_load((V13/'teacher_sources/configs/Open-MAGVIT2/gpu/pretrain_lfqgan_256_262144.yaml').read_text())['model']['init_args']
    assert cfg['embed_dim']==cfg['ddconfig']['z_channels']==18 and cfg['n_embed']==2**18
    source_module=module('v631a_official_teacher_encoder',V13/'teacher_sources/src/Open_MAGVIT2/modules/diffusionmodules/improved_model.py')
    encoder=source_module.Encoder(**cfg['ddconfig'])
    assert sha(source['path'])==source['sha256']
    payload=torch.load(source['path'],map_location='cpu',weights_only=True)['state_dict']
    encoder.load_state_dict({n:payload['model_ema.'+('encoder.'+n).replace('.','')] for n in encoder.state_dict()},strict=True)
    del payload
    encoder=encoder.cuda().float().eval().requires_grad_(False)
    initial=tensor_hash(encoder.state_dict());targets=[]
    with torch.inference_mode():
        for i,x in enumerate(frames_for(v,count)):
            target=encoder(F.pad(x.cuda(),(0,0,0,8),mode='replicate')*2-1)
            assert target.shape==(1,18,68,120) and torch.isfinite(target).all()
            targets.append(target.cpu())
            print('TEACHER',v['dataset'],v['name'],i,flush=True)
    assert tensor_hash(encoder.state_dict())==initial
    del encoder,target
    gc.collect();torch.cuda.empty_cache()
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(dict(targets=targets,signature=signature,teacher_hash=initial),tmp);tmp.replace(path)
    return targets,initial


def latent_metrics(z,t):
    import torch
    import torch.nn.functional as F
    assert z.shape==t.shape==(1,18,68,120)
    a,b=z.double().flatten(2)[0],t.double().flatten(2)[0]
    d=a-b
    rmse=d.square().mean().sqrt()
    ac=a-a.mean(1,keepdim=True);bc=b-b.mean(1,keepdim=True)
    variance=bc.square().mean(1).sqrt()
    denom=ac.norm(dim=1)*bc.norm(dim=1)
    valid=denom>0
    stats=dict(L1=float(d.abs().mean()),RMSE=float(rmse),normalized_L2=float(d.norm()/b.norm()),
               cosine=float(1-F.cosine_similarity(z.double(),t.double(),dim=1,eps=1e-8).mean()),
               channel_NRMSE=float((d.square().mean(1).sqrt()[variance>0]/variance[variance>0]).mean()) if (variance>0).any() else None,
               channel_Pearson=float(((ac*bc).sum(1)[valid]/denom[valid]).mean()) if valid.any() else None,
               mean_difference=float((a.mean(1)-b.mean(1)).square().mean().sqrt()),
               covariance_difference=float(((ac@ac.T-bc@bc.T)/(a.shape[1]-1)).square().mean().sqrt()),
               zero_variance_target_channels=int((variance==0).sum()))
    assert all(v is None or torch.isfinite(torch.tensor(v)) for v in stats.values())
    return stats


def runtime(method,targets):
    import torch
    # Reuse immutable, previously audited native-I/causal-RANS adapter with writable root rebound.
    v30=EXPS/'gvcrt_neural_wrapper_v6_30_fid_cache_closure_sample_robustness_audit'
    oldio=module('v631a_historical_io',v30/'io21.py')
    cfg=load(V29/'evaluation/config.json')
    cfg['checkpoints']['B1_initial']['path']=str(B1)
    cfg.update(experiment='B',IP=-1)
    ns=dict(oldio.__dict__)
    ns.update(ROOT=ROOT,evalcfg=lambda:cfg,IPS=(-1,))
    source=(v30/'adapter.py').read_text().replace('from io21 import *','')
    exec(compile(source,str(ROOT/'proxy_runtime.py'),'exec'),ns)
    eng=ns['engine']()
    rt=eng.Runtime(cfg,METHOD_KEYS[method],torch.device('cuda:0'))
    # Replace only historical Inception with audited official pytorch-fid extractor.
    sys.path.insert(0,str(V29/'vendor/pytorch-fid/src'))
    import pytorch_fid.inception as inc
    p=Path('/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth')
    assert sha(p)=='6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'
    inc.load_state_dict_from_url=lambda *a,**kw:torch.load(p,map_location='cpu',weights_only=True)
    class Flat(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net=inc.InceptionV3([3],resize_input=True,normalize_input=True,requires_grad=False,use_fid_inception=True).cuda().eval().requires_grad_(False)
        def forward(self,x):return self.net(x.float())[0].flatten(1)
    rt.metric_models=(*rt.metric_models[:2],Flat().eval().requires_grad_(False))
    rt.latent_rows=[];rt.wrapper_rows=[];rt.full_samples={};rt.models_count=0
    old_models=rt.models
    def models(self,h,w):
        ia,pa=old_models(h,w);self.models_count+=1
        if self.models_count==3:
            def hook(_m,args):
                i=len(self.latent_rows);z=args[0].detach();t=targets[i].to(z.device)
                quant=torch.where(t>0,1.,-1.)
                row=dict(frame=i,proxy_teacher=True,primary_target='continuous_pre_LFQ',
                         tensor_shape=list(z.shape),dtype=str(z.dtype),z_min=float(z.min()),z_max=float(z.max()),
                         z_sha256=eng.eco.tensor_sha(z),target_sha256=eng.eco.tensor_sha(t),
                         target_min=float(t.min()),target_max=float(t.max()),
                         quant_step_sha256=eng.eco.tensor_sha(args[1]),
                         z_source='independent persisted-RANS decode -> bridge codeword -> decoder',
                         target_source='paper_compatible_teacher_proxy(original_GT_RGB)',
                         **latent_metrics(z,t),**{'quant_'+k:v for k,v in latent_metrics(z,quant).items()})
                self.latent_rows.append(row)
                if i in self.sample_frames:
                    self.full_samples[i]=dict(z_gen_actual=z.cpu(),z_target_cont=t.cpu(),
                         z_target_quant=quant.cpu(),z_target_normalized=torch.nn.functional.normalize(t,dim=1).cpu(),
                         normalized_target_is_derived_diagnostic=True,quant_step=args[1].detach().cpu())
            ia.recon_generation_net.decoder.register_forward_pre_hook(hook)
            pa.recon_generation_net.decoder.register_forward_pre_hook(hook)
        return ia,pa
    rt.models=types.MethodType(models,rt)
    rt.sample_frames=[]
    if rt.wrapper is not None:
        alpha={'B1_alpha0':0.,'B1_alpha025':.25,'B1_alpha050':.5,'B1_alpha075':.75,'B1_alpha100':1.,'B1_alpha125':1.25}[method]
        normal_forward=rt.wrapper.forward
        def strength(self,x):
            p=normal_forward(x)
            return p if alpha==1. else x if alpha==0. else (x+alpha*(p-x)).clamp(0,1)
        rt.wrapper.forward=types.MethodType(strength,rt.wrapper)
    if rt.wrapper is not None:
        def wrapper_hook(_m,args,output):
            x=args[0].detach().float();y=output.detach().float();r=y-x
            # Fixed low/mid/high radial frequency bands, normalized FFT energy.
            fy=torch.fft.fftfreq(x.shape[-2],device=x.device)[:,None]
            fx=torch.fft.rfftfreq(x.shape[-1],device=x.device)[None,:]
            radius=(fx.square()+fy.square()).sqrt()
            ex=torch.fft.rfft2(x,norm='ortho').abs().square().mean((0,1))
            ey=torch.fft.rfft2(y,norm='ortho').abs().square().mean((0,1))
            row=dict(frame=len(self_rows)+1,RGB_residual_L1=float(r.abs().mean()),RGB_residual_RMSE=float(r.square().mean().sqrt()),RGB_residual_max_abs=float(r.abs().max()))
            for name,mask in [('low',radius<.125),('mid',(radius>=.125)&(radius<.25)),('high',radius>=.25)]:
                row['frequency_'+name+'_energy_delta']=float((ey[mask]-ex[mask]).mean())
            self_rows.append(row)
        self_rows=rt.wrapper_rows
        rt.wrapper.register_forward_hook(wrapper_hook)
    return eng,rt
