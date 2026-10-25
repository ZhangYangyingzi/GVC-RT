"""Fixed-frame receiver-side multiplier and actual-rho views."""
import argparse
import io
import json
from pathlib import Path

import task

P=Path(__file__).resolve().parent


def digest(tensor):
    import hashlib
    return None if tensor is None else hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def run(source,qp,mode,alpha_index,gpu):
    ctx=task.initialize(gpu)
    train,torch,_,_,_,unit,_,INDEX_MAP,stream=ctx
    key=task.key(mode,source,qp,alpha_index)
    record=json.loads((P/'server_artifacts/evaluation'/f'{key}.json').read_text())
    assert record['status']=='PASS'
    path=Path(record['stream_path'])
    assert task.sha(path)==record['stream_sha256']
    decoded_mode,decoded_alpha,native=task.read_header(path.read_bytes())
    assert (decoded_mode,decoded_alpha)==(mode,alpha_index)
    im,pm,checkpoint_sha=task.load_models(ctx,decoded_mode,decoded_alpha)
    assert checkpoint_sha==record['checkpoint_sha256']
    source_record=task.source_record(source)
    h,w=source_record['height'],source_record['width']
    sps={'sps_id':0,'height':(h+63)//64*64,'width':(w+63)//64*64,
         'ec_part':int(((h+63)//64*64)*((w+63)//64*64)>1280*720),'use_ada_i':0}
    im.set_use_two_entropy_coders(bool(sps['ec_part']))
    pm.set_use_two_entropy_coders(bool(sps['ec_part']))
    pm.clear_dpb();pm.set_curr_poc(0)
    buffer=io.BytesIO(native);helper=stream.SPSHelper()
    maps={};previews={};states=[];diagnostics=[]
    import time
    started=time.monotonic()
    pass_scales=[]
    original_indexes=pm.gaussian_encoder.build_indexes_decoder
    def capture_scales(scales):
        pass_scales.append(float((scales.float()<=.12).float().mean()))
        return original_indexes(scales)
    pm.gaussian_encoder.build_indexes_decoder=capture_scales
    with torch.inference_mode():
        for i in range(64):
            pass_scales.clear()
            head=stream.read_header(buffer)
            while head['nal_type']==stream.NalType.NAL_SPS:
                helper.add_sps_by_id(stream.read_sps_remaining(buffer,head['sps_id']))
                head=stream.read_header(buffer)
            decoded_sps=helper.get_sps_by_id(head['sps_id'])
            assert decoded_sps==sps
            q,bits=stream.read_ip_remaining(buffer)
            assert q==(qp if i==0 else pm.shift_qp(qp,INDEX_MAP[i%8]))
            if i==0:
                decoded=im.decompress(bits,decoded_sps,q)
                pm.clear_dpb();pm.set_curr_poc(0);pm.add_ref_frame(None,decoded['x_hat'])
                states.append({'frame':i,'reference':digest(decoded['x_hat']),
                               'latent':None,'qmap':None,'rgb':digest(decoded['x_hat'])})
            else:
                decoded=pm.decompress(bits,decoded_sps,q)
                states.append({'frame':i,'reference':digest(pm.dpb[0].feature),
                               'latent':digest(pm.decoding_trace['y_hat']),
                               'qmap':digest(pm.q_actual),'rgb':digest(decoded['x_hat'])})
                actual_rho=pm.q_actual/pm.prior_params[:,:128].clamp_min(.5)
                assert torch.equal(actual_rho,pm.decoding_trace['rho'])
                rho=actual_rho.detach().float().mean(1,keepdim=True)
                raw=pm.last_m_raw if mode!='A' else torch.ones_like(rho)
                mult=pm.last_m_effective if mode!='A' else torch.ones_like(rho)
                mean_error=0.0
                if mode=='ROLL':
                    assert torch.equal(torch.sort(raw.flatten()).values,
                                       torch.sort(pm.last_m_variant.flatten()).values)
                if mode=='MEAN':
                    mean_error=float((pm.last_m_variant-raw.mean((-2,-1),keepdim=True)).abs().max())
                    assert mean_error<=1e-7
                assert len(pass_scales)==2
                diagnostic={'frame':i,'q_clamp_fraction':pm.last_q_clamp_fraction or 0.0,
                    'pass0_skip_fraction':pass_scales[0],'pass1_skip_fraction':pass_scales[1],
                    'mean_definition_max_abs_error':mean_error,'rho_post_dtype_max_abs_error':0.0,
                    'roll_multiset_exact':True if mode=='ROLL' else '',
                    'roll_height_pixels':4*(raw.shape[-2]//4//2) if mode=='ROLL' else 0,
                    'roll_width_pixels':4*(raw.shape[-1]//4//2) if mode=='ROLL' else 0}
                for label,tensor in [('raw_m',raw),('m',mult),('rho',actual_rho),('actual_q',pm.q_actual)]:
                    diagnostic.update({f'{label}_{k}':v for k,v in task.stats(tensor).items()})
                diagnostics.append(diagnostic)
                if i in (15,47):
                    mult=pm.last_m_effective.detach().float() if mode!='A' else torch.ones_like(rho)
                    qmap=pm.q_actual.detach().float().mean(1,keepdim=True)
                    maps[i]=(mult.cpu(),rho.cpu(),qmap.cpu())
                    previews[i]=unit(decoded['x_hat'],h,w).detach().float().cpu()
    assert buffer.tell()==len(native) and len(states)==64 and len(diagnostics)==63
    del im,pm
    torch.cuda.empty_cache()
    parity=None
    if record['measurement'].startswith('reused_'):
        # Rebuild an original v2 decoder independently; compare exact tensors,
        # including RGB, before accepting any reused quality measurement.
        from runtime import SkipDomainDMC
        im,pm,_=task.load_models(ctx,decoded_mode,decoded_alpha)
        pm.__class__=SkipDomainDMC
        im.set_use_two_entropy_coders(bool(sps['ec_part']))
        pm.set_use_two_entropy_coders(bool(sps['ec_part']))
        pm.clear_dpb();pm.set_curr_poc(0)
        buffer=io.BytesIO(native);helper=stream.SPSHelper();original_states=[]
        with torch.inference_mode():
            for i in range(64):
                head=stream.read_header(buffer)
                while head['nal_type']==stream.NalType.NAL_SPS:
                    helper.add_sps_by_id(stream.read_sps_remaining(buffer,head['sps_id']))
                    head=stream.read_header(buffer)
                decoded_sps=helper.get_sps_by_id(head['sps_id'])
                q,bits=stream.read_ip_remaining(buffer)
                decoded=im.decompress(bits,decoded_sps,q) if i==0 else pm.decompress(bits,decoded_sps,q)
                if i==0:
                    pm.clear_dpb();pm.set_curr_poc(0);pm.add_ref_frame(None,decoded['x_hat'])
                original_states.append({'frame':i,'reference':digest(decoded['x_hat'] if i==0 else pm.dpb[0].feature),
                    'latent':None if i==0 else digest(pm.decoding_trace['y_hat']),
                    'qmap':None if i==0 else digest(pm.q_actual),'rgb':digest(decoded['x_hat'])})
        assert buffer.tell()==len(native) and states==original_states
        parity={'all_64_RGB_exact':True,'all_64_references_exact':True,'all_P_latents_qmaps_exact':True,
                'tensor_tolerance':0,'max_abs_error':0.0,'runtime_class':'runtime.SkipDomainDMC'}
        del im,pm
        torch.cuda.empty_cache()
    assert set(maps)=={15,47}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    from PIL import Image
    import numpy as np
    output=P/'figures/quant_maps';output.mkdir(parents=True,exist_ok=True)
    preview_dir=P/'previews';preview_dir.mkdir(exist_ok=True)
    for frame,(m,rho,qmap) in maps.items():
        crop=previews[frame][0,:,200:456,400:656].permute(1,2,0).clamp(0,1).numpy()
        Image.fromarray((crop*255).round().astype(np.uint8)).save(preview_dir/f'{key}_f{frame}.png')
        fig,axes=plt.subplots(1,3,figsize=(10,3.2))
        for ax,(title,tensor,vmin,vmax) in zip(axes,[('effective H multiplier',m,.5,2.0),
                                                    ('actual rho',rho,.5,2.0),('actual q mean-channel',qmap,.5,16.0)]):
            image=F.interpolate(tensor,size=(256,256),mode='nearest')[0,0].numpy()
            view=ax.imshow(image,cmap='viridis',vmin=vmin,vmax=vmax)
            ax.set_title(title);ax.axis('off')
            fig.colorbar(view,ax=ax,fraction=.046,pad=.02)
        fig.suptitle(f'{source} QP{qp} {mode} alpha={record["alpha"]} frame{frame} {record["actual_bpp"]:.6f} bpp')
        fig.savefig(output/f'{key}_f{frame}.png',dpi=120)
        plt.close(fig)
    audit_dir=P/'server_artifacts/receiver_audits';audit_dir.mkdir(exist_ok=True)
    task.write_csv(P/'server_artifacts/receiver_diagnostics'/f'{key}.csv',diagnostics)
    import sys
    audit={'status':'PASS','task_key':key,'stream_sha256':record['stream_sha256'],
        'checkpoint_sha256':checkpoint_sha,'source_sha256':record['source_sha256'],
        'maps_code_sha256':task.sha(P/'maps.py'),'task_code_sha256':task.sha(P/'task.py'),
        'ablation_code_sha256':task.sha(P/'ablation.py'),
        'runtime_path':str(Path(sys.modules['runtime'].__file__).resolve()),
        'module_paths':{name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
                        if name in ('train','runtime','ablation','core','gvc_hooks','src.models.video_model_gvcrt')},
        'frames':64,'P_frames':63,'state_hashes':states,'v2_receiver_parity':parity,
        'elapsed_seconds':time.monotonic()-started,'physical_gpu':gpu,
        'peak_MiB':torch.cuda.max_memory_reserved()/2**20,
        'rho_post_dtype_max_abs_error':0.0,'mean_definition_max_abs_error':max(x['mean_definition_max_abs_error'] for x in diagnostics),
        'roll_multiset_exact':True if mode=='ROLL' else None}
    (audit_dir/f'{key}.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True)
    ap.add_argument('--qp',required=True,type=int)
    ap.add_argument('--mode',required=True)
    ap.add_argument('--alpha-index',required=True,type=int)
    ap.add_argument('--gpu',required=True,type=int)
    a=ap.parse_args();run(a.source,a.qp,a.mode,a.alpha_index,a.gpu)
