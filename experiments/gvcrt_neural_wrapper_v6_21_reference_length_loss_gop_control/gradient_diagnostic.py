"""No-update diagnostic only; scaling here never changes training configuration."""
import argparse,traceback
from io21 import *
from window_training import replay_rows,replay
from gradient_tools import stats
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args()
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    from patchgan import create
    torch.set_num_threads(2);frozen();v4,im,pm,models,quality=load_training(torch.device('cuda:0'));initial={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm);qh=[v4.module_hash(m) for m in quality]
    disc,_,_=create(torch.device('cuda:0'));dh=v4.module_hash(disc)
    common=module('gop21_diag_common',V62/'fullqp_common.py');rows=[];params=[p for m in models.values() for p in m.parameters()]
    plan=replay_rows('C');samples=[plan[0],next(r for r in plan if r['domain']=='ulong')]
    for row in samples:
        fs=replay(row,'C',torch.device('cuda:0'));q=row['external_qp']
        for scale in (1.,256.,4096.):
            for p in params:p.grad=None
            pm.clear_dpb();pm.set_curr_poc(0)
            with torch.no_grad():pm.add_ref_frame(None,im.compress(v4.codec_input(fs[48]),q)['x_hat'])
            values={}
            def capture(name):
                def hook(m,args,out):
                    out.retain_grad();values[name]=out
                return hook
            hooks=[pm.enc.register_forward_hook(capture('encoder_latent_FP16')),pm.dec.register_forward_hook(capture('decoder_feature_FP16'))]
            try:
                proxy=models['wrapper'](fs[49]);proxy.retain_grad();values['proxy_FP32']=proxy
                rec,_,_=v4.joint_ste_forward(pm,proxy,common.frame_qp(pm,q,1));rec.retain_grad();values['reconstruction_FP32']=rec
                adv=-disc(rec*2-1).mean();weighted=.01*adv/15
                (weighted*scale).backward()
                gs={k:stats([None if p.grad is None else p.grad.float()/scale for p in m.parameters()],list(m.parameters())) for k,m in models.items()}
                layers={k:dict(dtype=str(t.dtype),requires_grad=t.requires_grad,stats=stats([None if t.grad is None else t.grad.float()/scale],[t])) for k,t in values.items()}
                rows.append(dict(domain=row['domain'],plan_step=row['adaptation_step'],source_indices=row['source_frame_indices'],crop=[row['crop_x'],row['crop_y']],input_crop_sha256=row['cropped_RGB_sha256'][49],external_qp=q,actual_qp=common.frame_qp(pm,q,1),backward_diagnostic_scale=scale,unscaled_weighted_loss=float(weighted),LPBG_gradients=gs,layers=layers,optimizer_updates=0,training_settings_changed=False))
            finally:
                for h in hooks:h.remove()
            print('DIAG',row['domain'],scale,gs,flush=True)
    assert initial=={k:v4.module_hash(m) for k,m in models.items()} and core==v4.compression_hash(im,pm) and qh==[v4.module_hash(m) for m in quality] and dh==v4.module_hash(disc)
    dump(ROOT/'audits/gradient_path_diagnostic.json',dict(status='RECORDED',optimizer_updates=0,models_unchanged=True,training_config_unchanged=True,rows=rows))
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'audits/gradient_path_diagnostic.json',dict(status='FAIL',error=traceback.format_exc()));raise

