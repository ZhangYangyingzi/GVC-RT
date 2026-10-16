"""Real RANS identity comparison; independent decoder and latent interface probe."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='7',CUDA_DEVICE_ORDER='PCI_BUS_ID')
from runtime import *

def main():
    import torch
    ctx=setup(BRANCHES[0]);v4,im,pm,models,quality=ctx
    pm.recon_generation_net.half()
    idec,pdec=v4.load_models(torch.device('cuda:0'),force_zero_thres=.12)
    before=v4.compression_hash(im,pm)
    row=load(V32/'plans/shared.json')['updates'][0];f=frames(row);x0=f[0];x1=f[1];f.close();qp=row['external_qp']
    latent=[];z=[];prequant=[]
    hy=pm.enc.register_forward_hook(lambda m,a,o:prequant.append(o.detach().clone()))
    h=pdec.dec.register_forward_pre_hook(lambda m,args:latent.append(args[0].detach().clone()))
    hz=pdec.hyper_dec.register_forward_pre_hook(lambda m,args:z.append(args[0].detach().clone()))
    def run(bypass):
        pm.clear_dpb();pdec.clear_dpb();pm.set_curr_poc(0);pdec.set_curr_poc(0)
        i=im.compress(v4.codec_input(x0),qp);si=dict(height=256,width=256,ec_part=1)
        d=idec.decompress(i['bit_stream'],si,qp);assert torch.equal(i['x_hat'],d['x_hat'])
        pm.add_ref_frame(None,d['x_hat']);pdec.add_ref_frame(None,d['x_hat'])
        value=bypass(x1);enc_in=v4.codec_input(value);qa=pm.shift_qp(qp,0)
        e=pm.compress(enc_in,qa);decoded=pdec.decompress(e['bit_stream'],si,qa)['x_hat']
        return enc_in.clone(),e['bit_stream'],decoded.clone(),latent[-1].clone()
    with torch.no_grad():
        a=run(lambda x:x);b=run(torch.nn.Identity())
    h.remove();hz.remove();hy.remove()
    checks=dict(codec_input_equal=torch.equal(a[0],b[0]),prequant_y_equal=torch.equal(prequant[0],prequant[1]),RANS_bytes_equal=a[1]==b[1],reconstruction_equal=torch.equal(a[2],b[2]),decoded_y_hat_equal=torch.equal(a[3],b[3]),compression_hash_unchanged=before==v4.compression_hash(im,pm))
    for name,val in checks.items():assert val,name
    p=ROOT/'bitstreams/identity_P.bin';p.parent.mkdir(exist_ok=True);p.write_bytes(a[1])
    dump('sanity/identity.json',dict(status='PASS',checks=checks,QP=qp,actual_P_QP=pm.shift_qp(qp,0),shape=list(x1.shape),payload_bytes=len(a[1]),payload_path=str(p),sha256=sha(p),scope='one training sample, 256x256; strict identity bypass same original receiver; P payload only, not formal total video bitrate'))
    dump('latent_interface_audit.json',dict(status='PASS_REAL_HOOK',primary=dict(tensor_name='DMC.decompress.y_hat -> DMC.dec first argument',shape=list(latent[0].shape),dtype=str(latent[0].dtype),channel_axis=1,quantization_position='centered rounded/clipped symbols + conditional means, multiplied by q_dec',entropy_position='after decompress_prior_2x_part2; y groups decoded from RANS',decoder_consumption='self.dec(y_hat, ctx, q_decoder)'),hyperlatent=dict(tensor_name='z_hat -> hyper_dec first argument',shape=list(z[0].shape),dtype=str(z[0].dtype),entropy_position='after bit_estimator_z.get_z; consumed by conditional prior, not direct synthesis'),source=str(REPO/'src/models/video_model_gvcrt.py'),source_sha256=sha(REPO/'src/models/video_model_gvcrt.py')))
    print('IDENTITY REAL RANS PASS',checks,flush=True)

if __name__=='__main__':
    import traceback
    try:main()
    except Exception:
        dump('sanity/failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
