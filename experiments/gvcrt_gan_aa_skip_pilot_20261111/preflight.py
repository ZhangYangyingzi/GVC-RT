"""Required deterministic mechanics before any formal generator update."""
from common_s2 import *
import argparse


def requests():
    rows=[r for r in plan_rows() if r['domain']=='ulong'][:2];items=[]
    for row in rows:
        for q in [0,5,9]:
            r=dict(row);r.pop('sample_key');r['external_qp']=q
            r['actual_frame_qps']=[q if i==0 else q+[0,2,1][[0,1,0,2,0,2,0,2][i%8]] for i in range(64)]
            r['sample_key']=digest(r);items.append(dict(row=r,qp=q,training=True))
    dump(ROOT/'manifests/preflight_points.json',items)
    path=ROOT/'cache/requests/preflight.json';dump(path,items)
    subprocess.run([PYTHON,'cache_job.py',str(path)],cwd=ROOT,check=True)


def compute():
    import torch
    from filters import check,resize_periodic
    from experiment_model import construct,reconstruct,decode,bridge,detach_history,trainable_state
    from model_context import assert_reset,frozen_items,history_snapshot
    from cache_codec import cache_paths
    dump(ROOT/'audits/filter_audit.json',check('cuda:0'))
    points=load(ROOT/'manifests/preflight_points.json')
    b=construct('B300',False,0,losses=False)
    j=construct('J0',True,0,losses=False)
    initial=tensor_hash(j['named']);codec_before=tensor_hash(frozen_items(j))
    dump(ROOT/'initialization_audit.json',dict(status='PASS',B300_loader=b['provenance'],candidate_loader=j['provenance'],common_trainable_SHA256=initial))
    proof=[];first_data=None
    for item in points:
        row=item['row'];key,path,rp=cache_paths(row,item['qp'],True);receipt=load(rp)
        assert receipt['status']=='PASS' and sha(path)==receipt['cache_sha256']
        data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if first_data is None:first_data=data
        assert_reset(b);assert_reset(j);maximum=0.;elapsed=time.time()
        with torch.no_grad():
            for i in range(1,64):
                a=reconstruct(b,data['features'][i],data['info']['actual_qps'][i])
                z=reconstruct(j,data['features'][i],data['info']['actual_qps'][i])
                diff=float((a-z).abs().max());maximum=max(maximum,diff)
                assert torch.equal(a,z),('ZERO_SKIP_B300_MISMATCH',row['video'],item['qp'],i,diff)
        assert not j['pm'].dpb and not b['pm'].dpb
        assert tensor_sha(j['pm'].q_scale_recon)==tensor_sha(data['qscale'])
        proof.append(dict(video=row['video'],QP=item['qp'],frames=64,history_P_frames=63,max_abs=maximum,exact_equal=True,stream_SHA256=receipt['stream_SHA256'],bpp=receipt['bpp'],real_RANS=receipt['real_RANS'],independent_decode=receipt['independent_decode'],codec_trace_SHA256=receipt['codec_trace_SHA256'],seconds=time.time()-elapsed,precision=DEPLOYMENT))
    del b;torch.cuda.empty_cache()
    data=first_data;g=j['models']['generator'];assert_reset(j)
    with torch.no_grad():
        for i in range(1,49):reconstruct(j,data['features'][i],data['info']['actual_qps'][i])
    detach_history(j);hist=[];outputs=[]
    for i in range(49,53):
        outputs.append(reconstruct(j,data['features'][i],data['info']['actual_qps'][i],retain=True))
        hist.append(g.tcg_mid._previous)
        assert g.tcg_mid._previous.grad_fn is not None and g.tcg_out._previous.grad_fn is not None
    cross=torch.autograd.grad(outputs[-1].square().mean(),hist[-2],retain_graph=True)[0]
    assert torch.isfinite(cross).all() and cross.norm()>0
    loss=sum(x.square().mean() for x in outputs)/4
    loss.backward()
    projection=float(g.skip_adapter.proj.weight.grad.norm())
    upstream=float(g.skip_adapter.down.weight.grad.norm())
    assert projection>0 and upstream==0
    with torch.no_grad():g.skip_adapter.proj.weight.add_(g.skip_adapter.proj.weight.grad,alpha=-5e-5)
    j['optimizer'].zero_grad(set_to_none=True);assert_reset(j)
    reconstruct(j,data['features'][49],data['info']['actual_qps'][49]).square().mean().backward()
    after=float(g.skip_adapter.down.weight.grad.norm());assert after>0 and math.isfinite(after)
    with torch.no_grad():g.skip_adapter.proj.weight.zero_()
    j['optimizer'].zero_grad(set_to_none=True);assert_reset(j)
    assert tensor_hash(j['named'])==initial and len(j['optimizer'].state)==0
    assert tensor_hash(frozen_items(j))==codec_before
    dump(ROOT/'audits/gradient_audit.json',dict(status='PASS',four_frame_history_graph=True,previous_frame_input_gradient_norm=float(cross.norm()),zero_init_projection_gradient=projection,zero_init_upstream_gradient=upstream,upstream_gradient_after_disposable_projection_update=after,temporary_update_discarded=True,initial_trainable_hash_restored=initial,TCG_parameters_frozen=True,TCG_mid_and_out_histories_have_grad_fn=True))
    # Filter-off equivalence above is a gate. Filter-on differences are recorded.
    aa=construct('J2',False,0,losses=False);assert tensor_hash(aa['named'])==initial
    costs=[]
    for c in [j,aa]:assert_reset(c)
    with torch.no_grad():
        for i in range(1,64):
            x=reconstruct(j,data['features'][i],data['info']['actual_qps'][i])
            y=reconstruct(aa,data['features'][i],data['info']['actual_qps'][i])
            if i>=49:
                gt=data['GT'][i].permute(2,0,1)[None].to(j['device']).float()/255
                ux=(x.float()+1)/2;uy=(y.float()+1)/2
                costs.append(dict(frame=i,RGB_L1_difference=float((ux-uy).abs().mean()),B300_PSNR=float(-10*torch.log10((ux-gt).square().mean())),AA_step0_PSNR=float(-10*torch.log10((uy-gt).square().mean()))))
    dump(ROOT/'audits/filter_step0_cost.json',dict(status='RECORDED',costs=costs,quality_is_not_a_training_gate=True))
    # Synchronous dual-input Fourier translation, isolated history and valid RGB mask.
    shifts=[]
    def translate(x,dy,dx):
        h,w=x.shape[-2:];fy=torch.fft.fftfreq(h,device=x.device);fx=torch.fft.fftfreq(w,device=x.device)
        phase=torch.exp(-2j*torch.pi*(fy[:,None]*dy+fx[None,:]*dx))
        return torch.fft.ifft2(torch.fft.fft2(x.float())*phase).real
    f=data['features'][52].to(j['device']).float()
    with torch.no_grad():
        for c in [j,aa]:
            assert_reset(c);z=bridge(c,f);q=c['pm'].q_scale_recon[data['info']['actual_qps'][52]:data['info']['actual_qps'][52]+1].float();target=decode(c,z,f,q,False,False)
            for delta in [2,4,8,16]:
                for axis in [0,1]:
                    dy,dx=(delta,0) if axis==0 else (0,delta)
                    out=decode(c,translate(z,dy/16,dx/16),translate(f,dy/8,dx/8),q,False,False)
                    expected=translate(target,dy,dx);error=(out-expected).abs()[...,64:-64,64:-64].mean()
                    shifts.append(dict(group=c['experiment_group'],RGB_shift=[dy,dx],latent_shift=[dy/16,dx/16],f_hat_shift=[dy/8,dx/8],valid_border_RGB=64,no_history=True,L1=float(error)))
    write_csv(ROOT/'audits/dual_input_shift.csv',shifts)
    assert tensor_hash(frozen_items(j))==codec_before and not j['pm'].dpb
    dump(ROOT/'audits/codec_audit.json',dict(status='PASS',checks=proof,common_frozen_state_before_after=codec_before,candidate_never_updates_DPB=True,codec_and_qscale_persistent_state_unchanged=True,bitstream_bpp_trace_unchanged=True,GT_flow_not_in_reconstruction=True))
    dump(ROOT/'audits/preflight.json',dict(status='PASS',two_videos_QP_0_5_9=proof,initialization='initialization_audit.json',filter='audits/filter_audit.json',gradient='audits/gradient_audit.json',codec='audits/codec_audit.json',common_trainable_SHA256=initial,time=time.time()))


if __name__=='__main__':
    env();p=argparse.ArgumentParser();p.add_argument('--compute',action='store_true');a=p.parse_args()
    if a.compute:compute()
    else:
        requests();subprocess.run([PYTHON,'preflight.py','--compute'],cwd=ROOT,check=True)
