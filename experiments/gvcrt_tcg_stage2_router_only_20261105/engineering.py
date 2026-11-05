"""Disposable engineering replicas; no formal optimizer cursor is modified."""
from common_s2 import *
import argparse,traceback
from model_context import build,assert_reset,history_snapshot,frozen_items
from pipeline import Frames,raw,Pipeline,optimized_loss

def forward_probe(mode,rank):
    import torch
    c=build(rank,train=True,mode=mode,reference=mode in ('reference','reference_repeat','off'),losses=False)
    seed(SEED);guard=FrozenGuard(frozen_items(c));rows=plan_rows();outputs=[];records=[]
    for index in [0,1]:
        row=rows[index];frames=Frames(raw(row),c['device']);pm=c['pm'];captured=[]
        hook=pm.dec.register_forward_hook(lambda m,a,o:captured.append(o))
        assert_reset(c)
        with torch.no_grad():
            pm.add_ref_frame(None,c['im'].compress(c['core'].codec_input(frames[0]),row['external_qp'])['x_hat'])
            for pos in range(1,64):
                qa=c['qp'].frame_qp(pm,row['external_qp'],pos)
                if mode=='off':
                    # Original generator control disables injection but still caches current.
                    gen=c['models']['generator'];original=gen.forward
                    def without(*a,**kw):
                        a=list(a)
                        if len(a)>=3:a[2]=False
                        else:kw['use_tcg']=False
                        return original(*a,**kw)
                    gen.forward=without
                try:rec,rate,_=c['core'].joint_ste_forward(pm,frames[pos],qa)
                finally:
                    if mode=='off':gen.forward=original
                feature=captured.pop();assert not captured
                pm.add_ref_frame(feature.detach(),(rec.detach()*2-1).half())
                outputs.append(rec.cpu())
                records.append(dict(sample=index,frame=pos,codec_feature=tensor_sha(feature),
                    reconstruction=tensor_sha(rec),TCG=history_snapshot(c)['TCG'],feature_shape=list(feature.shape),
                    mask_mean=None if c['router'] is None or c['router'].last_mask is None else float(c['router'].last_mask.mean())))
        hook.remove();assert_reset(c);frames.close();guard.check()
    state=dict(mode=mode,records=records,output=torch.cat(outputs),frozen_sha256=guard.full(),provenance=c['provenance'])
    atomic_torch(ROOT/f'large_artifacts/engineering/{mode}.pt',state)
    dump(ROOT/f'audits/forward_{mode}.json',dict(status='PASS',records=records,artifact_sha256=sha(ROOT/f'large_artifacts/engineering/{mode}.pt'),formal_updates=0))

def state_probe(c):
    import torch
    p=c['pm'];g=c['models']['generator'];r=c['router'];device=c['device'];assert_reset(c)
    dtype=next(c['models']['bridge'].parameters()).dtype
    q=p.q_scale_recon[5:6].to(dtype);x=torch.randn(1,256,32,32,device=device,dtype=dtype)
    with torch.no_grad():
        a=p.recon_generation_net(x,q);first=history_snapshot(c)
        assert not r.last_had_history and not r._previous.requires_grad
        p.recon_generation_net(x+1,q,update_buffer=False)
        assert history_snapshot(c)==first
        p.recon_generation_net(x+2,q,use_tcg=False,update_buffer=False)
        assert history_snapshot(c)==first
        p.add_ref_frame(x,None);before=history_snapshot(c)
        p.prepare_feature_adaptor_i(5)
        after=history_snapshot(c)
        assert before['TCG']==after['TCG'] and before['router_previous']==after['router_previous']
        assert p.dpb[0].frame is not None
        p.recon_generation_net(torch.randn(1,256,40,40,device=device,dtype=dtype),q)
        assert r._previous.shape[-2:]==(40,40) and not r.last_had_history
        assert all(getattr(g,n)._previous is not None for n in ['tcg_low','tcg_mid','tcg_out'])
        assert_reset(c)
    return dict(status='PASS',missing_history=True,history_detached=True,update_buffer_false_unchanged=True,
                internal_reference_bypasses_and_preserves_history=True,size_change_reset=True,sequence_I_reset=True)

def gradient_probe(rank):
    import torch
    import loss
    state_context=build(rank,train=False,losses=False);state_checks=state_probe(state_context);del state_context;torch.cuda.empty_cache()
    c=build(rank,train=True,losses=True);seed(SEED+rank);rows=plan_rows();guard=FrozenGuard(frozen_items(c))
    # Disposable full-sequence original loss vs identical loss with precomputed frozen GT-RAFT.
    row=rows[0];initial={n:t.detach().clone() for n,t in c['router'].state_dict().items()};saved=rng_state()
    f=Frames(raw(row),c['device']);start=time.time()
    gs,a=loss.basic_crop(c['ctx'],f,row['external_qp'],raft=c['raft'],raft_transform=c['transform']);f.close()
    reference_seconds=time.time()-start;assert_reset(c);restore_rng(saved)
    pipe=Pipeline([row,row,row],rank);f=pipe.next(row)
    hg,b=optimized_loss(loss.basic_crop)(c['ctx'],f,row['external_qp'],raft=c['raft'],raft_transform=c['transform']);f.close()
    numerator=math.sqrt(sum(float((x-y).double().square().sum()) for x,y in zip(gs,hg)))
    denominator=math.sqrt(sum(float(x.double().square().sum()) for x in gs))
    relative=numerator/max(denominator,1e-30);assert relative<1e-3,('GT_AUXILIARY_GRADIENT_EQUIVALENCE',relative)
    assert max(abs(a['losses'][k]-b['losses'][k]) for k in a['losses'])<1e-5
    # Original repeat error, same initial state and RNG, no optimizer update.
    assert_reset(c);restore_rng(saved);f=Frames(raw(row),c['device'])
    repeat,aa=loss.basic_crop(c['ctx'],f,row['external_qp'],raft=c['raft'],raft_transform=c['transform']);f.close()
    repeat_rel=math.sqrt(sum(float((x-y).double().square().sum()) for x,y in zip(gs,repeat)))/max(denominator,1e-30)
    smoke=[]
    for step in range(1,4):
        if step==1:grad,diag=hg,b
        else:
            f=pipe.next(row);grad,diag=optimized_loss(loss.basic_crop)(c['ctx'],f,row['external_qp'],raft=c['raft'],raft_transform=c['transform']);f.close()
        assert all(torch.isfinite(v).all() for v in grad)
        norm=math.sqrt(sum(float(v.double().square().sum()) for v in grad));assert norm>0
        named={n:float(v.float().norm()) for (n,p),v in zip(c['named'],grad)}
        if step==1:assert all(v==0 for n,v in named.items() if 'net.4.' not in n)
        if step>=2:assert all(v>0 for n,v in named.items())
        c['optimizer'].zero_grad(set_to_none=True)
        for p,v in zip(c['params'],grad):p.grad=v
        clipped=torch.nn.utils.clip_grad_norm_(c['params'],1.,error_if_nonfinite=True);c['optimizer'].step()
        guard.check();assert_reset(c)
        smoke.append(dict(step=step,gradient_norm=norm,parameter_gradient_norms=named,losses=diag['losses'],frozen_unchanged=True,
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(rank)))
    pipe.close();assert any(not torch.equal(initial[n],v) for n,v in c['router'].state_dict().items());guard.full()
    # Motion term must retain reconstruction differentiability with RAFT frozen.
    previous=torch.rand(1,3,256,256,device=c['device']);current=torch.rand_like(previous,requires_grad=True)
    x,y=c['transform'](previous.detach(),current);motion=c['raft'](x,y)[-1].abs().mean()
    motion_grad=torch.autograd.grad(motion,current)[0];assert torch.isfinite(motion_grad).all() and motion_grad.norm()>0
    dump(ROOT/'audits/gradient_and_state.json',dict(status='PASS',formal_updates=0,disposable_updates=3,
        gradient_relative_L2=relative,reference_repeat_relative_L2=repeat_rel,tolerance=1e-3,
        maximum_loss_component_absolute_difference=max(abs(a['losses'][k]-b['losses'][k]) for k in a['losses']),
        reconstruction_scale=4096,unscale_count=1,original_loss_sha256=sha(ROOT/'loss.py'),reference_seconds=reference_seconds,
        state_checks=state_checks,smoke=smoke,motion_reconstruction_gradient_norm=float(motion_grad.norm()),RAFT_parameters_frozen=not any(p.requires_grad for p in c['raft'].parameters()),
        baseline_provenance=c['provenance'],frozen_sha256=guard.digest))

def compare():
    import torch
    data={n:torch.load(ROOT/f'large_artifacts/engineering/{n}.pt',map_location='cpu',weights_only=True) for n in ['reference','reference_repeat','bypass','zero','one','off','enabled']}
    r=data['reference'];repeat=data['reference_repeat']
    assert torch.equal(r['output'],repeat['output']),'REFERENCE_BACKEND_NONDETERMINISTIC'
    for name in ['bypass','zero']:
        d=data[name];assert torch.equal(r['output'],d['output']),('ORIGINAL_TCG_RECONSTRUCTION_MISMATCH',name)
        for a,b in zip(r['records'],d['records']):assert all(a[k]==b[k] for k in ['codec_feature','reconstruction','TCG','feature_shape']),(name,a['frame'])
    assert torch.equal(data['one']['output'],data['off']['output']),'FORCED_ONE_RESIDUAL_DISABLE_MISMATCH'
    for a,b in zip(data['one']['records'],data['off']['records']):assert a['TCG']==b['TCG']
    e=data['enabled'];diff=(e['output']-r['output']).abs();means=[x['mask_mean'] for x in e['records']]
    assert min(means)>0.00999 and max(means)<0.01001
    assert all(a['codec_feature']==b['codec_feature'] for a,b in zip(r['records'],e['records']))
    dump(ROOT/'audits/forward_equivalence.json',dict(status='PASS',bypass_strict_equal=True,zero_strict_equal=True,one_matches_original_disabled=True,
        reference_repeat_maxabs=0.,enabled0_mask_mean_min=min(means),enabled0_mask_mean_max=max(means),
        enabled0_reconstruction_maxabs=float(diff.max()),enabled0_reconstruction_meanabs=float(diff.mean()),enabled0_claimed_identical=False,
        codec_features_strict_equal=True,TCG_pre_residual_cache_strict_equal_in_bypass_and_zero=True,samples=2,P_frames_per_sample=63))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode');ap.add_argument('--rank',type=int,default=0);args=ap.parse_args();env()
    if args.mode=='compare':return compare()
    if args.mode=='gradient':return gradient_probe(args.rank)
    if args.mode.startswith('rans_'):
        from model_context import backend
        from evaluate import point,videos
        reference=args.mode=='rans_reference';mode='bypass' if args.mode=='rans_bypass' else 'enabled'
        backend(reference,args.rank)
        v=next(v for v in videos('final') if v['sequence']=='Kimono1')
        p=point(v,5,None,'engineering_'+args.mode,args.rank,scoring=False,reference=reference,mode=mode)
        dump(ROOT/f'audits/{args.mode}.json',p)
        return
    return forward_probe(args.mode,args.rank)
if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
