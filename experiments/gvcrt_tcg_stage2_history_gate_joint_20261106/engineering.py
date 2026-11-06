"""Discardable correctness checks. Forced masks are never quality-scan candidates."""
from common_s2 import *
from model_context import build,assert_reset,history_snapshot,frozen_items,trainable_hash,tcg_state
from pipeline import Frames,raw,Pipeline,optimized_loss
import argparse,traceback

def forward_probe(mode,rank):
    import torch
    reference=mode in ('reference','reference_repeat')
    c=build(rank,train=True,mode='enabled' if mode=='enabled' else 'zero',reference=reference,losses=False)
    seed(SEED);guard=FrozenGuard(frozen_items(c));rows=plan_rows();outputs=[];records=[]
    for index in [0,1]:
        row=rows[index];frames=Frames(raw(row),c['device']);captured=[];pm=c['pm']
        hook=pm.dec.register_forward_hook(lambda m,a,o:captured.append(o));assert_reset(c)
        with torch.no_grad():
            pm.add_ref_frame(None,c['im'].compress(c['core'].codec_input(frames[0]),row['external_qp'])['x_hat'])
            for pos in range(1,64):
                q=c['qp'].frame_qp(pm,row['external_qp'],pos)
                rec,rate,_=c['core'].joint_ste_forward(pm,frames[pos],q)
                feature=captured.pop();assert not captured
                pm.add_ref_frame(feature.detach(),(rec.detach()*2-1).half())
                outputs.append(rec.cpu());records.append(dict(sample=index,frame=pos,codec_feature=tensor_sha(feature),reconstruction=tensor_sha(rec),
                    TCG=history_snapshot(c)['TCG'],feature_shape=list(feature.shape),
                    M=None if c['router'] is None or c['router'].last_mask is None else float(c['router'].last_mask.mean())))
        hook.remove();assert_reset(c);frames.close();guard.check()
    atomic_torch(ROOT/f'large_artifacts/engineering/{mode}.pt',dict(mode=mode,records=records,output=torch.cat(outputs),provenance=c['provenance']))
    dump(ROOT/f'audits/forward_{mode}.json',dict(status='PASS',records=records,formal_joint_updates=0,baseline_frozen_sha256=guard.full()))

def layer_and_state_checks(rank):
    import torch
    c=build(rank,train=False,losses=False);pm=c['pm'];gen=c['models']['generator'];router=c['router'];seed(SEED)
    records=[];before=tensor_hash([(n,t) for n,t in c['pm'].state_dict().items() if n.startswith(TRAINABLE_PREFIXES)])
    pm.half()
    after=tensor_hash([(n,t) for n,t in c['pm'].state_dict().items() if n.startswith(TRAINABLE_PREFIXES)])
    assert before==after and all(p.dtype==torch.float32 for p in c['params']),'PARENT_HALF_ROUNDED_FP32_ISLAND'
    with torch.no_grad():
        for name in ['tcg_low','tcg_mid','tcg_out']:
            layer=getattr(gen,name);ch=layer.previous_adapter.in_channels
            previous=torch.randn(1,ch,8,12,device=c['device'],dtype=torch.float16)
            current=torch.randn_like(previous);layer._previous=previous.detach();captures={}
            hooks=[layer.previous_adapter.register_forward_hook(lambda m,a,o:captures.update(history=o.detach().clone())),
                layer.current_adapter.register_forward_hook(lambda m,a,o:captures.update(current=o.detach().clone())),
                layer.fusion.register_forward_pre_hook(lambda m,a:captures.update(concat=a[0].detach().clone())),
                layer.fusion.register_forward_hook(lambda m,a,o:captures.update(delta=o.detach().clone()))]
            mask=torch.zeros(1,1,8,12,device=c['device'],dtype=torch.float32);mask[:,:,:,4:8]=1.;mask[:,:,:,8:]=1e-5
            snapshots=tensor_sha(layer._previous)
            with torch.autocast('cuda',dtype=torch.float16):out=layer(current,suppress_mask=mask,update_buffer=False)
            hp=captures['history'];hc=captures['current'];joined=captures['concat'];keep=1.-mask
            assert joined.dtype==hp.dtype==hc.dtype==captures['delta'].dtype==torch.float32
            assert torch.equal(joined[:,:hp.shape[1]],hp*keep)
            assert torch.equal(joined[:,hp.shape[1]:],hc),'CURRENT_PROJECTION_MUST_BE_UNGATED'
            assert torch.count_nonzero(joined[:,:hp.shape[1],:,4:8])==0
            assert torch.equal(joined[:,:hp.shape[1],:,:4],hp[:,:,:,:4])
            assert torch.all(keep[:,:,:,8:]<1.) and hc.norm()>0
            assert torch.equal(out,(current.float()+captures['delta']).half())
            assert tensor_sha(layer._previous)==snapshots
            one=torch.ones_like(mask);out_one=layer(current,suppress_mask=one,update_buffer=False)
            assert torch.count_nonzero(captures['concat'][:,:hp.shape[1]])==0
            assert captures['concat'][:,hp.shape[1]:].norm()>0 and captures['delta'].norm()>0
            # No fixed-M quality scoring: inspect branch tensors and residual only.
            records.append(dict(layer=name,status='PASS',history_zero_at_M1=True,current_projection_nonzero_at_M1=True,
                residual_norm_at_M1=float(captures['delta'].norm()),M1_claimed_TCG_disabled=False,spatial_mask_not_averaged=True,
                small_nonzero_mask_keep_distinct_from_one=True,output_dtype=str(out.dtype),internal_dtype='torch.float32',outer_autocast='FP16',
                history_branch_only=True,pre_residual_cache_preserved=True))
            for h in hooks:h.remove()
        assert_reset(c);dtype=next(c['models']['bridge'].parameters()).dtype;q=pm.q_scale_recon[5:6].to(dtype)
        x=torch.randn(1,256,32,32,device=c['device'],dtype=dtype)
        pm.recon_generation_net(x,q);first=history_snapshot(c)
        assert not router.last_had_history and not router._previous.requires_grad
        pm.recon_generation_net(x+1,q,update_buffer=False);assert history_snapshot(c)==first
        pm.recon_generation_net(x+2,q,use_tcg=False,update_buffer=False);assert history_snapshot(c)==first
        pm.add_ref_frame(x,None);before_internal=history_snapshot(c);pm.prepare_feature_adaptor_i(5);after_internal=history_snapshot(c)
        assert before_internal['TCG']==after_internal['TCG'] and before_internal['router_previous']==after_internal['router_previous']
        pm.recon_generation_net(torch.randn(1,256,40,40,device=c['device'],dtype=dtype),q)
        assert router._previous.shape[-2:]==(40,40) and not router.last_had_history
        assert_reset(c)
    dump(ROOT/'audits/branch_precision_and_state.json',dict(status='PASS',layer_checks=records,parent_half_parameter_values_exact=True,
        all_Router_TCG_parameters_FP32=True,missing_history=True,update_buffer_false_exact=True,internal_reference_bypass_preserves_histories=True,
        dimension_reset=True,sequence_I_reset=True,formal_joint_updates=0))

def gradient_probe(rank):
    import torch
    import loss
    c=build(rank,train=True,losses=True);seed(SEED+rank);guard=FrozenGuard(frozen_items(c));row=plan_rows()[0]
    initial={n:p.detach().clone() for n,p in c['named']};saved=rng_state();frames=Frames(raw(row),c['device'])
    gs,a=loss.basic_crop(c['ctx'],frames,row['external_qp'],raft=c['raft'],raft_transform=c['transform']);frames.close();assert_reset(c)
    restore_rng(saved);pipe=Pipeline([row,row,row],rank);frames=pipe.next(row)
    hg,b=optimized_loss(loss.basic_crop)(c['ctx'],frames,row['external_qp'],raft=c['raft'],raft_transform=c['transform']);frames.close()
    def norm(g):return math.sqrt(sum(float(v.double().square().sum()) for v in g))
    split=c['router_parameter_tensors'];relative={}
    for name,sl in [('all',slice(None)),('Router',slice(0,split)),('TCG',slice(split,None))]:
        relative[name]=norm([x-y for x,y in zip(gs[sl],hg[sl])])/max(norm(gs[sl]),1e-30)
        assert relative[name]<1e-3,('AUXILIARY_GT_FLOW_GRADIENT_EQUIVALENCE',name,relative[name])
    assert max(abs(a['losses'][k]-b['losses'][k]) for k in a['losses'])<1e-5
    assert_reset(c);restore_rng(saved);frames=Frames(raw(row),c['device'])
    repeat,_=loss.basic_crop(c['ctx'],frames,row['external_qp'],raft=c['raft'],raft_transform=c['transform']);frames.close()
    repeat_rel=norm([x-y for x,y in zip(gs,repeat)])/max(norm(gs),1e-30)
    smoke=[]
    for step in range(1,4):
        if step==1:grad,diag=hg,b
        else:
            frames=pipe.next(row);grad,diag=optimized_loss(loss.basic_crop)(c['ctx'],frames,row['external_qp'],raft=c['raft'],raft_transform=c['transform']);frames.close()
        assert all(torch.isfinite(g).all() for g in grad)
        group_norms=dict(Router=norm(grad[:split]),TCG=norm(grad[split:]));assert all(v>0 for v in group_norms.values())
        named={n:float(g.norm()) for (n,p),g in zip(c['named'],grad)}
        if step==1:assert all(v==0 for n,v in named.items() if n.startswith('router.') and 'net.4.' not in n)
        if step>=2:assert all(v>0 for n,v in named.items() if n.startswith('router.'))
        c['optimizer'].zero_grad(set_to_none=True)
        for p,g in zip(c['params'],grad):p.grad=g
        torch.nn.utils.clip_grad_norm_(c['params'],1.,error_if_nonfinite=True);c['optimizer'].step();guard.check()
        assert_reset(c);smoke.append(dict(step=step,group_gradient_norms=group_norms,parameter_gradient_norms=named,losses=diag['losses'],frozen_unchanged=True))
    pipe.close();guard.full()
    for prefix in ['router.','generator.tcg_low.','generator.tcg_mid.','generator.tcg_out.']:
        assert any(not torch.equal(p.detach(),initial[n]) for n,p in c['named'] if n.startswith(prefix))
    previous=torch.rand(1,3,256,256,device=c['device']);current=torch.rand_like(previous,requires_grad=True)
    x,y=c['transform'](previous.detach(),current);value=c['raft'](x,y)[-1].abs().mean();motion_grad=torch.autograd.grad(value,current)[0]
    assert torch.isfinite(motion_grad).all() and motion_grad.norm()>0
    dump(ROOT/'audits/gradient_equivalence.json',dict(status='PASS',relative_L2=relative,reference_repeat_relative_L2=repeat_rel,tolerance=1e-3,
        maximum_loss_difference=max(abs(a['losses'][k]-b['losses'][k]) for k in a['losses']),loss_source_sha256=sha(ROOT/'loss.py'),
        reconstruction_VJP_scale=4096,unscale_count=1,smoke=smoke,temporary_joint_updates=3,formal_joint_updates=0,
        Router_and_all_TCG_layers_changed=True,only_allowed_parameters_changed=True,frozen_SHA256=guard.digest,
        motion_gradient_to_reconstruction=float(motion_grad.norm()),no_old_optimizer_loaded=True))

def compare():
    import torch
    d={n:torch.load(ROOT/f'large_artifacts/engineering/{n}.pt',map_location='cpu',weights_only=True) for n in ['reference','reference_repeat','zero','enabled']}
    r=d['reference'];assert torch.equal(r['output'],d['reference_repeat']['output']),'REFERENCE_REPEAT_NOT_STRICTLY_DETERMINISTIC'
    assert torch.equal(r['output'],d['zero']['output']),'M0_NOT_STRICT_MATCHED_BASELINE'
    for a,b in zip(r['records'],d['zero']['records']):assert all(a[k]==b[k] for k in ['codec_feature','reconstruction','TCG','feature_shape'])
    e=d['enabled'];diff=(e['output']-r['output']).abs();means=[a['M'] for a in e['records']]
    assert all(abs(x-.2)<1e-6 for x in means)
    assert all(a['codec_feature']==b['codec_feature'] for a,b in zip(r['records'],e['records']))
    dump(ROOT/'audits/matched_M0_equivalence.json',dict(status='PASS',independent_B1500_FP32_reference=True,strict_output_feature_and_TCG_cache_match=True,
        reference_repeat_maxabs=0.,enabled_M0_2_min=min(means),enabled_M0_2_max=max(means),enabled_initialization_claimed_identical=False,
        enabled_initial_reconstruction_maxabs=float(diff.max()),enabled_initial_reconstruction_meanabs=float(diff.mean()),
        codec_features_unchanged=True,samples=2,P_frames_each=63,quality_scan_performed=False,formal_updates=0))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True);ap.add_argument('--rank',type=int,default=0);a=ap.parse_args();env()
    if a.mode=='compare':return compare()
    if a.mode=='layers':return layer_and_state_checks(a.rank)
    if a.mode=='gradient':return gradient_probe(a.rank)
    if a.mode.startswith('rans_'):
        from model_context import backend
        from evaluate import point,videos
        backend(False,a.rank)
        baseline=a.mode=='rans_matched';mode='zero' if a.mode=='rans_zero' else 'enabled'
        v=next(v for v in videos('final') if v['sequence']=='Kimono1')
        p=point(v,5,None,'engineering_'+a.mode,a.rank,scoring=False,mode=mode,
            candidate='matched_baseline' if baseline else 'joint',reuse_input=False)
        dump(ROOT/f'audits/{a.mode}.json',p)
        return
    return forward_probe(a.mode,a.rank)
if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
