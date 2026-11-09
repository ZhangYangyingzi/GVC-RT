"""Disposable implementation assertions. Formal states are freshly reloaded."""
from common_s2 import *
from experiment_model import construct,reconstruct,detach_history
from model_context import assert_reset,frozen_items
from cache_codec import cache_paths
import argparse,traceback

def tests(group,rank):
    import torch
    from temporal import checks
    env();row=plan_rows()[0];key,path,receipt=cache_paths(row,row['external_qp'],True);data=torch.load(path,map_location='cpu',weights_only=True)
    reference=load(ROOT/'audits/unmodified_A2000_reference.json');assert reference['codec_cache_identity']==key
    c=construct(group,False,rank);c['alignment_enabled']=False;gen=c['models']['generator'];hashes=[];cache_hash=[]
    before=tensor_hash(c['pm'].state_dict().items())
    with torch.no_grad():
        for i in range(1,64):
            out=reconstruct(c,data['features'][i].to(c['device']),data['info']['actual_qps'][i]);hashes.append(tensor_sha(out));cache_hash.append({n:tensor_sha(getattr(gen,n)._previous) for n in ['tcg_low','tcg_mid','tcg_out']})
    assert hashes==reference['reconstruction_hashes'],'A2000_MATCHED_PRECISION_STEP0_RECON_MISMATCH'
    assert cache_hash==reference['TCG_cache_hashes'],'STEP0_CACHE_SEMANTICS_CHANGED'
    assert before==tensor_hash(c['pm'].state_dict().items())
    snapshot={n:tensor_sha(getattr(gen,n)._previous) for n in ['tcg_low','tcg_mid','tcg_out']}
    with torch.no_grad():reconstruct(c,data['features'][63].to(c['device']),data['info']['actual_qps'][63],update=False)
    assert snapshot=={n:tensor_sha(getattr(gen,n)._previous) for n in snapshot},'UPDATE_BUFFER_FALSE_CHANGED_HISTORY'
    c['pm'].add_ref_frame(data['features'][63].to(c['device']),None)
    with torch.no_grad():c['pm'].prepare_feature_adaptor_i(data['info']['actual_qps'][63])
    assert snapshot=={n:tensor_sha(getattr(gen,n)._previous) for n in snapshot},'INTERNAL_REFERENCE_RECONSTRUCTION_POLLUTED_TCG_CACHE'
    shapes={n:list(getattr(gen,n)._previous.shape) for n in snapshot}
    assert_reset(c);del c,gen;torch.cuda.empty_cache()
    c=construct(group,True,rank);gen=c['models']['generator'];guard=FrozenGuard(frozen_items(c));checks_result=checks(c['device'])
    def alignment(i):return __import__('temporal').alignment_field(data,i,c['device']) if group in 'CD' else None
    witnesses={}
    for sever in [False,True]:
        assert_reset(c)
        with torch.no_grad():
            for i in range(1,49):reconstruct(c,data['features'][i].to(c['device']),data['info']['actual_qps'][i],alignment(i))
        first=reconstruct(c,data['features'][49].to(c['device']),data['info']['actual_qps'][49],alignment(49),retain=True)
        # tcg_low pre-residual input can have a frozen upstream in A/C. Test
        # an actually reachable cache, without demanding all caches require grad.
        cache=gen.tcg_out._previous;assert cache.requires_grad
        if sever:detach_history(c)
        later=reconstruct(c,data['features'][50].to(c['device']),data['info']['actual_qps'][50],alignment(50),retain=True)
        gradient=torch.autograd.grad(later.float().square().mean(),[cache]+c['params'],allow_unused=True)
        witness=0. if gradient[0] is None else float(gradient[0].float().norm())
        if sever:assert witness==0,'DETACH_CONTROL_DID_NOT_SEVER_CACHE_PATH'
        else:assert witness>0 and math.isfinite(witness),'CROSS_FRAME_CACHE_PATH_NOT_DIFFERENTIABLE'
        witnesses[str(sever)]=dict(previous_out_cache_requires_grad=True,later_loss_previous_cache_gradient=witness,reachable_parameter_gradient_norm=math.sqrt(sum(float(g.double().square().sum()) for g in gradient[1:] if g is not None)))
        detach_history(c);assert not gen.tcg_out._previous.requires_grad
        after_boundary=reconstruct(c,data['features'][53].to(c['device']),data['info']['actual_qps'][53],alignment(53),retain=True)
        past=torch.autograd.grad(after_boundary.float().sum(),cache,allow_unused=True)[0]
        assert past is None,'SEGMENT_BOUNDARY_RETAINED_OLD_GRAPH'
        assert_reset(c);del first,later,cache,gradient,after_boundary
    # Legacy RGB VJP scale is unscaled once. Verify on live trainable parameters.
    out=reconstruct(c,data['features'][1].to(c['device']),data['info']['actual_qps'][1],alignment(1),retain=True)
    gt=data['GT'][1].permute(2,0,1)[None].to(c['device']).float()/255
    rgb=(c['core'].unit(out)-gt).abs().mean();a=torch.autograd.grad(rgb*.1383589248954108,c['params'],retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(rgb*4096,c['params'],allow_unused=True)
    diff=math.sqrt(sum(float((x-y/4096*.1383589248954108).double().square().sum()) for x,y in zip(a,b) if x is not None and y is not None))
    den=math.sqrt(sum(float(x.double().square().sum()) for x in a if x is not None));assert diff/max(den,1e-20)<1e-3
    guard.full();assert len(c['optimizer'].state)==0;assert_reset(c)
    result=dict(status='PASS',group=group,step0_exact_reconstruction=True,step0_exact_cache=True,alignment_disabled_for_step0_comparison=group in 'CD',warp=checks_result,cross_frame_gradient=witnesses,
        segment_boundary_detach=True,update_buffer_false_unchanged=True,internal_reference_reconstruction_cache_unchanged=True,actual_TCG_cache_shapes=shapes,cache_position='pre-residual current',RGB_VJP4096_unscale_once_relative_L2=diff/max(den,1e-20),
        trainable_parameters=sum(p.numel() for p in c['params']),Bridge_type=c['models']['bridge'][3].__class__.__name__,frozen_SHA256=guard.digest,optimizer_updates=0,
        source_model='Original A2000 only',initial_A2000_SHA256=INIT_SHA,physical_GPU=physical_gpu(rank),time=time.time())
    dump(ROOT/f'audits/engineering_{group}.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',required=True,choices=GROUPS);p.add_argument('--rank',type=int,required=True);a=p.parse_args()
    try:tests(a.group,a.rank)
    except Exception as e:
        env();dump(ROOT/f'audits/engineering_{a.group}_failure.json',dict(status='FAILED',exception=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
