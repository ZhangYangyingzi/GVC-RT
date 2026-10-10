import argparse,random,math,statistics,traceback
from io21 import *
from window_training import replay,make_runner
from gradient_tools import stats,dists_details

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args();os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);random.seed(20261023);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023);frozen();command([PYTHON,'-B',str(Path(__file__).resolve()),*sys.argv[1:]])
    v4,im,pm,models,quality=load_training(torch.device('cuda:0'));params=[p for m in models.values() for p in m.parameters()];initial={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm);qh=[v4.module_hash(m) for m in quality];rows=load(ROOT/'plans/calibration.json')['updates'];run=make_runner(v4,im,pm,models,quality)
    def clear():
        for p in params:p.grad=None
    def copyg():return [None if p.grad is None else p.grad.detach().cpu().clone() for p in params]
    def match(g):
        for p,h in zip(params,g):
            assert (p.grad is None)==(h is None)
            if h is not None:assert torch.allclose(p.grad.cpu(),h,atol=2e-6,rtol=2e-4)
    def summary(gs):
        out={};offset=0
        for k,m in models.items():
            ps=list(m.parameters());n=len(ps);out[k]=stats(gs[offset:offset+n],ps);offset+=n
        out['all']=stats(gs,params);return out
    r=rows[0];frames=replay(r,BRANCHES[0],torch.device('cuda:0'));q=r['external_qp']
    oldns=dict(globals(),BRANCHES=('A','B','C'));exec((V21/'window_training.py').read_text().replace('from io21 import *',''),oldns);old=oldns['make_runner'](v4,im,pm,models,quality)
    clear();ol,oq,ot,ow,_=old(frames,q,'B');og=copyg()
    clear();al,aq,at,aw,_=run(frames,q,BRANCHES[0],rec_weight=0);match(og)
    assert aq==oq and aw==ow and all(x['reconstruction_sha256']==y['reconstruction_sha256'] for x,y in zip(at,ot))
    assert all(abs(al[k]-ol[k])<2e-6+1e-5*abs(ol[k]) for k in al if k in ol)
    ag=copyg();clear();bl,bq,bt,bw,_=run(frames,q,BRANCHES[1],rec_weight=0);match(ag);assert al==bl and at==bt and aw==bw
    clear();loss1,_,_,_,cg=run(frames,q,BRANCHES[0],rec_weight=0,mode='calibrate',rec_scale=4096.);high=cg['rec'];hs=summary(high)
    assert all(v['finite'] and v['effective_parameters']>0 for v in hs.values())
    loss2,_,_,_,low=run(frames,q,BRANCHES[0],rec_weight=0,mode='calibrate',rec_scale=256.);assert all(loss1[k]==loss2[k] for k in loss1 if k not in ('reconstruction_backward_scale','reconstruction_unit_gradient_scale'))
    differences={};offset=0
    for k,m in models.items():
        n=len(list(m.parameters()));aa=high[offset:offset+n];bb=low['rec'][offset:offset+n];offset+=n
        norm=sum(float(x.double().square().sum()) for x in aa)**.5;err=sum(float((x.double()-y.double()).square().sum()) for x,y in zip(aa,bb))**.5
        cosine=sum(float((x.double()*y.double()).sum()) for x,y in zip(aa,bb))/(norm*sum(float(y.double().square().sum()) for y in bb)**.5)
        differences[k]=dict(relative_L2=err/norm,cosine=cosine);assert err/norm<.25 and cosine>.97
    x=torch.rand_like(frames[0],requires_grad=True);y=torch.rand_like(x);_,layers,_=dists_details(quality,x,y,True);assert all(t['output_gradient']['finite'] and t['output_gradient']['effective_parameters']>0 for t in layers)
    dump(ROOT/'audits/reconstruction_backward_scaling.json',dict(status='PASS',scale=4096.,compared_scales=[256.,4096.],restored_gradients=hs,differences=differences,parameters_dtype='torch.float32',grad_buffers_dtype='torch.float32',independent_FP32_unscale=True,base_gradients_not_rescaled=True,loss_weight_not_changed=True,forward_unchanged=True,rec_not_detached=True))
    dump(ROOT/'audits/implementation_check.json',dict(status='PASS',initial_modules=initial,compression_hash=core,quality_hashes=qh,source_B_path_matched=True,branch_initial_weights_equal=True,branch_reference_states_equal=True,zero_lambda_loss_gradient_equal=True,reconstruction_gradients=hs,DISTS_layers=layers,optimizer_updates=0,formal_plan_consumed=0,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20))
    del high,low,cg,frames,ag,og;clear();records=[]
    for row in rows:
        dest=ROOT/'calibration/samples'/f"sample_{row['adaptation_step']:02d}.json"
        if dest.exists():record=load(dest);assert record['sample']==row and record['source_sha256']==EXPECTED
        else:
            frames=replay(row,BRANCHES[0],torch.device('cuda:0'));clear();loss,qs,trace,warmup,gs=run(frames,row['external_qp'],BRANCHES[0],rec_weight=0,mode='calibrate');assert qs==row['actual_qps'] and len(warmup)==48
            norms={k:summary(g) for k,g in gs.items()};assert all(v['finite'] and v['effective_parameters']>0 for g in norms.values() for v in g.values())
            ratio=.05*norms['perc']['all']['norm']/(norms['rec']['all']['norm']+1e-12);assert math.isfinite(ratio) and ratio>0
            record=dict(sample=row,losses=loss,gradient_norms=norms,candidate_lambda_rec=ratio,source_sha256=EXPECTED,scale=4096.,optimizer_updates=0);dump(dest,record);del gs,frames
        records.append(record);dump(ROOT/'calibration/status.json',dict(status='RUNNING',completed=len(records),expected=40));print('CALIBRATION',len(records),'/40',flush=True)
    value=statistics.median(r['candidate_lambda_rec'] for r in records);dump(ROOT/'calibration/lambda_rec.json',dict(lambda_rec=value,formula='median(0.05*norm(mean15 perceptual gradients)/(norm(mean15 reconstruction_L1 gradients)+1e-12))',samples=40,seed=20261023,optimizer_updates=0,source_sha256=EXPECTED,plan_sha256=sha(ROOT/'plans/calibration.json')));write(ROOT/'calibration/calibration_samples.csv',[dict(domain=r['sample']['domain'],video=r['sample']['video'],QP=r['sample']['external_qp'],sample=r['sample']['adaptation_step'],**r['losses'],candidate_lambda_rec=r['candidate_lambda_rec'],gradient_norms=r['gradient_norms']) for r in records])
    assert initial=={k:v4.module_hash(m) for k,m in models.items()} and core==v4.compression_hash(im,pm) and qh==[v4.module_hash(m) for m in quality]
    # Weighted reconstruction contribution and base remain separate, then combine in FP32.
    frames=replay(rows[0],BRANCHES[0],torch.device('cuda:0'));clear();base,_,_,_,_=run(frames,q,BRANCHES[0],rec_weight=0);bg=copyg();clear();_,_,_,_,gg=run(frames,q,BRANCHES[1],rec_weight=0,mode='calibrate');rg=[g.detach().cpu() for g in gg['rec']];del gg
    weighted_summary=summary([g.to(params[i].device)*value for i,g in enumerate(rg)]);assert all(x['finite'] and x['effective_parameters']>0 for x in weighted_summary.values())
    dump(ROOT/'audits/weighted_reconstruction_gradients.json',dict(status='PASS',lambda_rec=value,weighted_loss_scale=4096./value,unit_gradient_scale=4096.,gradients=weighted_summary,FP32_restore_then_weight=True))
    clear();_,_,_,_,_=run(frames,q,BRANCHES[1],rec_weight=value)
    for p,b,g in zip(params,bg,rg):assert torch.allclose(p.grad.cpu(),b+value*g,rtol=2e-4,atol=2e-7)
    assert all(p.grad.dtype==torch.float32 and torch.isfinite(p.grad).all() for p in params)
    checkpoint=torch.load(load(ROOT/'config.json')['source_checkpoint']['path'],map_location='cpu',weights_only=True)
    for k,m in models.items():m.load_state_dict(checkpoint[k],strict=True)
    del checkpoint
    clear();pm.clear_dpb();pm.set_curr_poc(0);random.seed(20261022);torch.manual_seed(20261022);torch.cuda.manual_seed_all(20261022)
    assert initial=={k:v4.module_hash(m) for k,m in models.items()}
    cfg=load(ROOT/'config.json');cfg['lambda_rec']=value;dump(ROOT/'config.json',cfg);proto=load(ROOT/'protocol.json');proto['training']['lambda_rec']=value;proto['training']['reconstruction_unit_gradient_scale']=4096.;proto['training']['weighted_reconstruction_backward_scale']=4096./value;proto['training']['gradient_formula']='g_base + lambda_rec * grad(4096 * mean15 reconstruction_L1) / 4096, FP32 restoration and weighting';dump(ROOT/'protocol.json',proto)
    hashes=load(ROOT/'audits/protocol_hashes.json');hashes.update({n:sha(ROOT/n) for n in ('config.json','protocol.json')});dump(ROOT/'audits/protocol_hashes.json',hashes)
    dump(ROOT/'audits/calibration_integrity.json',dict(status='PASS',samples=40,lambda_rec=value,optimizer_updates=0,models_unchanged=True,quality_frozen=True,compression_frozen=True,formal_plan_consumed=0,training_reloads_initial_weights_and_resets_RNG=True,reloaded_initial_parameters=True,RNG_reset_seed=20261022,seed=20261023,plan_sha256=sha(ROOT/'plans/calibration.json'),gradient_scale=4096.,weighted_loss_effective_scale=4096./value,merged_gradient_verified=True))
    if (ROOT/'audits/calibration_failure.json').exists():dump(ROOT/'audits/calibration_failure.json',dict(status='RESOLVED',log='logs/calibration.log',resolution='Compare forward loss components separately from backward-scale metadata'))
    dump(ROOT/'audits/executed_source_hashes.json',{str(p.resolve()):sha(p) for p in ROOT.glob('*.py')})
    dump(ROOT/'calibration/status.json',dict(status='PASS',completed=40,expected=40,lambda_rec=value));print('CALIBRATION PASS',value,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'audits/calibration_failure.json',dict(status='FAIL',error=traceback.format_exc()));raise
