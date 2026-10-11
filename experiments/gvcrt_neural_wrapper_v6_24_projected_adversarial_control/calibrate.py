"""Training-only, forty-clip joint gradient calibration; zero optimizer updates."""
import argparse,random,statistics,math,traceback
from io21 import *
from window_training import replay,make_runner
from gradient_tools import stats,dists_details

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args()
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(a.gpu))
    import torch
    from projected import create
    torch.set_num_threads(2);torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025);random.seed(20261025);frozen()
    v4,im,pm,models,quality=load_training(torch.device('cuda:0'));params=[p for m in models.values() for p in m.parameters()]
    disc=create(torch.device('cuda:0'));disc.g_mode();fhash=tensor_hash(disc.feature.state_dict());dhash=tensor_hash(disc.discriminator.state_dict())
    initial={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm);qh=[v4.module_hash(m) for m in quality]
    run=make_runner(v4,im,pm,models,quality,disc);arun=make_runner(v4,im,pm,models,quality)
    rows=load(ROOT/'plans/calibration.json')['updates']
    def clear():
        for p in params:p.grad=None
    def summary(gs):
        out={};off=0
        for k,m in models.items():
            ps=list(m.parameters());out[k]=stats(gs[off:off+len(ps)],ps);off+=len(ps)
        out['all']=stats(gs,params);return out
    frames=replay(rows[0],BRANCHES[0],torch.device('cuda:0'));q=rows[0]['external_qp']
    clear();al,aq,at,aw,_=arun(frames,q,BRANCHES[0]);ag=[p.grad.detach().cpu().clone() for p in params]
    clear();bl,bq,bt,bw,_=run(frames,q,BRANCHES[1],adv_weight=0)
    ignored={'projected_adv','weighted_projected_adv'}
    assert all(al[k]==bl[k] for k in al if k not in ignored) and aq==bq and aw==bw
    assert all({k:v for k,v in x.items() if k not in ignored}=={k:v for k,v in y.items() if k not in ignored} for x,y in zip(at,bt))
    for p,g in zip(params,ag):assert torch.allclose(p.grad.cpu(),g,atol=2e-7,rtol=2e-4)
    assert fhash==tensor_hash(disc.feature.state_dict()) and dhash==tensor_hash(disc.discriminator.state_dict())
    del ag;clear()
    _,_,_,_,gg=run(frames,q,BRANCHES[1],mode='calibrate',adv_weight=0,adv_scale=4096.)
    high=gg['adv'];hs=summary(high);assert all(x['finite'] and x['norm']>0 for x in hs.values())
    _,_,_,_,gg=run(frames,q,BRANCHES[1],mode='calibrate',adv_weight=0,adv_scale=256.)
    differences={};off=0
    for k,m in models.items():
        n=len(list(m.parameters()));x=high[off:off+n];y=gg['adv'][off:off+n];off+=n
        norm=sum(float(g.double().square().sum()) for g in x)**.5;err=sum(float((g.double()-h.double()).square().sum()) for g,h in zip(x,y))**.5
        cosine=sum(float((g.double()*h.double()).sum()) for g,h in zip(x,y))/(norm*sum(float(h.double().square().sum()) for h in y)**.5)
        differences[k]=dict(relative_L2=err/norm,cosine=cosine);assert math.isfinite(cosine) and cosine>.95 and err/norm<.3
    del gg,high;clear()
    z=frames[49].detach().clone().requires_grad_(True);value=disc.adversarial(z);dz=torch.autograd.grad(value,z)[0];assert torch.isfinite(dz).all() and dz.abs().max()>0
    x=torch.rand_like(z,requires_grad=True);y=torch.rand_like(z);_,layers,_=dists_details(quality,x,y,True);assert all(t['output_gradient']['finite'] and t['output_gradient']['effective_parameters']>0 for t in layers)
    disc.d_mode();opt=torch.optim.Adam(disc.discriminator.parameters(),lr=2e-4,betas=(0.,.9));opt.zero_grad();dl=disc.d_loss(frames[49],z.detach());dl.backward();assert all(p.grad is None for p in params);opt.step();assert initial=={k:v4.module_hash(m) for k,m in models.items()} and fhash==tensor_hash(disc.feature.state_dict())
    # Discard diagnostic discriminator, optimizer and all diagnostic RNG.
    del disc,opt;disc=create(torch.device('cuda:0'));disc.g_mode();assert dhash==tensor_hash(disc.discriminator.state_dict());run=make_runner(v4,im,pm,models,quality,disc)
    dump(ROOT/'audits/implementation_check.json',dict(status='PASS',initial_modules=initial,compression_hash=core,feature_hash=fhash,projection_and_backbone_eval_frozen=True,G_discriminator_buffers_unchanged=True,D_PBG_unchanged=True,zero_lambda_adv_loss_gradient_equal=True,adversarial_gradients=hs,adversarial_rec_gradient_norm=float(dz.norm()),adversarial_scale=4096.,scale_stability=differences,DISTS_layers=layers,optimizer_updates=0,formal_plan_consumed=0,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20))
    records=[]
    for row in rows:
        p=ROOT/'calibration/samples'/f"sample_{row['adaptation_step']:02d}.json"
        if p.exists():record=load(p);assert record['sample']==row and record['source_sha256']==EXPECTED
        else:
            clear();frames=replay(row,BRANCHES[0],torch.device('cuda:0'));loss,qs,tr,warm,gg=run(frames,row['external_qp'],BRANCHES[1],mode='calibrate',adv_weight=0)
            assert qs==row['actual_qps'];norms={k:summary(v) for k,v in gg.items()};assert all(x['finite'] and x['norm']>0 for ns in norms.values() for x in ns.values())
            ratio=norms['base']['all']['norm']/norms['adv']['all']['norm'];assert math.isfinite(ratio) and ratio>0
            record=dict(sample=row,source_sha256=EXPECTED,losses=loss,gradient_norms=norms,norm_ratio=ratio,optimizer_updates=0,adversarial_unit_scale=4096.,reconstruction_unit_scale=4096.)
            dump(p,record);del gg,frames
        records.append(record);dump(ROOT/'calibration/status.json',dict(status='RUNNING',completed=len(records),expected=40));print('CALIBRATION',len(records),flush=True)
    value=.05*statistics.median(r['norm_ratio'] for r in records)
    dump(ROOT/'calibration/lambda_adv.json',dict(lambda_adv=value,formula='0.05 * median(norm(mean15 complete B1 gradients)/norm(mean15 unweighted projected adversarial gradients))',valid_samples=40,expected_samples=40,source_sha256=EXPECTED,optimizer_updates=0,seed=20261025))
    dump(ROOT/'calibration/lambda_rec.json',dict(lambda_rec=.1383589248954108,source='fixed v6.22 B1, not recalibrated'))
    write(ROOT/'calibration/calibration_samples.csv',[dict(sample=r['sample']['adaptation_step'],domain=r['sample']['domain'],video=r['sample']['video'],QP=r['sample']['external_qp'],**r['losses'],norm_ratio=r['norm_ratio'],gradient_norms=r['gradient_norms']) for r in records])
    assert initial=={k:v4.module_hash(m) for k,m in models.items()} and core==v4.compression_hash(im,pm) and qh==[v4.module_hash(m) for m in quality]
    assert fhash==tensor_hash(disc.feature.state_dict()) and dhash==tensor_hash(disc.discriminator.state_dict())
    cfg=load(ROOT/'config.json');cfg['lambda_adv']=value;dump(ROOT/'config.json',cfg)
    proto=load(ROOT/'protocol.json');proto['training'].update(lambda_adv=value,lambda_rec=.1383589248954108,reconstruction_unit_scale=4096.,adversarial_unit_scale=4096.,gradient_formula='g_base + lambda_rec*grad(4096*mean15 reconstruction_L1)/4096 + lambda_adv*grad(4096*mean15 projected_adv)/4096; independent FP32 restore and weighting');dump(ROOT/'protocol.json',proto)
    hashes=load(ROOT/'audits/protocol_hashes.json');hashes.update({n:sha(ROOT/n) for n in ('config.json','protocol.json')});dump(ROOT/'audits/protocol_hashes.json',hashes)
    clear();pm.clear_dpb();pm.set_curr_poc(0);random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    dump(ROOT/'audits/calibration_integrity.json',dict(status='PASS',samples=40,lambda_adv=value,lambda_rec=.1383589248954108,models_unchanged=True,backbone_projection_buffers_unchanged=True,optimizer_updates=0,formal_plan_consumed=0,training_reloads_initial_weights_optimizer_and_RNG=True,adversarial_unit_scale=4096.,effective_weighted_scale=4096./value,FP32_restore_and_accumulate=True))
    dump(ROOT/'calibration/status.json',dict(status='PASS',completed=40,expected=40,lambda_adv=value));print('CALIBRATION PASS',value,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'audits/calibration_failure.json',dict(status='FAIL',error=traceback.format_exc()));raise
