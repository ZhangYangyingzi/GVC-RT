"""One common fifty-update D warmup; equal-architecture uncut gradient calibration."""
from common_s2 import *
import torch,statistics
from experiment_model import construct,reconstruct,trainable_state
from model_context import assert_reset
from cache_codec import cache_paths
from loss import targets,backward_window,norm
from gan import discriminator,d_mode,d_loss,g_mode


def data(row):
    _,p,r=cache_paths(row,row['external_qp'],True);assert sha(p)==load(r)['cache_sha256']
    return torch.load(p,map_location='cpu',weights_only=True,mmap=True)


def main():
    env();assert load(ROOT/'audits/preflight.json')['status']=='PASS'
    rows=load(ROOT/'plans/calibration.json.gz');device=torch.device('cuda:0')
    seed(TRAIN_SEED+1);d,opt=discriminator(TRAIN_SEED+1,device);initial=tensor_hash(d.state_dict().items())
    # Only immutable B300 outputs for D pretraining, never mutable candidate outputs.
    c=construct('B300',False,0,losses=False)
    for ix,row in enumerate(rows):
        dest=ROOT/'cache/D_warmup'/f'{row["sample_key"]}.pt'
        if dest.exists():continue
        x=data(row);assert_reset(c);frames=[]
        with torch.no_grad():
            for i in range(1,x['info']['frames']):
                fake=c['core'].unit(reconstruct(c,x['features'][i],x['info']['actual_qps'][i]))
                if i in targets(x):frames.append((fake.cpu(),x['GT'][i].permute(2,0,1)[None].float()/255))
        atomic_torch(dest,dict(frames=frames,B300_SHA256=B300_SHA,cache_receipt=load(cache_paths(row,row['external_qp'],True)[2])))
        dump(ROOT/'calibration_progress.json',dict(stage='fixed_B300_D_warmup_fakes',windows=ix+1,total=24,time=time.time()))
    del c;torch.cuda.empty_cache()
    start=0;warm=ROOT/'checkpoints/D_warmup_latest.pt'
    if warm.exists():
        saved=torch.load(warm,map_location='cpu',weights_only=True);d.load_state_dict(saved['D']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['RNG']);start=saved['step']
    for step in range(start+1,51):
        d_mode(d);opt.zero_grad(set_to_none=True);loss=0.
        batch=[rows[(4*(step-1)+k)%24] for k in range(4)]
        for row in batch:
            pairs=torch.load(ROOT/'cache/D_warmup'/f'{row["sample_key"]}.pt',map_location='cpu',weights_only=True)['frames']
            for fake,real in pairs:
                value,_,_=d_loss(d,real.to(device),fake.to(device));value=value/(4*len(pairs));value.backward();loss+=float(value.detach())
        assert all(torch.isfinite(p.grad).all() for p in d.parameters())
        opt.step();atomic_torch(warm,dict(step=step,D=d.state_dict(),optimizer=opt.state_dict(),RNG=rng_state(),initial_SHA256=initial))
        dump(ROOT/'calibration_progress.json',dict(stage='D_warmup',committed_D_updates=step,loss=loss,time=time.time()))
    warm_D=tensor_hash(d.state_dict().items());g_mode(d);ratios=[];grads=[]
    # Same trained D with fixed spectral-normalization buffers for both architectures.
    for group in ['J0','J2']:
        c=construct(group,True,0);baseline=tensor_hash(c['named'])
        for ix,row in enumerate(rows):
            dest=ROOT/'audits/calibration_gradients'/f'{group}_{ix:02d}.json'
            if dest.exists():
                a=load(dest);assert a['D_SHA256']==warm_D and a['G_initial_SHA256']==baseline
                grads.append(a);ratios.append(a['ratio']);continue
            x=data(row);values={}
            for mode in ['base','adv']:
                c['optimizer'].zero_grad(set_to_none=True)
                backward_window(c,x,batch=1,d=d,mode=mode)
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in c['params'])
                value=norm([p.grad for p in c['params']]);assert value>0 and math.isfinite(value)
                values[mode]=value
            ratio=values['base']/(values['adv']+1e-12)
            a=dict(group=group,window=row['sample_key'],QP=row['external_qp'],domain=row['domain'],g_base_FP32_unclipped=values['base'],g_adv_FP32_unclipped=values['adv'],ratio=ratio,D_SHA256=warm_D,G_initial_SHA256=baseline)
            dump(dest,a);grads.append(a);ratios.append(ratio)
            dump(ROOT/'calibration_progress.json',dict(stage='G_gradient_calibration',group=group,windows=ix+1,total=24,time=time.time()))
        assert tensor_hash(c['named'])==baseline and len(c['optimizer'].state)==0
        del c;torch.cuda.empty_cache()
    assert tensor_hash(d.state_dict().items())==warm_D
    coefficient=.1*statistics.median(ratios);assert math.isfinite(coefficient) and coefficient>0
    atomic_torch(ROOT/'checkpoints/common_D_initial.pt',dict(D=d.state_dict(),D_optimizer=opt.state_dict(),D_RNG=rng_state(),warmup_updates=50,initial_SHA256=initial,warm_D_SHA256=warm_D))
    dump(ROOT/'gradient_calibration.json',dict(status='PASS',lambda_adv=coefficient,rule='.1*median(g_base/(g_adv+1e-12))',independent_windows=24,architectures=2,architecture_equal_weight=True,gradient_records=grads,D_warmup_updates=50,D_initial_SHA256=initial,D_post_warmup_SHA256=warm_D,D_state_SHA256=sha(ROOT/'checkpoints/common_D_initial.pt'),G_optimizer_updates=0,fresh_G_reloaded_for_training=True,time=time.time()))
    freeze_sources()


if __name__=='__main__':main()
