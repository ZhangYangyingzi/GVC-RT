#!/usr/bin/env python3
import argparse, csv, json, math, os, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
from wrapper_model import NeuralWrapper
from core import ROOT, load_models, load_train_pair, establish_dpb, ste_forward_p, quality_models, perceptual, model_hash, tensor_hash, csv_write


CHECKPOINTS={0,500,1000,2000,5000}
def save(path,wrapper,opt,step,label,beta,history,frozen_hash):
    torch.save({'schema':'gvcrt_neural_wrapper_v1_checkpoint','beta_label':label,'beta':beta,'step':step,'wrapper':wrapper.state_dict(),'optimizer':opt.state_dict(),'wrapper_hash':tensor_hash(wrapper),'frozen_model_hash':frozen_hash,'history_tail':history[-10:]},path)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gpu',type=int,required=True); ap.add_argument('--beta-label',choices=['beta_low','beta_mid','beta_high'],required=True); args=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu); seed=20260924; random.seed(seed); torch.manual_seed(seed); dev=torch.device('cuda:0')
    cfg=json.load(open(ROOT/'config.json')); beta=float(cfg['beta_weights'][args.beta_label]); tc=cfg['training']
    wrapper=NeuralWrapper().to(dev).float().train(); im,pm=load_models(dev); quality=quality_models(dev)
    frozen_before=model_hash(im,pm); opt=torch.optim.AdamW(wrapper.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
    outdir=ROOT/'checkpoints'/args.beta_label; outdir.mkdir(parents=True,exist_ok=True); history=[]; start_step=0
    candidates=sorted(outdir.glob('step_*.pt'))
    if candidates:
        checkpoint=torch.load(candidates[-1],map_location='cpu',weights_only=True)
        wrapper.load_state_dict(checkpoint['wrapper'],strict=True);opt.load_state_dict(checkpoint['optimizer']);start_step=int(checkpoint['step'])
        old=ROOT/'parts'/f'training_{args.beta_label}.csv'
        if old.exists():
            with open(old,newline='') as f: history=[r for r in csv.DictReader(f) if int(r['step'])<=start_step]
    else: save(outdir/'step_0000.pt',wrapper,opt,0,args.beta_label,beta,history,frozen_before)
    start=time.time()
    for step in range(start_step+1,tc['updates']+1):
        # Deterministic coprime traversal through the frozen 1500-step cache.
        cache_index=((step-1)*977 + 37) % 1500 + 1
        (previous,current),plan=load_train_pair(cache_index,dev); qp=(step-1)%4
        with torch.no_grad(): previous_proxy=wrapper(previous)
        establish_dpb(im,pm,previous_proxy,qp)
        opt.zero_grad(set_to_none=True); proxy=wrapper(current); output,rate,ste=ste_forward_p(pm,proxy,qp)
        d,lp,di=perceptual(output,current,quality); rnorm=rate/(current.shape[-2]*current.shape[-1]); l1=F.l1_loss(proxy,current)
        loss=d+beta*rnorm+float(tc['lambda_proxy'])*l1
        if not all(torch.isfinite(x) for x in (loss,d,rnorm,lp,di,l1)): raise RuntimeError(f'nonfinite step {step}')
        loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(wrapper.parameters(),tc['gradient_clip_norm']))
        if not math.isfinite(grad): raise RuntimeError(f'nonfinite grad {step}: {grad}')
        opt.step(); pnorm=math.sqrt(sum(float(p.detach().square().sum()) for p in wrapper.parameters()))
        row={'beta_label':args.beta_label,'beta':beta,'step':step,'cache_index':cache_index,'source_video_index':plan['video_index'],'requested_qp':qp,'loss':float(loss),'R_est_bpp':float(rnorm),'LPIPS':float(lp),'DISTS':float(di),'proxy_source_L1':float(l1),'gradient_norm':grad,'wrapper_parameter_norm':pnorm,'STE_forward_max_abs_diff':ste,'finite':True}
        history.append(row)
        if step%25==0 or step in CHECKPOINTS:
            print(f"{args.beta_label} {step}/5000 loss={float(loss):.6f} R={float(rnorm):.6f} LP={float(lp):.6f} DI={float(di):.6f} grad={grad:.6f}",flush=True)
        if step in CHECKPOINTS: save(outdir/f'step_{step:04d}.pt',wrapper,opt,step,args.beta_label,beta,history,frozen_before)
        if step%100==0: csv_write(ROOT/'parts'/f'training_{args.beta_label}.csv',history)
    save(outdir/'final.pt',wrapper,opt,5000,args.beta_label,beta,history,frozen_before)
    csv_write(ROOT/'parts'/f'training_{args.beta_label}.csv',history)
    result={'status':'PASS','beta_label':args.beta_label,'beta':beta,'steps':5000,'seconds':time.time()-start,'checkpoint_count':6,'frozen_hash_before':frozen_before,'frozen_hash_after':model_hash(im,pm),'frozen_unchanged':frozen_before==model_hash(im,pm),'wrapper_hash':tensor_hash(wrapper)}
    (ROOT/'parts'/f'train_{args.beta_label}_done.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
