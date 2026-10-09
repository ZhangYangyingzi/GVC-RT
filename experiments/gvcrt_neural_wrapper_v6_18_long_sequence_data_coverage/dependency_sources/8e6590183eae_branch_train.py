import argparse,fcntl,math,random,time
from v61_io import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--branch',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
 import torch,torch.nn.functional as F
 lock=(ROOT/'branches'/a.branch/'parts/train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 br=ROOT/'branches'/a.branch;cfg=load(br/'config.json');v4=module('v62_v4train',V4/'train.py');dev=torch.device('cuda:0');assert torch.cuda.device_count()==1
 wrapper=v4.NeuralWrapper().to(dev).float().train();im,pm=v4.load_models(dev,force_zero_thres=.12);im.requires_grad_(False);pm.requires_grad_(False)
 bridge=pm.recon_generation_net.mlp.float().train().requires_grad_(True);gen=pm.recon_generation_net.decoder.float().train().requires_grad_(True);models={'wrapper':wrapper,'bridge':bridge,'generator':gen}
 cp=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=True);assert cp['step']==20000
 [models[k].load_state_dict(cp[k],strict=True) for k in models];init={k:v4.module_hash(m) for k,m in models.items()};comp=v4.compression_hash(im,pm)
 oc=cfg['optimizer'];opt=torch.optim.AdamW([dict(params=list(m.parameters()),lr=oc[k+'_lr'],name=k) for k,m in models.items()],weight_decay=oc['weight_decay'])
 assert all(len(s)==0 for s in opt.state.values());rng=random.Random(20260928);records=load(cfg['train_manifest'])['videos'];quality=v4.quality_models(dev);hist={q:0 for q in range(10)};log=br/'training_log.jsonl';idx=[0,1,0,2,0,2,0,2];steps=cfg['updates'];
 def save(step):
  p=br/'checkpoints'/f'step_{step:04d}.pt';payload=dict(schema='gvcrt_v6_2',branch=a.branch,beta=cfg['beta'],step=step,wrapper=wrapper.state_dict(),bridge=bridge.state_dict(),generator=gen.state_dict(),optimizer=opt.state_dict(),initialization_hashes=init,compression_hash=comp,qp_histogram=hist,external_qps=list(range(10)),force_zero_thres=.12);torch.save(payload,p);dump(br/'checkpoint_hashes.json',dict((str(s),dict(path=str(br/'checkpoints'/f'step_{s:04d}.pt'),sha256=sha(br/'checkpoints'/f'step_{s:04d}.pt'),step=s)) for s in cfg['checkpoint_steps'] if (br/'checkpoints'/f'step_{s:04d}.pt').exists()))
 save(0);f=log.open('w')
 for step in range(1,steps+1):
  frames,plan=v4.sample_clip(records,rng,dev);q=rng.randrange(10);hist[q]+=1;pm.clear_dpb();pm.set_curr_poc(0)
  with torch.no_grad():pm.add_ref_frame(None,im.compress(v4.codec_input(wrapper(frames[0])),q)['x_hat'])
  opt.zero_grad(set_to_none=True);capt=[];hook=pm.dec.register_forward_hook(lambda _m,_a,o:capt.append(o));sums={k:0. for k in ('loss','LPIPS','DISTS','rate_bpp','proxy_L1','beta_rate','lambda_proxy')}
  try:
   for fi,frame in enumerate(frames[1:],1):
    aq=pm.shift_qp(q,idx[fi%8]);proxy=wrapper(frame);rec,rate,_=v4.joint_ste_forward(pm,proxy,aq);dist,lp,ds=v4.perceptual(rec,frame,quality);rb=rate/(frame.shape[-2]*frame.shape[-1]);pl=F.l1_loss(proxy,frame);loss=(dist+cfg['beta']*rb+cfg['lambda_proxy']*pl)/3;assert torch.isfinite(loss);loss.backward();sums['loss']+=float(loss);sums['LPIPS']+=float(lp)/3;sums['DISTS']+=float(ds)/3;sums['rate_bpp']+=float(rb)/3;sums['proxy_L1']+=float(pl)/3;sums['beta_rate']+=float(cfg['beta']*rb)/3;sums['lambda_proxy']+=float(cfg['lambda_proxy']*pl)/3
    with torch.no_grad():pm.add_ref_frame(capt[-1].detach(),(rec.detach()*2-1).half())
  finally:hook.remove()
  norm=float(torch.nn.utils.clip_grad_norm_([p for m in models.values() for p in m.parameters()],1.0));assert math.isfinite(norm);opt.step();r=dict(beta=cfg['beta'],update=step,qp=q,**plan,**sums,gradient_norm=norm,all_finite=True);f.write(json.dumps(r)+'\n')
  if step%100==0:dump(br/'training_status.json',dict(status='RUNNING',update=step,target=steps,qp_histogram=hist));write(br/'training_qp_histogram.csv',[dict(qp=q,count=hist[q],fraction=hist[q]/step) for q in range(10)]);print(a.branch,step,flush=True)
  if step in cfg['checkpoint_steps']:save(step)
 f.close();dump(br/'training_status.json',dict(status='PASS',update=steps,target=steps,qp_histogram=hist));dump(br/'optimizer_audit.json',dict(status='PASS',fresh_optimizer=True,optimizer_state_restored=False,updates=steps,initialization_hashes=init,compression_hash=comp));print('TRAIN PASS',a.branch)
if __name__=='__main__':main()
