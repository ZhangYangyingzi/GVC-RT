"""Frozen GLC; exactly one optimizer update for each shared 64-frame sample."""
from preflight import *
import argparse,fcntl,random,csv,resource
from data import load,Frames
def atomic_torch(path,x):
 import torch
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(path)
def seed(n):
 import torch,numpy as np
 random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def rng():
 import torch,numpy as np
 return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore(x):
 import torch,numpy as np
 random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch']);torch.cuda.set_rng_state_all(x['cuda'])
def signature():
 return {str(p.relative_to(R)):sha(p) for p in [R/x for x in ('runtime.py','train.py','data.py','preflight.py','protocol.json','plans/shared.json')]}|{'GLC':sha(R/'audits/source_inventory.json')}
def norm(gr):
 import torch
 return float(sum(g.double().square().sum() for g in gr if g is not None).sqrt())
def gradients(im,pm,p,ql,row,beta=None):
 import torch
 from runtime import step,losses
 f=Frames(row,True);dpb=None;params=list(p.parameters());total={};gref=[torch.zeros_like(t) for t in params];grate=[torch.zeros_like(t) for t in params];trace=[]
 for pos in range(64):
  x=f.tensor(pos,'cuda:0')
  with torch.set_grad_enabled(pos>=49):
   e,out,dpb,pad=step(im,pm,x,dpb,row['q'],pos,p)
   if pos<49:continue
   values=losses(out,x,ql);values['rate']=e['bpp'].mean();ref=values['LPIPS']+.5*values['DISTS']+.1383589248954108*values['L1']
   if beta is None:
    a=torch.autograd.grad(ref/15,params,retain_graph=True,allow_unused=True);b=torch.autograd.grad(values['rate']/15,params,allow_unused=True)
    for i,(u,v) in enumerate(zip(a,b)):
     if u is not None:gref[i]+=u.detach()
     if v is not None:grate[i]+=v.detach()
   else:((ref+beta*values['rate'])/15).backward()
   for k,v in values.items():total[k]=total.get(k,0)+float(v)/15
   trace.append(dict(position=pos,q=row['q'],fa_idx=[0,1,0,2,0,2,0,2][pos%8],reset_ref_y=pos%32==1))
 f.close();return total,trace,gref,grate
def calibrate(gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch,numpy as np
 from runtime import models,quality,Sender
 im,pm=models('cuda:0');ql=quality('cuda:0');fixed=[state_hash(m.state_dict()) for m in (im,pm,*ql)];plan=load(R/'plans/shared.json')['updates'];selected=[]
 for d in ('ulong','uvg'):
  for q in load(R/'protocol.json')['q_grid']:selected.extend([r for r in plan if r['domain']==d and r['q']==q][:2])
 save('plans/calibration.json',dict(rows=selected,optimizer_updates=0,rule='first two training-plan windows per domain/q; both adapter parameter spaces',gradient_scale_target=.1))
 records=[]
 for i,r in enumerate(selected):
  for b in ('B_rgb_P','C_latent_P'):
   path=R/'calibration'/f'{i:02d}_{b}.json'
   if path.exists():records.append(load(path));continue
   seed(20261015);p=Sender(b).cuda();values,trace,a,c=gradients(im,pm,p,ql,r);na,nc=norm(a),norm(c)
   assert na>0 and nc>0 and all(torch.isfinite(g).all() for g in a+c)
   row=dict(index=i,branch=b,domain=r['domain'],q=r['q'],plan_step=r['step'],reference_gradient_norm=na,rate_gradient_norm=nc,ratio=na/nc,values=values,source_sha256=r['source_sha256'],RGB_sha256=r['cropped_RGB_sha256']);save(str(path.relative_to(R)),row);records.append(row);del p
  print('CALIBRATION',i+1,'/40',flush=True)
 beta={str(q):.1*float(np.median([r['ratio'] for r in records if r['q']==q])) for q in load(R/'protocol.json')['q_grid']}
 assert fixed==[state_hash(m.state_dict()) for m in (im,pm,*ql)]
 save('calibration/final.json',dict(status='PASS',beta=beta,optimizer_updates=0,training_windows=40,measurements=80,rule='.10 * median(||g_quality|| / ||g_official_rate||), pooled training domains and B/C, per q',frozen_unchanged=True,calibration_sha256=sha(R/'plans/calibration.json')))
def train(branch,gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch
 from runtime import models,quality,Sender
 assert load(R/'publication_status.json')['status']=='PUBLISHED';assert load(R/'audits/gradient.json')['status']=='PASS'
 assert load(R/'calibration/final.json')['status']=='PASS'
 root=R/'branches'/branch;root.mkdir(parents=True,exist_ok=True);lock=(root/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 im,pm=models('cuda:0');ql=quality('cuda:0');fixed=[state_hash(m.state_dict()) for m in (im,pm,*ql)];seed(20261015);p=Sender(branch).cuda().train();opt=torch.optim.AdamW(p.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=1e-4);params=list(p.parameters());sig=signature();sig['beta']=sha(R/'calibration/final.json');start=0
 latest=R/'checkpoints/resume'/branch/'latest.pt';index=load(root/'checkpoint_index.json') if (root/'checkpoint_index.json').exists() else {}
 if latest.exists():
  s=torch.load(latest,map_location='cpu',weights_only=False);assert s['signature']==sig and s['branch']==branch;p.load_state_dict(s['P'],strict=True);opt.load_state_dict(s['optimizer']);restore(s['rng']);start=s['step']
 def checkpoint(step,row=None):
  assert fixed==[state_hash(m.state_dict()) for m in (im,pm,*ql)]
  s=dict(branch=branch,step=step,P=p.state_dict(),optimizer=opt.state_dict(),rng=rng(),signature=sig,last_receipt=row,frozen_hashes=fixed);atomic_torch(latest,s)
  if step in (0,1,250,500,1000):
   path=R/'checkpoints/resume'/branch/f'step_{step}.pt';atomic_torch(path,s)
   deploy=R/('final_P' if step==1000 else 'checkpoints/P')/f'{branch}_{step}.pt';atomic_torch(deploy,dict(P=p.state_dict(),branch=branch,step=step,signature=sig));index[str(step)]=dict(path=str(path),sha256=sha(path),P_path=str(deploy),P_sha256=sha(deploy),module_sha256=state_hash(p.state_dict()));save(str((root/'checkpoint_index.json').relative_to(R)),index)
 if not start:checkpoint(0)
 elif s.get('last_receipt'):save(f'branches/{branch}/receipts/{start:04d}.json',s['last_receipt'])
 beta=load(R/'calibration/final.json')['beta'];torch.cuda.reset_peak_memory_stats()
 for step_no in range(start+1,1001):
  r=load(R/'plans/shared.json')['updates'][step_no-1];tic=time.time();opt.zero_grad(set_to_none=True);before=[t.detach().clone() for t in params]
  values,trace,_,_=gradients(im,pm,p,ql,r,beta[str(r['q'])]);gr=[t.grad for t in params];assert all(g is None or torch.isfinite(g).all() for g in gr) and norm(gr)>0
  nonzero=sum(g is not None and bool(g.count_nonzero()) for g in gr)
  if step_no>=2:assert nonzero>2,'upstream sender gradients persistently absent after output-layer update'
  pre=norm(gr);torch.nn.utils.clip_grad_norm_(params,1.)
  with (root/'optimizer_calls.jsonl').open('a') as j:j.write(json.dumps(dict(event='STARTED',step=step_no,time=time.time()))+'\n')
  opt.step();delta=norm([t.detach()-u for t,u in zip(params,before)]);assert delta>0 and all(torch.isfinite(t).all() for t in params)
  row=dict(r,branch=branch,loss=values,beta=beta[str(r['q'])],weighted_rate=beta[str(r['q'])]*values['rate'],total_loss=values['LPIPS']+.5*values['DISTS']+.1383589248954108*values['L1']+beta[str(r['q'])]*values['rate'],frame_trace=trace,gradient_preclip=pre,gradient_nonzero_parameter_count=nonzero,clip_coefficient=min(1,1/(pre+1e-6)),parameter_delta=delta,lr=5e-5,effective_updates=step_no,seconds=time.time()-tic,peak_MiB=torch.cuda.max_memory_reserved()/2**20,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
  checkpoint(step_no,row);save(f'branches/{branch}/receipts/{step_no:04d}.json',row);save(f'branches/{branch}/status.json',dict(status='COMPLETE' if step_no==1000 else 'RUNNING',updates=step_no,PID=os.getpid(),gpu=gpu,time=time.time()))
  if step_no%25==0:print('TRAIN',branch,step_no,flush=True)
 save(f'branches/{branch}/integrity.json',dict(status='PASS',updates=1000,supervised_frames=15000,frozen_unchanged=True,fresh_optimizer=True))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--calibrate',action='store_true');a.add_argument('--branch',choices=['B_rgb_P','C_latent_P']);a.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True);args=a.parse_args()
 try:calibrate(args.gpu) if args.calibrate else train(args.branch,args.gpu)
 except Exception:
  e=traceback.format_exc();save('BLOCKED.json',dict(status='BLOCKED',stage='calibration' if args.calibrate else 'training',branch=args.branch,error=e));print(e);sys.exit(1)
