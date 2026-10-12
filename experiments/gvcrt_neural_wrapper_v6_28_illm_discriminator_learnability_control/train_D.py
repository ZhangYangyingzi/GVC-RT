from runtime import *
import argparse,traceback

def save_state(d,opt,branch,stage,step,last,frequency):
 base=ROOT/'branches'/branch/stage;cfg=load(ROOT/'config.json');s=dict(discriminator=d.state_dict(),optimizer=opt.state_dict(),rng=state_rng(),branch=branch,stage=stage,effective_updates=step,cycle=step//1000 if stage=='MAIN' else None,cursor=step%1000 if stage=='MAIN' else None,config_sha256=sha(ROOT/'config.json'),protocol_sha256=sha(ROOT/'protocol.json'),shared_plan_sha256=sha(ROOT/'plans/shared.json'),D_source_checkpoint_sha256=cfg['D_source_checkpoint']['sha256'],B1_sha256=EXPECTED,labeler_sha256=cfg['labeler_sha256'],last_cache_sha256=last,training_frequency=frequency,FP32_D=True,G_updates=0)
 p=base/'checkpoints'/f'D_step_{step:05d}.pt';atomic_torch(p,s);item=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,D_hash=tensor_hash(d.state_dict()),effective_updates=step,stage=stage,branch=branch);dump(base/'resume.json',item)
 if step in cfg['stages'][stage]['checkpoints']:
  idx=load(base/'checkpoint_index.json') if (base/'checkpoint_index.json').exists() else {};idx[str(step)]=item;dump(base/'checkpoint_index.json',idx)
 return item

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--branch',choices=BRANCHES28,required=True);p.add_argument('--stage',choices=('TINY','MAIN'),required=True);a=p.parse_args();device=init(a.gpu);base=ROOT/'branches'/a.branch/a.stage;base.mkdir(parents=True,exist_ok=True);lk=(base/'training.lock').open('a');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert load(ROOT/'audits/preflight.json')['status']=='PASS';cfg=load(ROOT/'config.json');budget=cfg['stages'][a.stage]['updates'];d=createD(device);opt=optfor(d,a.branch);step=0;freq=torch.zeros(1024,dtype=torch.int64);last=None;log=base/'training.jsonl'
 if (base/'resume.json').exists():
  item=load(base/'resume.json');assert sha(item['path'])==item['sha256'];s=torch.load(item['path'],map_location='cpu',weights_only=True);assert s['config_sha256']==sha(ROOT/'config.json') and s['protocol_sha256']==sha(ROOT/'protocol.json');d.load_state_dict(s['discriminator'],strict=True);opt.load_state_dict(s['optimizer']);restore_rng(s['rng']);step=s['effective_updates'];freq=s['training_frequency'];last=s['last_cache_sha256'];del s
  if log.exists():
   lines=[z for z in log.read_text().splitlines() if json.loads(z)['effective_step']<=step];log.write_text(''.join(z+'\n' for z in lines))
 else:save_state(d,opt,a.branch,a.stage,0,None,freq)
 allrows=load(ROOT/'plans/shared.json')['updates'];starttime=time.time();torch.cuda.reset_peak_memory_stats();failures=0
 while step<budget:
  if a.stage=='TINY':
   needed=[w['row_id'] for w in load(ROOT/'plans/tiny.json')['windows']]
  else:needed=[rowkey(allrows[step%1000])]
  while not all((ROOT/'cache/status'/f'{k}.json').exists() for k in needed):
   dump(base/'status.json',dict(status='WAITING_CACHE',effective_updates=step,needed=needed,PID=os.getpid(),gpu=a.gpu));time.sleep(3)
  pp=tiny_pairs() if a.stage=='TINY' else pairs(safe_cache(needed[0]));last=sha(ROOT/'cache/status'/f'{needed[-1]}.json');before={n:p.detach().clone() for n,p in d.named_parameters()};t=time.time()
  try:out=Dstep(d,opt,pp,a.branch,device)
  except Exception:
   failures+=1;dump(base/'failure.json',dict(status='FAIL',step=step+1,needed=needed,traceback=traceback.format_exc(),failure_count=failures));raise
  update={name:sum(float((p.detach()-before[n]).double().square().sum()) for n,p in d.named_parameters() if n.startswith(prefix))**.5/max(sum(float(before[n].double().square().sum()) for n in before if n.startswith(prefix))**.5,1e-12) for prefix,name in [('input_sequence','condition_adapter_relative_update'),('_unet','image_unet_relative_update')]};del before
  for _,_,_,y,_ in pp:freq+=torch.bincount(y.flatten(),minlength=1024)
  row=allrows[step%1000] if a.stage=='MAIN' else None;step+=1
  entry=dict(branch=a.branch,stage=a.stage,effective_step=step,cycle=(step-1)//1000+1 if row else None,plan_cursor=(step-1)%1000+1 if row else None,domain=row['domain'] if row else 'fixed_8_pairs',external_QP=row['external_qp'] if row else None,loss_scale=1,gradient_clip=None,learning_rate=opt.param_groups[0]['lr'],seconds=time.time()-t,physical_gpu=a.gpu,logical_gpu=0,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,D_updates=step,G_updates=0,cache_keys=needed,**out,**update)
  with log.open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n');f.flush()
  if step%25==0 or step in cfg['stages'][a.stage]['checkpoints'] or step==budget:save_state(d,opt,a.branch,a.stage,step,last,freq)
  if step%5==0 or step==budget:dump(base/'status.json',dict(status='PASS' if step==budget else 'RUNNING',effective_updates=step,expected=budget,PID=os.getpid(),physical_gpu=a.gpu,logical_gpu=0,seconds=time.time()-starttime,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20))
  if step%100==0:print(a.branch,a.stage,step,budget,flush=True)
 rows=[json.loads(z) for z in log.read_text().splitlines()];assert len(rows)==budget and all(r['effective_step']==i+1 for i,r in enumerate(rows));write(base/'gradient_summary.csv',rows)
 dump(base/'integrity.json',dict(status='PASS',effective_D_updates=step,expected_D_updates=budget,G_updates=0,frames_per_step=cfg['stages'][a.stage]['frames'],cycles=5 if a.stage=='MAIN' else None,no_gradient_clip=True,FP32_D=True,optimizer=cfg['branches'][a.branch],loss_formula_verified=True,checkpoints=list(load(base/'checkpoint_index.json')),main_counts={str(k):v for k,v in __import__('collections').Counter((r['domain'],r['external_QP']) for r in rows).items()} if a.stage=='MAIN' else None))
if __name__=='__main__':
 try:main()
 except Exception:dump(ROOT/'logs'/f'train_failure_{os.getpid()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
