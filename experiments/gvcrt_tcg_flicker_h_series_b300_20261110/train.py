"""One GPU per model, sequential four-window accumulation and recoverable updates."""
from common_s2 import *
import argparse,traceback

def run(group,until,smoke=False):
 env();import torch
 from experiment_model import construct,restore,trainable_state
 from model_context import frozen_items
 from pipeline import wait_rows
 from cache_codec import cache_paths
 from loss import backward_window,window_masks,norm
 assert int(os.environ['CUDA_VISIBLE_DEVICES'])==GROUPS.index(group)
 assert load(ROOT/'gradient_calibration.json')['status']=='PASS' and load(ROOT/'audits/preflight.json')['status']=='PASS'
 if group==GROUPS[2]:assert load(ROOT/'audits/H2_gate.json')['status']=='PASS'
 root=ROOT/group/('engineering' if smoke else 'training');root.mkdir(parents=True,exist_ok=True);latest=root/'latest.pt';rows=plan_rows();coeff=load(ROOT/'gradient_calibration.json');c=construct(group,True,0);seed(TRAIN_SEED);step=0
 if not smoke:assert load(ROOT/f'audits/engineering_{group}.json')['status']=='PASS'
 if latest.exists():
  a=torch.load(latest,map_location='cpu',weights_only=True);assert a['protocol_SHA256']==sha(ROOT/'protocol.json') and a['plan_SHA256']==sha(ROOT/'plans/index.json');restore(c,a);restore_rng(a['RNG']);step=a['committed_updates']
  if step:
   assert a['last_receipt']['global_optimizer_updates']==step
   dump(root/f'receipts/{step:04d}.json',a['last_receipt']);dump(root/'progress.json',dict(status='RUNNING',committed_updates=step,next_cursor=4*step,seconds=a['last_receipt']['seconds'],time=time.time()))
  del a
 frozen=tensor_hash(frozen_items(c))
 def save(s,receipt):
  names={id(p):n for n,p in c['named']};a=dict(experiment_group=group,B300_SHA256=B300_SHA,common_step0_SHA256=sha(ROOT/'checkpoints/common_B300_step0.pt'),committed_updates=s,additional_updates=s,next_cursor=4*s,trainable_state=trainable_state(c),optimizer=c['optimizer'].state_dict(),optimizer_name_groups={g['group_name']:[names[id(p)] for p in g['params']] for g in c['optimizer'].param_groups},RNG=rng_state(),protocol_SHA256=sha(ROOT/'protocol.json'),plan_SHA256=sha(ROOT/'plans/index.json'),calibration_SHA256=sha(ROOT/'gradient_calibration.json'),s=.5,last_receipt=receipt,provenance=c['provenance'])
  atomic_torch(latest,a)
  if s in [0,25,50,75,100] or smoke:atomic_torch(root/f'checkpoints/step_{s:04d}.pt',a)
 if not latest.exists():save(0,None)
 for update in range(step+1,(2 if smoke else min(until,100))+1):
  begin=time.time();batch=rows[4*(update-1):4*update];assert len({r['source_group'] for r in batch})==4;waiting=wait_rows(batch);local=[];torch.cuda.reset_peak_memory_stats(0)
  for row in batch:
   _,p,rp=cache_paths(row,row['external_qp'],True);assert sha(p)==load(rp)['cache_sha256'];d=torch.load(p,map_location='cpu',weights_only=True,mmap=True);assert d['info']['source_row']==row and d['info']['actual_qps']==row['actual_frame_qps'];m=window_masks(d,c['device']);local.append((d,m))
  counts=dict(batch=4,output=sum(m['output_frames']>0 for d,m in local),latent=sum(m['latent_frames']>0 for d,m in local));c['optimizer'].zero_grad(set_to_none=True);traces=[]
  for d,m in local:traces.append(backward_window(c,d,counts,coeff,masks=m))
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in c['params']);total_norm=norm([p.grad for p in c['params']]);assert math.isfinite(total_norm) and total_norm>0
  if smoke and update==1:
   original=[p.grad.detach().cpu().clone() for p in c['params']];objective=sum(t['objective_contribution'] for t in traces);c['optimizer'].zero_grad(set_to_none=True);obj_ref=0.
   for d,m in local:obj_ref+=backward_window(c,d,counts,coeff,reference=True,masks=m)['objective_contribution']
   error=norm([p.grad.detach().cpu()-g for p,g in zip(c['params'],original)])/norm(original);assert error<1e-3 and abs(objective-obj_ref)<1e-5
   dump(ROOT/f'audits/aggregation_{group}.json',dict(status='PASS',gradient_relative_L2=error,objective_absolute_difference=abs(objective-obj_ref),global_batch=4,valid_window_denominators=counts,segment_total_backward_vs_independent_autograd=True))
   for p,g in zip(c['params'],original):p.grad=g.to(c['device'])
  torch.nn.utils.clip_grad_norm_(c['params'],1,error_if_nonfinite=True);post=norm([p.grad for p in c['params']]);c['optimizer'].step();assert all(int(v['step'])==update for v in c['optimizer'].state.values());assert tensor_hash(frozen_items(c))==frozen;assert all(torch.isfinite(p).all() for p in c['params'])
  receipt=dict(group=group,additional_step=update,global_optimizer_updates=update,effective_batch=4,next_cursor=4*update,sample_keys=[r['sample_key'] for r in batch],loss=sum(t['objective_contribution'] for t in traces),gradient_norm=total_norm,post_clip_norm=post,frozen_SHA256=frozen,windows=traces,counts=counts,physical_GPU=physical_gpu(0),PID=os.getpid(),peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),cache_wait_seconds=waiting,seconds=time.time()-begin,time=time.time())
  save(update,receipt);dump(root/f'receipts/{update:04d}.json',receipt);dump(root/'progress.json',dict(status='RUNNING',committed_updates=update,next_cursor=4*update,seconds=receipt['seconds'],time=time.time()));event('disposable_update' if smoke else 'optimizer_update_committed',group=group,step=update,loss=receipt['loss'],seconds=receipt['seconds']);del local,traces,receipt
  if update==1 and not smoke:dump(ROOT/f'audits/first_step_{group}.json',dict(seconds_from_task_start=time.time()-load(ROOT/'task_started.json')['time'],time=time.time()))
 if smoke:dump(ROOT/f'audits/engineering_{group}.json',dict(status='PASS',disposable_updates=2,formal_reloads_B300_fresh_optimizer_RNG=True,time=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--group',required=True,choices=GROUPS);p.add_argument('--until',type=int,default=100);p.add_argument('--smoke',action='store_true');a=p.parse_args();env()
 try:
  with lock(f'locks/train_{a.group}_{a.smoke}.lock'):run(a.group,a.until,a.smoke)
 except Exception as e:dump(ROOT/f'audits/train_failure_{a.group}_{os.getpid()}.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
