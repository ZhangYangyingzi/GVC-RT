"""Training-only engineering audit; never executes an optimizer update."""
import inspect,os,sys,time
from common_expanded import *
os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='4')
def main():
 import torch
 import train as original
 import runtime_acceleration as accel
 source=inspect.getsource(original.train)
 prefix=source.split(' initial_frozen=frozen_hash()')[0]
 prefix=prefix.replace('def train():','def setup_profile():')
 prefix=prefix.replace(" rank=int(os.environ['LOCAL_RANK']);assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'",' rank=0')
 prefix=prefix.replace(" assert int(os.environ['WORLD_SIZE'])==4\n",'')
 prefix=prefix.replace(" dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=300))\n",'')
 ns=dict(original.__dict__)
 exec(compile(prefix+' return locals()\n',str(ROOT/'profile_acceleration.py'),'exec'),ns)
 c=ns['setup_profile']();tr=c['tr'];gen=c['gen'];params=c['params'];pm=c['pm'];fm=c['frames32']
 state=torch.load(ROOT/'checkpoints/latest.pt',map_location='cpu',weights_only=True)
 gen.load_state_dict(state['tcg_model_state'],strict=False)
 schedule=load(ROOT/'plans/schedule.json')['updates'];rows=[r for p in sorted((ROOT/'plans').glob('base_rows_*.json')) for r in load(p)]
 candidates=[s['ranks'][0] for s in schedule[state['plan_cursor']:]]
 domain=sys.argv[-1] if '--domain' in sys.argv else None
 item=next(x for x in candidates if domain is None or rows[x['base_row']]['domain']==domain)
 row=dict(rows[item['base_row']],external_qp=item['external_qp'])
 original_class=fm.PlanFrames
 raw_class=accel.build_raw_reader(fm)
 # Check every crop value after transfer against the exact existing GPU reader.
 t=time.monotonic();cpu=accel.load_uint8(row,raw_class);decode_seconds=time.monotonic()-t
 batch=cpu.pin_memory().cuda(non_blocking=True).float()/255
 frames=original_class(row,torch.device('cuda:0'))
 try:
  strides=[]
  for pos in range(64):
   original_frame=frames[pos]
   cached=batch[pos:pos+1].as_strided((1,3,256,256),(3,1,768,3))
   assert torch.equal(cached,original_frame)
   assert cached.stride()==original_frame.stride()
   if pos==0:strides=[list(original_frame.stride()),list(cached.stride())]
 finally:frames.close()
 frozen=c['frozen_hash'];hash_times=[]
 for _ in range(3):
  torch.cuda.synchronize();t=time.monotonic();digest=frozen();hash_times.append(time.monotonic()-t)
 # Patch only the outer frozen-hash implementation for timing; model unchanged.
 import ast
 function=next(n for n in ast.parse(source).body[0].body if isinstance(n,ast.FunctionDef) and n.name=='frozen_hash')
 fsource=ast.get_source_segment(source,function).replace('return tensor_hash(', 'return accel.frozen_hash(')
 fns=dict(original.__dict__,accel=accel,**{k:c[k] for k in ('im','pm','quality','raft')})
 exec(compile(fsource,str(ROOT/'runtime_acceleration.py'),'exec'),fns)
 fast=fns['frozen_hash'];assert fast()==digest
 fast_times=[]
 for _ in range(3):
  torch.cuda.synchronize();t=time.monotonic();assert fast()==digest;fast_times.append(time.monotonic()-t)
 baseline_rng=tr.rng_state()
 def crop(frames):
  torch.cuda.synchronize();t=time.monotonic()
  try:gs,s=c['basic_crop'](c['ctx'],frames,row['external_qp'],dists_weight=.5,raft=c['raft'],raft_transform=c['transform'])
  finally:frames.close()
  torch.cuda.synchronize()
  return [g.cpu().clone() for g in gs],s,time.monotonic()-t
 gs1,s1,baseline_seconds=crop(original_class(row,torch.device('cuda:0')))
 tr.restore_rng(baseline_rng)
 gs_repeat,s_repeat,repeat_seconds=crop(original_class(row,torch.device('cuda:0')))
 tr.restore_rng(baseline_rng)
 pinned=cpu.pin_memory()
 class Cached:
  def __len__(self):return 64
  def __getitem__(self,pos):
   if pos>=64:raise IndexError(pos)
   return pinned[pos:pos+1].as_strided((1,3,256,256),(3,1,768,3)).cuda(non_blocking=True).float()/255
  def close(self):pass
 gs2,s2,prefetched_seconds=crop(Cached())
 gradient_bit_exact=all(torch.equal(a,b) for a,b in zip(gs1,gs2))
 differences=[float((a-b).abs().max()) for a,b in zip(gs1,gs2)]
 def relative(a,b):
  numerator=sum(float((x.double()-y.double()).square().sum()) for x,y in zip(a,b))**.5
  denominator=sum(float(x.double().square().sum()) for x in a)**.5
  return numerator/max(denominator,1e-12)
 dump(ROOT/'audits/acceleration_attempt.json',dict(input_strides=strides,losses_bit_exact=s1['losses']==s2['losses'],
      gradients_bit_exact=gradient_bit_exact,max_gradient_abs_difference=max(differences),
      baseline_repeat_losses_bit_exact=s1['losses']==s_repeat['losses'],
      baseline_repeat_gradients_bit_exact=all(torch.equal(a,b) for a,b in zip(gs1,gs_repeat)),
      baseline_repeat_gradient_relative_L2=relative(gs1,gs_repeat),candidate_gradient_relative_L2=relative(gs1,gs2),
      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
      baseline_hash_seconds=hash_times,fast_hash_seconds=fast_times,decode_seconds=decode_seconds,
      baseline_crop_seconds=baseline_seconds,prefetched_crop_seconds=prefetched_seconds))
 assert s1['losses']==s2['losses'],'Loss changed'
 # v6.40 diagnostic.py already uses relative L2 < 1e-3 for gradient replay.
 # Keep this existing tolerance; report original-repeat variation separately.
 assert relative(gs1,gs_repeat)<1e-3,'Original replay exceeds existing numerical tolerance'
 assert relative(gs1,gs2)<1e-3,'Candidate exceeds existing numerical tolerance'
 assert fast()==digest and accel.full_frozen_hash_audit()==digest
 pm.clear_dpb()
 audit=dict(status='PASS',optimizer_updates=0,training_only=True,profile_checkpoint_cursor=state['plan_cursor'],
   sample_sha256=item['sample_sha256'],domain=row['domain'],all_64_input_frames_bit_exact=True,losses_bit_exact=True,
   gradients_bit_exact=gradient_bit_exact,gradient_relative_L2=relative(gs1,gs2),
   original_repeat_gradient_relative_L2=relative(gs1,gs_repeat),gradient_relative_L2_tolerance=1e-3,
   tolerance_source='v6.40 diagnostic.py gradient replay relative L2 < 1e-3',
   frozen_SHA256_preserved=True,baseline_frozen_check_seconds=hash_times,device_byte_check_seconds=fast_times,
   CPU_decode_and_hash_seconds=decode_seconds,baseline_crop_seconds=baseline_seconds,prefetched_crop_seconds=prefetched_seconds,
   caveat='Training continues on the same GPU during this engineering audit; wall times include co-residency and are not quality results')
 dump(ROOT/f"audits/acceleration_equivalence_{row['domain']}.json",audit);print(audit,flush=True)
if __name__=='__main__':main()
