"""Zero-update audit of a separate GPU target-flow producer."""
import inspect,os,sys,time
from common_expanded import *
os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='4,5,6,7')
def main():
 import torch
 import train as original
 import runtime_acceleration as acceleration
 import parallel_flow_runtime as parallel
 source=inspect.getsource(original.train);prefix=source.split(' initial_frozen=frozen_hash()')[0]
 prefix=prefix.replace('def train():','def setup_profile():')
 prefix=prefix.replace(" rank=int(os.environ['LOCAL_RANK']);assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'",' rank=0')
 prefix=prefix.replace(" assert int(os.environ['WORLD_SIZE'])==4\n",'')
 prefix=prefix.replace(" dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=300))\n",'')
 ns=dict(original.__dict__);exec(compile(prefix+' return locals()\n',str(ROOT/'profile_parallel_flow.py'),'exec'),ns)
 c=ns['setup_profile']();state=torch.load(ROOT/'checkpoints/latest.pt',map_location='cpu',weights_only=True)
 c['gen'].load_state_dict(state['tcg_model_state'],strict=False)
 schedule=load(ROOT/'plans/schedule.json')['updates'];rows=[r for p in sorted((ROOT/'plans').glob('base_rows_*.json')) for r in load(p)]
 domain=sys.argv[-1];assert domain in ('uvg','ulong')
 item=next(s['ranks'][0] for s in schedule[state['plan_cursor']:] if rows[s['ranks'][0]['base_row']]['domain']==domain)
 row=dict(rows[item['base_row']],external_qp=item['external_qp'])
 raw=acceleration.load_uint8(row,acceleration.build_raw_reader(c['frames32'])).pin_memory()
 service=parallel.FlowService(0);started=time.monotonic();service.submit(row)
 target=service.receive(row).cuda();aux_seconds=time.monotonic()-started
 for pos in range(49,64):
  a=raw[pos-1:pos].as_strided((1,3,256,256),(3,1,768,3)).cuda().float()/255
  b=raw[pos:pos+1].as_strided((1,3,256,256),(3,1,768,3)).cuda().float()/255
  aa,bb=c['transform'](a,b)
  with torch.no_grad():flow=c['raft'](aa,bb)[-1]
  assert torch.equal(flow,target[pos-49]),'Auxiliary target flow differs from original target'
 class Cached:
  target_flows=target
  def __len__(self):return 64
  def __getitem__(self,pos):
   if pos>=64:raise IndexError(pos)
   return raw[pos:pos+1].as_strided((1,3,256,256),(3,1,768,3)).cuda(non_blocking=True).float()/255
  def close(self):pass
 def crop(function):
  torch.cuda.synchronize();t=time.monotonic()
  gs,summary=function(c['ctx'],Cached(),row['external_qp'],dists_weight=.5,raft=c['raft'],raft_transform=c['transform'])
  torch.cuda.synchronize()
  return [g.cpu() for g in gs],summary,time.monotonic()-t
 rng=c['tr'].rng_state();g1,s1,t1=crop(c['basic_crop']);c['tr'].restore_rng(rng)
 g2,s2,t2=crop(parallel.loss_function(c['basic_crop']))
 difference=sum(float((a.double()-b.double()).square().sum()) for a,b in zip(g1,g2))**.5
 norm=sum(float(a.double().square().sum()) for a in g1)**.5
 relative=difference/max(norm,1e-12)
 assert s1['losses']==s2['losses'],'Loss changed'
 assert relative<1e-3,'Gradient exceeds existing v6.40 replay tolerance'
 service.close();c['pm'].clear_dpb()
 audit=dict(status='PASS',domain=domain,optimizer_updates=0,training_only=True,
      separate_GPU_process=True,physical_GPU=4,profile_checkpoint_cursor=state['plan_cursor'],sample_sha256=item['sample_sha256'],
      target_flows_bit_exact=True,losses_bit_exact=True,gradient_relative_L2=relative,gradient_tolerance=1e-3,
      tolerance_source='v6.40 diagnostic.py',auxiliary_seconds=aux_seconds,
      baseline_cached_sample_seconds=t1,parallel_target_sample_seconds=t2,
      source_hashes={n:sha(ROOT/n) for n in ('parallel_flow_runtime.py','runtime_acceleration.py')},
      no_model_or_optimizer_or_budget_change=True)
 dump(ROOT/f'audits/parallel_flow_equivalence_{domain}.json',audit);print(audit,flush=True)
if __name__=='__main__':main()
