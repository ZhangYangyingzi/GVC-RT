"""Four train ranks plus four frozen, fixed-plan target-flow GPU producers."""
import inspect,os,sys,traceback
from common_expanded import *
def main():
 import train as original
 import parallel_runtime_adapter as acceleration
 for domain in ('uvg','ulong'):
  a=load(ROOT/f'audits/parallel_flow_equivalence_{domain}.json')
  assert a['status']=='PASS' and a['target_flows_bit_exact'] and a['losses_bit_exact']
  assert a['gradient_relative_L2']<1e-3
  for n,h in a['source_hashes'].items():assert sha(ROOT/n)==h
 transition={} if '--check' in sys.argv else load(ROOT/'audits/runtime_transition_parallel.json')
 source=inspect.getsource(original.train)
 def change(old,new):
  nonlocal source
  assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
 change(' from loss import basic_crop',' from loss import basic_crop\n basic_crop=acceleration.loss_function(basic_crop)')
 change('return tensor_hash((prefix+','return acceleration.frozen_hash((prefix+')
 target=" protocol_sha=sha(ROOT/'protocol.json');code=load(ROOT/'source_manifest.json')['training_sources']"
 change(target,target+"\n for name in ('runtime_acceleration.py','train_accelerated.py','runner_accelerated.py','parallel_flow_runtime.py','parallel_runtime_adapter.py','train_parallel.py','runner_parallel.py'):\n  code[str((ROOT/name).relative_to(REPO))]=sha(ROOT/name)")
 change("  assert st['source_checkpoint_sha256']==sha(SOURCE) and st['source_hashes']==code",
        "  assert st['source_checkpoint_sha256']==sha(SOURCE)\n  assert st['source_hashes']==code or (st['source_hashes']==transition['original_source_hashes'] and st['committed_global_updates']==transition['committed_global_updates'])")
 target=" schedule=load(ROOT/'plans/schedule.json')['updates'];rows=[r for p in sorted((ROOT/'plans').glob('base_rows_*.json')) for r in load(p)]"
 change(target,target+'\n acceleration.configure(frames32,schedule,rows,rank,start)')
 change(' def checkpoint(step,receipt,all_rng,all_buffers):',
        ' def checkpoint(step,receipt,all_rng,all_buffers):\n  if step in STEPS:acceleration.full_frozen_hash_audit()')
 change('local=dict(rank=rank,physical_GPU=rank+4,',
        'local=dict(auxiliary_flow_process=acceleration.process_info(),rank=rank,physical_GPU=rank+4,')
 change('receipt=dict(global_update=step,plan_cursor=step,world_size=4,effective_batch_size=4,',
        "receipt=dict(runtime_revision='four-train-ranks-four-frozen-flow-workers-v1',global_update=step,plan_cursor=step,world_size=4,effective_batch_size=4,")
 change(' check()\n st=torch.load(latest,map_location=',
        ' check()\n acceleration.full_frozen_hash_audit();acceleration.shutdown()\n st=torch.load(latest,map_location=')
 namespace=dict(original.__dict__,acceleration=acceleration,transition=transition)
 exec(compile(source,str(ROOT/'train_parallel.py'),'exec'),namespace)
 if '--check' in sys.argv:
  print('Parallel runtime source compilation PASS',flush=True);return
 namespace['train']()
if __name__=='__main__':
 try:main()
 except Exception as exc:
  rank=int(os.environ.get('LOCAL_RANK','0'))
  dump(ROOT/f'logs/rank_{rank}_parallel_error.json',dict(status='FAILED',error=str(exc),traceback=traceback.format_exc()))
  traceback.print_exc();sys.exit(1)
