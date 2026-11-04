"""Independent one-update four-GPU/reference versus two-GPU accumulation copies."""
from v3_common import *
import socket,traceback
def launch(arm,source,target,cards,output=None,logname=None):
 sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
 env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=','.join(map(str,cards)),TCG_ARM=arm,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 command=[PYTHON,'-m','torch.distributed.run','--nnodes=1',f'--nproc_per_node={len(cards)}','--master_addr=127.0.0.1',f'--master_port={port}',str(ROOT/'v3_train.py'),'--arm',arm,'--source',str(source),'--target',str(target),'--cards',env['CUDA_VISIBLE_DEVICES']]
 if output:command+=['--probe-output',str(output)]
 path=ROOT/'logs'/(logname or f'v3_{arm}_{target}.log');handle=path.open('a')
 begun=time.time();process=subprocess.Popen(command,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
 record=dict(command=command,pid=process.pid,port=port,physical_GPUs=cards,physical_world_size=len(cards),logical_world_size=4,effective_batch_size=4,started_unix=begun,log=str(path.relative_to(ROOT)))
 return process,handle,record
def main():
 verify_sources();takeover=load(MIG/'takeover.json');arm=ARMS[0];source=ROOT/takeover['frozen_checkpoints'][arm]['checkpoint'];start=takeover['frozen_checkpoints'][arm]['additional_step'];expected=sha(source)
 outputs=[];runs=[]
 for name,cards in [('four_card_reference',[4,5,6,7]),('two_card_accumulation',[4,5])]:
  output=MIG/'checkpoints'/f'{name}.pt';process,handle,run=launch(arm,source,start+1,cards,output,logname=f'v3_probe_{name}.log');code=process.wait();handle.close();run.update(exit_code=code,elapsed_seconds=time.time()-run['started_unix']);runs.append(run)
  if code:
   dump(MIG/'gradient_equivalence.json',dict(status='FAILED',fallback='four_gpu_alternating_segments',reason=f'{name} exited {code}; see log',runs=runs,no_formal_optimizer_update=True));return
  outputs.append(output)
 import torch
 ref,new=[torch.load(p,map_location='cpu',weights_only=True) for p in outputs];assert sha(source)==expected
 comparisons={}
 def compare(name,a,b):
  a=a.double().reshape(-1);b=b.double().reshape(-1);difference=b-a;norm=float(a.norm());relative=float(difference.norm())/max(norm,1e-12)
  record=dict(max_absolute_error=float(difference.abs().max()) if a.numel() else 0.,relative_L2=relative,tolerance_relative_L2=.001,reference_norm=norm,bitwise_equal=bool(torch.equal(a,b)))
  assert bool(torch.isfinite(a).all() and torch.isfinite(b).all());comparisons[name]=record
  return relative<.001 if norm>0 else bool(torch.equal(a,b))
 passed=[]
 for name in ('reduced_gradient','clipped_gradient','parameter_after'):passed.append(compare(name,ref[name],new[name]))
 passed.append(compare('parameter_update_delta',ref['parameter_after']-ref['parameter_before'],new['parameter_after']-new['parameter_before']))
 for name in ref['receipt']['mean_losses']:
  passed.append(compare('loss_'+name,torch.tensor([ref['receipt']['mean_losses'][name]],dtype=torch.float64),torch.tensor([new['receipt']['mean_losses'][name]],dtype=torch.float64)))
 assert ref['optimizer']['param_groups']==new['optimizer']['param_groups']
 for key,values in ref['optimizer']['state'].items():
  for name,value in values.items():
   other=new['optimizer']['state'][key][name]
   if name=='step':assert torch.equal(value,other) and int(value)==5000+start+1
   else:passed.append(compare(f'optimizer_{key}_{name}',value,other))
 assert ref['frozen_state_hash']==new['frozen_state_hash']
 assert ref['formal_updates_committed']==new['formal_updates_committed']==0
 assert all(a['buffer_reset'] and a['frozen_check']=='EXACT_DEVICE_BYTES_PASS' for state in (ref,new) for a in state['receipt']['rank_audits'])
 assert [a['sample_sha256'] for a in ref['receipt']['rank_audits']]==[a['sample_sha256'] for a in new['receipt']['rank_audits']]
 def equal(a,b):
  if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
  if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
  if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
  return a==b
 assert equal(ref['all_rank_RNG_states'],new['all_rank_RNG_states'])
 for lr in range(4):
  a=ref['receipt']['rank_audits'][lr]['losses'];b=new['receipt']['rank_audits'][lr]['losses']
  for name in a:passed.append(compare(f'logical{lr}_loss_{name}',torch.tensor([a[name]],dtype=torch.float64),torch.tensor([b[name]],dtype=torch.float64)))
 dump(MIG/'gradient_equivalence.json',dict(status='PASS' if all(passed) else 'FAILED',selected_runtime='two_gpu_parallel_arms' if all(passed) else 'four_gpu_alternating_segments',fallback_reason=None if all(passed) else 'At least one unchanged relative-L2<1e-3 numerical criterion failed',comparisons=comparisons,runs=runs,source_checkpoint_sha256=expected,source_additional_step=start,source_and_formal_cursor_unchanged=True,no_formal_optimizer_update=True,logical_RNG_streams=4,logical_sample_hashes_match=True,frozen_state_hash=ref['frozen_state_hash'],buffer_reset=True,claim='Numerical tolerance acceptance only; changed reduction order is not claimed bitwise equivalent'))
if __name__=='__main__':
 try:main()
 except Exception:
  dump(MIG/'gradient_equivalence.json',dict(status='FAILED',selected_runtime='four_gpu_alternating_segments',fallback_reason=traceback.format_exc(),no_formal_optimizer_update=True));traceback.print_exc()
