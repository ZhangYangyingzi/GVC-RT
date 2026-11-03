"""Transfer only this experiment to an audited runtime at a committed cursor."""
import os,signal,subprocess,sys,time,shutil,traceback
from common_expanded import *
def processes_for(name):
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   args=(p/'cmdline').read_bytes().split(b'\0')
   cwd=(p/'cwd').resolve()
   if cwd!=REPO:continue
   paths=[Path(a.decode()) for a in args if a]
   if any((v if v.is_absolute() else cwd/v)==ROOT/name for v in paths):found.append(int(p.name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 return found
def active(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False
def main():
 import torch
 for domain in ('uvg','ulong'):assert load(ROOT/f'audits/acceleration_equivalence_{domain}.json')['status']=='PASS'
 runners=processes_for('runner.py');finalizers=processes_for('finalize.py');training=processes_for('train.py')
 assert len(runners)==len(finalizers)==1 and len(training)==5,(runners,finalizers,training)
 runner=runners[0];finalizer=finalizers[0]
 torchrun=next(p for p in training if b'torch.distributed.run' in Path(f'/proc/{p}/cmdline').read_bytes())
 ranks=[p for p in training if p!=torchrun]
 for p in ranks:assert int(Path(f'/proc/{p}/stat').read_text().split()[3])==torchrun
 assert int(Path(f'/proc/{torchrun}/stat').read_text().split()[3])==runner
 # Pause only this experiment's CPU supervisors so they cannot mistake this
 # deliberate transfer for a model/data failure or publish a partial result.
 os.kill(finalizer,signal.SIGSTOP);os.kill(runner,signal.SIGSTOP)
 old=load(ROOT/'training_summary.json')['committed_global_updates']
 deadline=time.monotonic()+60
 while time.monotonic()<deadline:
  current=load(ROOT/'training_summary.json')['committed_global_updates']
  if current>old:break
  time.sleep(.1)
 else:raise RuntimeError('No new committed boundary observed; manual inspection required')
 os.kill(torchrun,signal.SIGTERM)
 deadline=time.monotonic()+40
 while any(active(p) for p in [torchrun,*ranks]) and time.monotonic()<deadline:time.sleep(.1)
 assert not any(active(p) for p in [torchrun,*ranks]),'Old experiment ranks did not exit'
 latest=ROOT/'checkpoints/latest.pt';state=torch.load(latest,map_location='cpu',weights_only=True)
 step=state['committed_global_updates'];assert step==state['plan_cursor'] and state['rank_sample_cursor']==step*4
 assert {int(s['step']) for s in state['optimizer']['state'].values()}=={step}
 assert state['source_hashes']==load(ROOT/'source_manifest.json')['training_sources']
 assert all(x is None for d in state['TCG_buffers'] for x in d.values())
 receipt=ROOT/f'receipts/{step:05d}.json'
 if not receipt.exists():dump(receipt,state['last_committed_receipt'])
 assert len(list((ROOT/'receipts').glob('*.json')))==step
 backup=ROOT/f'checkpoints/runtime_transition_step_{step}_original.pt'
 assert not backup.exists();shutil.copy2(latest,backup);assert sha(backup)==sha(latest)
 audit=dict(status='READY',reason='User requested training throughput improvement',
      committed_global_updates=step,rank_sample_cursor=step*4,old_checkpoint_filename=backup.name,
      old_checkpoint_sha256=sha(backup),protocol_sha256=state['protocol_sha256'],
      original_source_hashes=state['source_hashes'],all_rank_RNG_states_preserved=True,
      optimizer_state_preserved=True,no_committed_optimizer_replay=True,
      original_processes=dict(runner=runner,torchrun=torchrun,ranks=ranks,finalizer=finalizer),
      finalizer_paused_until_resumed_smoke=True,
      runtime_hashes={n:sha(ROOT/n) for n in ('runtime_acceleration.py','train_accelerated.py','runner_accelerated.py')},
      equivalence_audits={d:sha(ROOT/f'audits/acceleration_equivalence_{d}.json') for d in ('uvg','ulong')},
      unchanged=['5000 global updates','four ranks / effective batch 4','data plan and cursor','64 frames / 48 warmup P / 15 supervised P',
                 'losses and weights','precision','AdamW state and hyperparameters','global clipping','per-step frozen-byte checks','no BPTT','RANS coding path'])
 dump(ROOT/'audits/runtime_transition.json',audit)
 # This stopped CPU supervisor has no remaining worker. Terminate only it.
 os.kill(runner,signal.SIGTERM);os.kill(runner,signal.SIGCONT)
 manifest=load(ROOT/'source_manifest.json');manifest['runtime_acceleration']=audit['runtime_hashes']
 dump(ROOT/'source_manifest.json',manifest)
 with (ROOT/'logs/runner_accelerated.log').open('a') as log:
  p=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'runner_accelerated.py')],cwd=REPO,
       env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6,7',CUDA_DEVICE_ORDER='PCI_BUS_ID'),stdout=log,stderr=subprocess.STDOUT)
  audit.update(status='RESUMING',new_runner_pid=p.pid);dump(ROOT/'audits/runtime_transition.json',audit)
  event('RUNTIME_TRANSFER',committed_global_updates=step,new_runner_pid=p.pid)
  sys.exit(p.wait())
if __name__=='__main__':
 try:main()
 except Exception as exc:
  dump(ROOT/'audits/runtime_transition_error.json',dict(error=str(exc),traceback=traceback.format_exc()))
  traceback.print_exc();sys.exit(1)
