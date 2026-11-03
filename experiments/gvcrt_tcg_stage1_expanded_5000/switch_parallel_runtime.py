"""Reuse the audited own-process transfer at a new committed boundary."""
import inspect,sys
import switch_runtime as original
from common_expanded import *
if __name__=='__main__':
 source=inspect.getsource(original.main)
 changes={
  "processes_for('runner.py')":"processes_for('runner_accelerated.py')",
  "processes_for('train.py')":"processes_for('train_accelerated.py')",
  'acceleration_equivalence_':'parallel_flow_equivalence_',
  "assert state['source_hashes']==load(ROOT/'source_manifest.json')['training_sources']":"assert all(sha(REPO/name)==h for name,h in state['source_hashes'].items())",
  'runtime_transition_step_{step}_original.pt':'runtime_transition_step_{step}_prefetch_v1.pt',
  "audits/runtime_transition.json":"audits/runtime_transition_parallel.json",
  "('runtime_acceleration.py','train_accelerated.py','runner_accelerated.py')":"('runtime_acceleration.py','train_accelerated.py','runner_accelerated.py','parallel_flow_runtime.py','parallel_runtime_adapter.py','train_parallel.py','runner_parallel.py')",
  "ROOT/'logs/runner_accelerated.log'":"ROOT/'logs/runner_parallel.log'",
  "ROOT/'runner_accelerated.py'":"ROOT/'runner_parallel.py'",
 }
 for old,new in changes.items():
  assert old in source,old;source=source.replace(old,new)
 namespace=dict(original.__dict__)
 exec(compile(source,str(ROOT/'switch_parallel_runtime.py'),'exec'),namespace)
 try:namespace['main']()
 except Exception as exc:
  import traceback
  dump(ROOT/'audits/runtime_transition_parallel_error.json',dict(error=str(exc),traceback=traceback.format_exc()))
  traceback.print_exc();sys.exit(1)
