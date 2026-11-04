"""Explicit entry for the verified evaluator scheduling addendum and unchanged v3 training."""
from eval_parallel_common import *

verify_parallel_sources()
for name in ('runner.lock','runner_v3.lock','eval_parallel_controller.lock'):
    with (ROOT/name).open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert load(EP/'benchmark.json')['correctness_status']=='PASS'
if '--resume' in sys.argv:
    control=load(ROOT/'control_v3.json');control.update(pause_requested=False,explicit_resume_unix=time.time(),evaluation_addendum=ADDENDUM);dump(ROOT/'control_v3.json',control)
active=load(ROOT/'ACTIVE_PROTOCOL.json');active.update(runner='eval_parallel_controller.py',launcher='launch_eval_parallel.py',evaluation_runtime_addendum='migrations/eval_parallel_v1/protocol.json',evaluation_source_manifest='migrations/eval_parallel_v1/source_manifest.json');dump(ROOT/'ACTIVE_PROTOCOL.json',active)
os.execv(sys.executable,[sys.executable,'-B','-u',str(ROOT/'eval_parallel_controller.py')])
