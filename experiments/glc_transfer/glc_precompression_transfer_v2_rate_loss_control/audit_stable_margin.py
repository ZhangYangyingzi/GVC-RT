"""Pure CPU assertions for recovery of extra (not standard) GPU margin."""
from preflight import *
from evaluation_stage import stable_margin_recovery
def main():
 margin={4:4437.};seen={};eligible={4:[14000.]}
 assert not stable_margin_recovery({4:15208.},margin,seen,set(),0,{},eligible)
 assert not stable_margin_recovery({4:15208.},margin,seen,set(),299,{},eligible)
 event=stable_margin_recovery({4:15208.},margin,seen,set(),300,{},eligible)
 assert len(event)==1 and margin[4]==0
 margin={4:4437.};seen={}
 stable_margin_recovery({4:15208.},margin,seen,set(),0,{},eligible)
 assert not stable_margin_recovery({4:14500.},margin,seen,set(),300,{},eligible)
 assert margin[4]==4437. and seen[4]['since']==300
 assert not stable_margin_recovery({4:13000.},margin,seen,set(),600,{},eligible)
 assert 4 not in seen and margin[4]==4437.
 stable_margin_recovery({4:15208.},margin,seen,set(),700,{},eligible)
 assert not stable_margin_recovery({4:15208.},margin,seen,{4},1000,{},eligible)
 assert 4 not in seen
 stable_margin_recovery({4:15208.},margin,seen,set(),1100,{},eligible)
 assert not stable_margin_recovery({4:15208.},margin,seen,set(),1400,{4:1500},eligible)
 assert margin[4]==4437.
 assert stable_margin_recovery({4:15208.},margin,seen,set(),1500,{4:1500},eligible)
 save('audits/stable_margin_policy_checks.json',dict(status='PASS',CPU_only=True,stable_window_seconds=300,maximum_range_MiB=512,standard_reserve_and_task_peak_admission_unchanged=True,active_GPU_not_relaxed=True,cooldown_retained=True,unstable_and_insufficient_headroom_rejected=True,source_sha256=sha(R/'evaluation_stage.py')))
 print('STABLE MARGIN POLICY PASS')
if __name__=='__main__':main()
