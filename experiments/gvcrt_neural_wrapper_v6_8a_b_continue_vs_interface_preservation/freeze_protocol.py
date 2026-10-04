"""Seal implementation only before formal training; preserve development snapshots."""
from v68_io import *
def main():
    assert all(p.stat().st_size==0 for p in (ROOT/'training_logs').glob('*.jsonl')),'Cannot revise a started training protocol'
    assert not list((ROOT/'branches').glob('*/checkpoints/step_*.pt')),'Cannot revise checkpointed training'
    prior=ROOT/'audits/local_protocol.json'
    if prior.exists():dump(ROOT/'audits/development_protocol_history'/f'{time.time_ns()}.json',load(prior))
    fixed=[*ROOT.glob('*.py'),ROOT/'config.json',ROOT/'source_manifest.json',ROOT/'qp_semantics_audit.json',ROOT/'audits/continuation_training_plans.json',ROOT/'audits/interface_diagnostic_plan.json',ROOT/'audits/fid_protocol_audit.json',ROOT/'audits/vimeo_heldout_manifest.json',*[ROOT/'audits'/f'fid_bootstrap_indices_{d}.json' for d in DATASETS]]
    dump(prior,{str(p.relative_to(ROOT)):sha(p) for p in fixed})
    failed=ROOT/'audits/interface_preservation_feasibility.json'
    if failed.exists() and load(failed)['status']!='PASS':dump(ROOT/'logs/failures'/f'preflight_{time.time_ns()}.json',load(failed))
    print('PROTOCOL SEALED',len(fixed),flush=True)
if __name__=='__main__':main()
