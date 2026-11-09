"""Immutable inputs/code, and protected read-only reference fingerprints."""
from common_s2 import *

def create():
    assert not (ROOT/'source_manifest.json').exists(),'SOURCE_ALREADY_FROZEN'
    code={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if not any(x in p.parts for x in ['cache','tmp'])}
    immutable={str(p.relative_to(ROOT)):sha(p) for d in ['plans','manifests'] for p in (ROOT/d).rglob('*') if p.is_file()}
    protected={}
    for src in [REFERENCE_EXPERIMENT,SOURCE,GAIN_SOURCE]:
        for name in ['protocol.json','source_manifest.json','baseline_provenance.json','final_integrity.json','selection.json','REPORT.md']:
            p=src/name
            if p.exists():protected[str(p)]=sha(p)
    for p in [INIT,BASELINE,PILOT,RAFT,PROJECT/'checkpoints/GVC-RT_I.pt',PROJECT/'checkpoints/GVC-RT_P.pt',HIST/'manifests/B_expanded_ulong.json',REFERENCE_EXPERIMENT/'manifests/effective_ULong_pool.json',REFERENCE_EXPERIMENT/'audits/source_isolation.json']:
        protected[str(p)]=sha(p)
    # Explicit expected SHA checks remain in the model loader, not substituted
    # by fingerprints of arbitrary discovered weights.
    assert protected[str(INIT)]==INIT_SHA and protected[str(BASELINE)]==BASELINE_SHA and protected[str(PILOT)]==PILOT_SHA and protected[str(RAFT)]==RAFT_SHA
    for n in ['GVC-RT_I.pt','GVC-RT_P.pt']:assert protected[str(PROJECT/'checkpoints'/n)]==expected_source_sha(PROJECT/'checkpoints'/n)
    dump(ROOT/'source_manifest.json',dict(status='FROZEN',code=code,immutable_manifests=immutable,protected_sources=protected,protocol_sha256=sha(ROOT/'protocol.json'),plan_index_sha256=sha(ROOT/'plans/index.json'),calibration_sha256=sha(ROOT/'audits/calibration.json'),time=time.time()))

def verify():
    m=load(ROOT/'source_manifest.json');assert m['status']=='FROZEN'
    assert sha(ROOT/'protocol.json')==m['protocol_sha256'];assert sha(ROOT/'plans/index.json')==m['plan_index_sha256'];assert sha(ROOT/'audits/calibration.json')==m['calibration_sha256']
    for name,h in m['code'].items():assert sha(ROOT/name)==h,('SOURCE_CHANGED_AFTER_FREEZE',name)
    for name,h in m['immutable_manifests'].items():assert sha(ROOT/name)==h,('MANIFEST_CHANGED_AFTER_FREEZE',name)
    for name,h in m['protected_sources'].items():assert sha(name)==h,('READONLY_SOURCE_CHANGED',name)
    return True

if __name__=='__main__':
    if (ROOT/'source_manifest.json').exists():verify()
    else:create()
