"""Publication-only sparse-index adapter; never changes frozen inference code or results."""
from common_s2 import *
import re
import publish

def main():
    env();verify_sources();assert load(ROOT/'final_integrity.json')['status']=='PASS'
    if not (ROOT/'audits/run_failure_v1.json').exists():dump(ROOT/'audits/run_failure_v1.json',load(ROOT/'status.json'))
    from runner import update
    update('publication_retry','RUNNING',local_evaluation_complete=True)
    failure=ROOT/'audits/publication_failure_v1.json'
    if not failure.exists():
        previous=load(ROOT/'publication_status.json');assert previous['status']=='FAILED'
        dump(failure,previous)
    report=ROOT/'REPORT.md';before=report.read_text();after=re.sub(r'(?m)(^\|[^\n]*\|)\n\n(?=\|)',r'\1\n',before)
    assert [x for x in before.splitlines() if x.strip()]==[x for x in after.splitlines() if x.strip()]
    if after!=before:
        old_sha=sha(report);report.write_text(after)
        dump(ROOT/'audits/report_formatting.json',dict(reason='Normalize blank lines inside Markdown table only',before_sha256=old_sha,after_sha256=sha(report),numeric_content_unchanged=True,time=time.time()))
    dump(ROOT/'audits/publication_repair.json',dict(status='APPLIED',reason='Existing worktree sparse checkout excludes the newly authorized experiment directory',
        change='Only inject git add --sparse into the existing DEST-only allowlist publisher. No sparsity pattern, existing file, model, protocol, or metric is changed.',
        repair_script_sha256=sha(ROOT/'publish_repaired.py'),original_publisher_sha256=sha(ROOT/'publish.py'),
        original_inference_code_manifest_sha256=sha(ROOT/'sources.json'),GPU_evaluation_repeated=False,time=time.time()))
    manifest=load(ROOT/'artifact_manifest.json')
    for item in manifest['files']:
        p=ROOT/item['path'];item.update(bytes=p.stat().st_size,sha256=sha(p))
    known={r['path'] for r in manifest['files']}
    for name in ['publish_repaired.py','audits/run_failure_v1.json','audits/publication_failure_v1.json','audits/publication_repair.json','audits/report_formatting.json']:
        p=ROOT/name
        if p.exists() and name not in known:manifest['files'].append(dict(path=name,bytes=p.stat().st_size,sha256=sha(p)))
    dump(ROOT/'artifact_manifest.json',manifest)
    original_git=publish.git
    def sparse_git(args,check=True):
        if args and args[0]=='add':args=['add','--sparse',*args[1:]]
        return original_git(args,check)
    publish.git=sparse_git
    try:publish.main()
    except Exception:
        p=load(ROOT/'publication_status.json');p['resume_command']=f'{PYTHON} -B -u {ROOT}/publish_repaired.py';dump(ROOT/'publication_status.json',p)
        update('publication_retry','FAILED',local_evaluation_complete=True,failure_stage='publication')
        raise

if __name__=='__main__':
    with lock('runner.lock'):main()
