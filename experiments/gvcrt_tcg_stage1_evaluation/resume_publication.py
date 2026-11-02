"""Resume the prepared, uncommitted worktree after CSV line-ending correction."""
import json
import shutil
import sys
from publish import (ROOT,REPO,LARGE,PREFIX,CODE,ALLOWED_SUFFIXES,git,stage_check,verify)
from evaluate import load,dump,sha,public


def run():
    commands=[json.loads(line) for line in (LARGE/'publication_commands.jsonl').open()]
    creation=next(c for c in commands if c['args'][:2]==['worktree','add'] and c['returncode']==0)
    _,_,_,branch,worktree,base=creation['args']
    from pathlib import Path
    wt=Path(worktree);dest=wt/PREFIX
    assert git('branch','--show-current',cwd=wt).stdout.strip()==branch
    assert git('rev-parse','HEAD',cwd=wt).stdout.strip()==base,'A commit already exists; do not replay'
    assert not git('ls-remote','--heads','origin','refs/heads/'+branch).stdout.strip()
    state=load(ROOT/'run_status.json');assert state['status']=='PASS'
    code_audit=load(ROOT/'code_publication_audit.json')
    include_code=code_audit['code_included']
    for entry in code_audit['files']:
        assert sha(REPO/entry['file'])==entry['sha256']
    manifest=load(ROOT/'source_manifest.json')
    manifest['publication_sources']=[dict(file=str((ROOT/n).relative_to(REPO)),sha256=sha(ROOT/n)) for n in ('publish.py','resume_publication.py')]
    dump(ROOT/'source_manifest.json',manifest)
    dump(ROOT/'publication_recovery.json',dict(status='PASS',
         initial_precommit_check='CSV CRLF flagged as trailing whitespace',
         correction='Normalize CSV CRLF to LF; numerical fields unchanged',
         reused_same_prepared_worktree=True,push_attempts_before_correction=0))
    publication=dict(status='PREPARED',push_success=False,remote_verified=False,
         branch=branch,evaluation_status='PASS',evaluation_complete=True,base_commit=base,code_included=include_code)
    dump(ROOT/'publication_status.json',publication)
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.suffix in ALLOWED_SUFFIXES and '__pycache__' not in p.parts:
            assert not p.is_symlink() and p.stat().st_size<50*1024*1024
            target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target);files.append(str(target.relative_to(wt)))
    if include_code:
        for name in CODE:
            shutil.copy2(REPO/name,wt/name);files.append(name)
    git('add','--',*files,cwd=wt)
    entries=stage_check(wt,files)
    audit_name=str(PREFIX/'staging_audit.json')
    dump(ROOT/'staging_audit.json',dict(status='PASS',files=[x for x in entries if x['file']!=audit_name],
         files_over_50_MB=[],binary_artifacts=[],
         scope='this evaluation package and three reviewed TCG source files only',
         note='The audit file itself is omitted from hashes to avoid self-reference; all staged files are rechecked'))
    shutil.copy2(ROOT/'staging_audit.json',dest/'staging_audit.json')
    git('add','--',audit_name,cwd=wt)
    entries=stage_check(wt,files)
    git('diff','--cached','--check',cwd=wt)
    git('commit','-m','Add fixed TCG Stage 1 checkpoint evaluation and pilot audits',cwd=wt)
    data_commit=git('rev-parse','HEAD',cwd=wt).stdout.strip()
    git('push','origin','HEAD:refs/heads/'+branch,cwd=wt)
    verify(wt,branch,data_commit)
    publication.update(status='PUBLISHED',push_success=True,remote_verified=True,
         data_commit=data_commit,data_commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{data_commit}',
         branch_url=f'https://github.com/ZhangYangyingzi/GVC-RT/tree/{branch}',published_files=[x['file'] for x in entries])
    dump(ROOT/'publication_status.json',publication)
    shutil.copy2(ROOT/'publication_status.json',dest/'publication_status.json')
    git('add','--',str(PREFIX/'publication_status.json'),cwd=wt)
    stage_check(wt,[str(PREFIX/'publication_status.json')])
    git('commit','-m','Record verified TCG Stage 1 evaluation publication',cwd=wt)
    head=git('rev-parse','HEAD',cwd=wt).stdout.strip()
    git('push','origin','HEAD:refs/heads/'+branch,cwd=wt)
    verify(wt,branch,head)
    committed=git('diff','--name-only',base,head,cwd=wt).stdout.splitlines()
    assert set(committed)==set(files)
    import subprocess
    for name in files:
        content=subprocess.check_output(['git','show',head+':'+name],cwd=wt)
        assert len(content)<50*1024*1024 and content==(wt/name).read_bytes(),name
    publication.update(commit=head,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{head}',committed_files_verified=True)
    dump(ROOT/'publication_status.json',publication)
    dump(LARGE/'publication_worktree.json',dict(worktree=str(wt),branch=branch,commit=head))
    print(json.dumps(dict(status='PUBLISHED',branch=branch,commit=head,remote_verified=True)),flush=True)


if __name__=='__main__':
    try:
        run()
    except Exception as exc:
        old=load(ROOT/'publication_status.json')
        old.update(status='FAILED',push_success=False,error=public(type(exc).__name__+': '+str(exc)))
        dump(ROOT/'publication_status.json',old)
        print(json.dumps(old),flush=True);sys.exit(1)
