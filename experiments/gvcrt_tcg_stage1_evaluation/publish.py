"""Allowlisted publication via a new isolated worktree and a new origin branch."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evaluate import ROOT, REPO, PILOT, LARGE, public, dump, load, sha

PREFIX=ROOT.relative_to(REPO)
REQUESTED_BRANCH='codex/tcg-stage1-eval-20261102'
CODE=['src/models/image_model_gvcrt.py','src/models/improved_model_gvcrt.py','src/models/video_model_gvcrt.py']
ALLOWED_SUFFIXES={'.py','.json','.jsonl','.csv','.md'}
GIT_ENV=dict(os.environ,GIT_TERMINAL_PROMPT='0')
GIT_ENV['GIT_SSH_COMMAND']='ssh -o BatchMode=yes -o ConnectTimeout=20'


def git(*args,cwd=REPO,check=True):
    p=subprocess.run(['git',*args],cwd=cwd,env=GIT_ENV,text=True,
                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    with (LARGE/'publication_commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(args=list(args),returncode=p.returncode,output=p.stdout,time=time.time()))+'\n')
    if check and p.returncode:
        raise RuntimeError(f'git {args[0]} failed ({p.returncode}): '+public(p.stdout))
    return p


def stage_check(wt,allowed):
    paths=git('diff','--cached','--name-only',cwd=wt).stdout.splitlines()
    assert paths and set(paths)<=set(allowed),paths
    entries=[]
    for name in paths:
        p=wt/name
        assert not p.is_symlink() and p.is_file()
        assert p.suffix in ALLOWED_SUFFIXES and p.stat().st_size<50*1024*1024,name
        entries.append(dict(file=name,size_bytes=p.stat().st_size,sha256=sha(p)))
    return entries


def verify(wt,branch,head):
    remote=git('ls-remote','--heads','origin','refs/heads/'+branch,cwd=wt).stdout.split()
    assert remote and remote[0]==head,'Remote branch/commit verification failed'
    return True


def run():
    state=load(ROOT/'run_status.json')
    assert state['status'] in ('PASS','BLOCKED','FAILED'), 'Evaluation is still running'
    assert (ROOT/'REPORT.md').is_file()
    # Existing pilot files are read-only; publish normalized small audit copies.
    extra={}
    for name in ('step0_audit.json','smoke_updates_1_2.json'):
        p=PILOT/name
        extra[name]=dict(sha256=sha(p),content=load(p))
    dump(ROOT/'pilot_audit_details.json',extra)
    manifest=load(ROOT/'source_manifest.json')
    manifest['publication_sources']=[dict(file=str((ROOT/'publish.py').relative_to(REPO)),sha256=sha(ROOT/'publish.py'))]
    dump(ROOT/'source_manifest.json',manifest)

    probe=git('ls-remote','--heads','origin','refs/heads/'+REQUESTED_BRANCH)
    branch=REQUESTED_BRANCH
    if probe.stdout.strip():
        branch+='-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        assert not git('ls-remote','--heads','origin','refs/heads/'+branch).stdout.strip()
    assert git('show-ref','--verify','--quiet','refs/heads/'+branch,check=False).returncode==1
    # This read-only fetch does not switch/stash/clean the shared working tree.
    git('fetch','origin','main')
    base=git('rev-parse','origin/main').stdout.strip()
    # All three source diffs were reviewed: TCG modules, calls, reset behavior,
    # legacy load support and disabling TCG on the I decoder. No unrelated hunks.
    unchanged_base=git('diff','origin/main','HEAD','--',*CODE).stdout==''
    include_code=unchanged_base
    code_audit=dict(base_commit=base,code_included=include_code,
                    reviewed_scope='P-decoder TCG modules/calls; DPB/temporal reset; legacy state loading; I-decoder disable',
                    files=[dict(file=n,sha256=sha(REPO/n)) for n in CODE])
    if not include_code:
        code_audit['reason']='Source bases differ from origin/main; publish hashes/results only to avoid mixed changes'
    dump(ROOT/'code_publication_audit.json',code_audit)
    wt=LARGE/('publish_worktree_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    git('worktree','add','-b',branch,str(wt),base)
    dest=wt/PREFIX
    assert not dest.exists(),'Result destination already exists in base tree'
    dest.mkdir(parents=True)
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.suffix in ALLOWED_SUFFIXES and '__pycache__' not in p.parts:
            assert not p.is_symlink() and p.stat().st_size<50*1024*1024
            target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target);files.append(str(target.relative_to(wt)))
            if target.suffix=='.csv':
                target.write_bytes(target.read_bytes().replace(b'\r\n',b'\n'))
    if include_code:
        for name in CODE:
            shutil.copy2(REPO/name,wt/name);files.append(name)
    publication=dict(status='PREPARED',push_success=False,remote_verified=False,
                     branch=branch,evaluation_status=state['status'],evaluation_complete=state['status']=='PASS',
                     base_commit=base,code_included=include_code)
    dump(ROOT/'publication_status.json',publication)
    shutil.copy2(ROOT/'publication_status.json',dest/'publication_status.json')
    files.append(str(PREFIX/'publication_status.json'))
    files=sorted(set(files))
    git('add','--',*files,cwd=wt)
    entries=stage_check(wt,files)
    staging=dict(status='PASS',files=entries,files_over_50_MB=[],binary_artifacts=[],
                 scope='this evaluation package and reviewed TCG source files only',
                 note='This staging audit records the data files; this audit file is added and rechecked below')
    dump(ROOT/'staging_audit.json',staging)
    shutil.copy2(ROOT/'staging_audit.json',dest/'staging_audit.json')
    files.append(str(PREFIX/'staging_audit.json'))
    git('add','--',str(PREFIX/'staging_audit.json'),cwd=wt)
    entries=stage_check(wt,files)
    git('diff','--cached','--check',cwd=wt)
    git('commit','-m','Add fixed TCG Stage 1 checkpoint evaluation and pilot audits',cwd=wt)
    data_commit=git('rev-parse','HEAD',cwd=wt).stdout.strip()
    # One push per commit, no force, no retries to other branches, never main.
    git('push','origin','HEAD:refs/heads/'+branch,cwd=wt)
    verify(wt,branch,data_commit)
    # This metadata commit records a data commit already present on the remote,
    # following the v6.40 two-commit publication pattern without self-reference.
    publication.update(status='PUBLISHED',push_success=True,remote_verified=True,
                       data_commit=data_commit,data_commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{data_commit}',
                       branch_url=f'https://github.com/ZhangYangyingzi/GVC-RT/tree/{branch}',
                       published_files=[x['file'] for x in entries])
    dump(ROOT/'publication_status.json',publication)
    shutil.copy2(ROOT/'publication_status.json',dest/'publication_status.json')
    git('add','--',str(PREFIX/'publication_status.json'),cwd=wt)
    stage_check(wt,[str(PREFIX/'publication_status.json')])
    git('commit','-m','Record verified TCG Stage 1 evaluation publication',cwd=wt)
    head=git('rev-parse','HEAD',cwd=wt).stdout.strip()
    git('push','origin','HEAD:refs/heads/'+branch,cwd=wt)
    verify(wt,branch,head)
    # Read the committed tree and compare its blobs to the prepared allowlist.
    committed=git('diff','--name-only',base,head,cwd=wt).stdout.splitlines()
    assert set(committed)==set(files)
    for name in files:
        content=subprocess.check_output(['git','show',head+':'+name],cwd=wt)
        assert len(content)<50*1024*1024
        assert content==(wt/name).read_bytes(),name
    publication.update(commit=head,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{head}',
                       committed_files_verified=True)
    dump(ROOT/'publication_status.json',publication)
    dump(LARGE/'publication_worktree.json',dict(worktree=str(wt),branch=branch,commit=head))
    print(json.dumps(publication,indent=2),flush=True)


if __name__=='__main__':
    try:
        run()
    except Exception as exc:
        old=load(ROOT/'publication_status.json') if (ROOT/'publication_status.json').exists() else {}
        old.update(status='FAILED',push_success=False,error=public(type(exc).__name__+': '+str(exc)))
        dump(ROOT/'publication_status.json',old)
        print(json.dumps(old,indent=2),flush=True)
        sys.exit(1)
