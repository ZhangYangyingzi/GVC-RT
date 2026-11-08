"""Explicit experiment-only allowlist; normal commit/push and remote blob verification."""
from common_s2 import *
import argparse,traceback

BRANCH='codex/tcg-lora-ulong-only-a2000-20261108'
REL=str(ROOT.relative_to(WORKTREE))
SUFFIXES={'.py','.json','.gz','.md','.csv','.png'}
DENY={'cache','tmp','logs','large_artifacts','checkpoints','receipts','window_reads','canonical_MCL','__pycache__'}

def git(args,check=True):
    p=subprocess.run(['git',*args],cwd=WORKTREE,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
    with (ROOT/'logs/publication.log').open('a') as f:f.write('$ git '+' '.join(args)+'\n'+p.stdout+'\n')
    if check and p.returncode:raise RuntimeError('GIT_COMMAND_FAILED '+str(args)+' '+p.stdout[-6000:])
    return p.stdout.strip()

def allowlist():
    found=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p.is_symlink():continue
        r=p.relative_to(ROOT)
        if set(r.parts)&DENY or p.suffix not in SUFFIXES or p.name.endswith('.tmp'):continue
        if p.name in ['publication_remote_receipt.json','heartbeat.json','runner_identity.json','training_status.json']:continue
        if p.stat().st_size>=50*2**20:raise RuntimeError('PUBLICATION_FILE_EXCEEDS_50_MIB '+str(r))
        # No media/tensor archive can be included solely because it has a gzip suffix.
        if p.suffix=='.gz':assert r.parts[0] in ['plans','published_receipts','published_read_audits'],'NON_ALLOWLIST_GZIP'
        found.append(str(p.relative_to(WORKTREE)))
    assert found and all(p.startswith(REL+'/') for p in found)
    return found

def remote_head():
    value=git(['ls-remote','--heads','origin','refs/heads/'+BRANCH]);assert value
    return value.split()[0]

def verify_core(commit,core):
    records={}
    for name in core:
        p=ROOT/name;relative=str(p.relative_to(WORKTREE));blob=git(['rev-parse',commit+':'+relative]);assert blob==git(['hash-object',str(p)])
        records[name]=dict(git_blob=blob,SHA256=sha(p))
    return records

def stage_commit_push(files,title):
    prior=git(['diff','--cached','--name-only']);assert not prior,'UNEXPECTED_PREEXISTING_STAGING_AREA'
    for start in range(0,len(files),100):git(['add','--sparse','--',*files[start:start+100]])
    staged=git(['diff','--cached','--name-only']).splitlines()
    assert set(staged)<=set(files) and all(x.startswith(REL+'/') for x in staged)
    for name in staged:assert (WORKTREE/name).stat().st_size<50*2**20
    dump(ROOT/'audits/publication_staging.json',dict(status='PASS',allowlist_files=files,staged_files=staged,
        staged_file_sizes={x:(WORKTREE/x).stat().st_size for x in staged},no_weights_media_cache=True,time=time.time()))
    git(['diff','--cached','--check'])
    if staged:git(['commit','-m',title])
    commit=git(['rev-parse','HEAD']);git(['push','origin','HEAD:refs/heads/'+BRANCH]);assert remote_head()==commit
    return commit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['preparation','results'],required=True);a=ap.parse_args();env()
    phase=a.phase;old=load(ROOT/'publication_status.json') if (ROOT/'publication_status.json').exists() else {}
    try:
        assert git(['branch','--show-current'])==BRANCH
        assert 'ZhangYangyingzi/GVC-RT' in git(['remote','get-url','origin'])
        if phase=='preparation' and old.get('preparation_published'):
            commit=old['preparation_commit'];git(['merge-base','--is-ancestor',commit,'HEAD'])
            assert git(['ls-remote','--heads','origin','refs/heads/'+BRANCH]);return
        if phase=='results':assert load(ROOT/'final_integrity.json')['status']=='PASS' and load(ROOT/'status.json')['complete']
        # The dedicated worktree shares repository history but never stages other worktrees.
        remote=git(['ls-remote','--heads','origin','refs/heads/'+BRANCH])
        if remote:
            git(['fetch','origin',BRANCH]);git(['merge-base','--is-ancestor','origin/'+BRANCH,'HEAD'])
        value=dict(old,status='PENDING',phase=phase,branch=BRANCH,repository='ZhangYangyingzi/GVC-RT',remote_verified=False,core_blobs_verified=False,time=time.time())
        dump(ROOT/'publication_status.json',value)
        core=['protocol.json','source_manifest.json','runner.py','train.py','model_context.py','loss.py','plans/index.json','audits/source_isolation.json']
        if phase=='results':core+=['REPORT.md','metrics.csv','selection.json','final_integrity.json','training_coverage.json','stream_identity_checks.csv']
        files=allowlist();payload=stage_commit_push(files,'Add ULong-only A2000 continuation '+phase)
        blobs=verify_core(payload,core)
        value.update(status='PUBLISHED',remote_verified=True,core_blobs_verified=True,payload_commit=payload,
            commit_url='https://github.com/ZhangYangyingzi/GVC-RT/commit/'+payload,core_commit_blobs=blobs,time=time.time())
        if phase=='preparation':value.update(preparation_published=True,preparation_commit=payload)
        else:value.update(results_published=True,results_commit=payload)
        dump(ROOT/'publication_status.json',value)
        if phase=='results':
            s=load(ROOT/'status.json');s.update(publication_status='PUBLISHED',commit_url=value['commit_url'],remote_verified=True,core_blobs_verified=True);dump(ROOT/'status.json',s)
        metadata=stage_commit_push([REL+'/publication_status.json',*([REL+'/status.json'] if phase=='results' else [])],
            'Record verified ULong continuation '+phase+' publication')
        assert remote_head()==metadata
        assert git(['rev-parse',metadata+':'+REL+'/publication_status.json'])==git(['hash-object',str(ROOT/'publication_status.json')])
        dump(ROOT/'publication_remote_receipt.json',dict(status='VERIFIED',phase=phase,branch=BRANCH,remote_HEAD=metadata,payload_commit=payload,
            commit_url='https://github.com/ZhangYangyingzi/GVC-RT/commit/'+metadata,remote_verified=True,core_blobs_verified=True,core_blobs=verify_core(metadata,core),time=time.time()))
        print('GITHUB_PUBLICATION_VERIFIED',metadata)
    except Exception as error:
        traceback.print_exc();value=dict(old,status='FAILED',phase=phase,branch=BRANCH,error=repr(error),remote_verified=False,core_blobs_verified=False,
            local_HEAD=git(['rev-parse','HEAD'],check=False),resume_command=f'{PYTHON} -B -u {ROOT}/publish.py --phase {phase}',time=time.time())
        dump(ROOT/'publication_status.json',value)
        if phase=='results':
            s=load(ROOT/'status.json');s.update(publication_status='FAILED',publication_error=repr(error));dump(ROOT/'status.json',s)
        # GPU engineering/training/results do not depend on network availability.
        print('PUBLICATION_FAILED_LOCAL_EXECUTION_RETAINED',repr(error),flush=True)

if __name__=='__main__':main()
