"""Resume precommit publication; explicitly allow the authorized bounded log."""
import json,shutil,subprocess,sys,traceback
from common_expanded import *
from finalize import PREFIX,BASE,BRANCH,git,stage_check,remote_verify

def main():
 assert load(ROOT/'run_status.json')['status']=='PASS'
 assert load(ROOT/'training_summary.json')['committed_global_updates']==5000
 assert load(ROOT/'evaluation/run_status.json')['completed_points']==156
 assert load(ROOT/'final_integrity.json')['status']=='PASS'
 old=load(ROOT/'publication_status.json')
 assert old['status']=='FAILED' and 'git add failed' in old['error']
 assert not git('ls-remote','--heads','origin','refs/heads/'+BRANCH)
 calls=[json.loads(line) for line in (ROOT/'logs/publication_full.log').open()]
 creation=next(x for x in calls if x['argv'][:2]==['worktree','add'] and x['code']==0)
 wt=Path(creation['argv'][3]);dest=wt/PREFIX
 assert git('rev-parse','HEAD',cwd=wt)==BASE
 assert not any(x['argv'][0]=='push' for x in calls),'A push was already attempted; stop for inspection'
 # Checkpoint files remain strictly read-only throughout publication recovery.
 for c in load(ROOT/'checkpoints_manifest.json')['checkpoints']:
  p=ROOT/'checkpoints'/c['filename']
  assert p.stat().st_size==c['size_bytes'] and sha(p)==c['sha256']
 recovery=dict(status='PASS',failure_stage='PRECOMMIT_GIT_ADD',
     original_error=old['error'],correction='Explicit force-add of this task authorized bounded log summary only',
     authorized_log='logs/summary.log',log_size_bytes=(ROOT/'logs/summary.log').stat().st_size,
     no_prior_push_attempt=True,same_isolated_publication_worktree=True,
     force_push=False,training_or_evaluation_rerun=False)
 dump(ROOT/'publication_recovery.json',recovery)
 manifest=load(ROOT/'source_manifest.json')
 manifest.setdefault('publication_sources',{})[str((ROOT/'publish_repair.py').relative_to(REPO))]=sha(ROOT/'publish_repair.py')
 dump(ROOT/'source_manifest.json',manifest)
 state=dict(status='PREPARED',branch=BRANCH,push_success=False,remote_verified=False,
            training_status='PASS',evaluation_status='PASS',evaluation_complete=True)
 dump(ROOT/'publication_status.json',state)
 allowed=[]
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or p.is_symlink():continue
  rel=p.relative_to(ROOT)
  if any(x in rel.parts for x in ('checkpoints','large_artifacts','__pycache__','receipts')):continue
  if p.name.endswith('.local.json') or p.name in ('paper_text.txt','publication_full.log','last_status.json'):continue
  if rel.parts[0]=='logs' and p.name!='summary.log':continue
  if not(p.name=='.gitignore' or p.suffix in ('.py','.json','.jsonl','.csv','.md','.log')):continue
  assert p.stat().st_size<50*1024*1024
  target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  if p.suffix=='.csv':target.write_bytes(target.read_bytes().replace(b'\r\n',b'\n'))
  allowed.append(str(target.relative_to(wt)))
 logname=str(PREFIX/'logs/summary.log')
 git('add','--',*[p for p in allowed if p!=logname],cwd=wt)
 # -f applies only to adding one known 80 KiB log. No push uses force.
 git('add','-f','--',logname,cwd=wt)
 staged=stage_check(wt,allowed)
 audit_name=str(PREFIX/'audits/staging_audit.json')
 audit=dict(status='PASS',staged_files=[dict(file=p,size_bytes=(wt/p).stat().st_size,sha256=sha(wt/p)) for p in staged if p!=audit_name],
       prohibited_files=[],scope='expanded experiment only; TCG source is in pinned baseline',
       explicitly_added_ignored_file=logname,note='Audit omits its own hash to avoid self-reference; entire stage is rechecked')
 dump(ROOT/'audits/staging_audit.json',audit)
 shutil.copy2(ROOT/'audits/staging_audit.json',dest/'audits/staging_audit.json')
 if audit_name not in allowed:allowed.append(audit_name)
 git('add','--',audit_name,cwd=wt);stage_check(wt,allowed)
 git('diff','--cached','--check',cwd=wt)
 git('commit','-m','Add expanded causal TCG Stage 1 training and fixed evaluation',cwd=wt)
 data=git('rev-parse','HEAD',cwd=wt)
 git('push','origin','HEAD:refs/heads/'+BRANCH,cwd=wt);remote_verify(wt,data)
 state.update(status='PUBLISHED',push_success=True,remote_verified=True,data_commit=data,
       data_commit_url='https://github.com/ZhangYangyingzi/GVC-RT/commit/'+data,
       branch_url='https://github.com/ZhangYangyingzi/GVC-RT/tree/'+BRANCH)
 dump(ROOT/'publication_status.json',state);shutil.copy2(ROOT/'publication_status.json',dest/'publication_status.json')
 git('add','--',str(PREFIX/'publication_status.json'),cwd=wt)
 stage_check(wt,[str(PREFIX/'publication_status.json')])
 git('commit','-m','Record verified expanded TCG Stage 1 publication',cwd=wt)
 head=git('rev-parse','HEAD',cwd=wt)
 git('push','origin','HEAD:refs/heads/'+BRANCH,cwd=wt);remote_verify(wt,head)
 assert set(git('diff','--name-only',BASE,head,cwd=wt).splitlines())==set(allowed)
 for name in allowed:
  content=subprocess.check_output(['git','show',head+':'+name],cwd=wt)
  assert content==(wt/name).read_bytes() and len(content)<50*1024*1024
 state.update(commit=head,commit_url='https://github.com/ZhangYangyingzi/GVC-RT/commit/'+head,committed_files_verified=True)
 dump(ROOT/'publication_status.json',state)
 result=dict(status='PASS',experiment_directory=str(ROOT),training_status='PASS',committed_global_updates=5000,
      GPU_mapping=[4,5,6,7],training_ranks=4,auxiliary_frozen_GPU_processes=4,
      training_checkpoint_status='SAVED: step 0, 500, 1000, 2500, 5000 and latest; checkpoint reload PASS',
      evaluation_status='PASS',evaluation_summary=dict(points=156,checkpoint_steps=[0,1000,2500,5000],
        fresh_encode=True,independent_decode=True),report_path=str(ROOT/'REPORT.md'),
      GitHub_branch_URL=state['branch_url'],verified_commit=head,commit_URL=state['commit_url'])
 dump(ROOT/'last_status.json',result)
 dump(Path('/Huang_group/zyyz/home_dir/tmp/tcg_stage1_expanded_codex_last.txt'),result)
 print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  old=load(ROOT/'publication_status.json');old.update(status='FAILED',push_success=False,error=str(exc),traceback=traceback.format_exc())
  dump(ROOT/'publication_status.json',old);traceback.print_exc();sys.exit(1)
