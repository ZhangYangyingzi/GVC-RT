"""Detached sparse publication worktree, explicit allowlist, normal branch push."""
from ab_common import *
import gzip,traceback
BRANCH='codex/tcg-stage1-data-expansion-ab-20261103'
REL='experiments/gvcrt_tcg_stage1_data_expansion_ab'
def eligible(p):
 rel=p.relative_to(ROOT)
 if any(x in rel.parts for x in ('cache','large_artifacts','checkpoints','__pycache__','verified')):return False
 if p.is_symlink() or p.suffix in ('.pt','.pth','.ckpt','.bin','.mp4','.yuv','.npz','.npy','.jpg','.jpeg','.pdf','.pid','.lock','.tmp'):return False
 if rel.parts[0]=='logs':return p.name=='summary.txt'
 if p.suffix=='.png':return rel.parts[0]=='figures'
 return p.name=='.gitignore' or p.suffix in ('.py','.md','.json','.csv','.txt','.gz')
def main():
 with lock('publication.lock'):
  integrity=load(ROOT/'final_integrity.json');blocked='--blocked' in sys.argv
  assert integrity['status']==('BLOCKED' if blocked else 'PASS')
  state=dict(status='PREPARING',branch=BRANCH,training_complete=not blocked,evaluation_complete=not blocked,experiment_status=integrity['status'],parent_GTFlow_complete=True,push_success=False,remote_verified=False)
  dump(ROOT/'publication_status.json',state)
  lines=[]
  for f in sorted((ROOT/'logs').glob('*.log')):
   with f.open('rb') as h:h.seek(max(0,f.stat().st_size-4096));tail=h.read().decode(errors='replace')
   lines.append('## '+f.name+'\n'+tail)
  (ROOT/'logs/summary.txt').write_text('\n'.join(lines)[-128*1024:])
  files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and eligible(p)]
  inventory=[dict(file=str(p.relative_to(ROOT)),size_bytes=p.stat().st_size,sha256=sha(p)) for p in files if p.name not in ('publication_status.json','SHA256_MANIFEST.json')]
  dump(ROOT/'SHA256_MANIFEST.json',dict(baseline=BASELINE,files=inventory,excluded='self and publication_status to avoid self-reference; weights/media/cache/large raw logs never published'))
  files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and eligible(p)]
  existing=subprocess.check_output(['git','ls-remote','origin',f'refs/heads/{BRANCH}'],cwd=REPO,text=True).strip()
  record=ROOT/'audits/publication_worktree.local.json'
  if record.exists():pub=Path(load(record)['path'])
  else:
   assert not existing,'REMOTE_BRANCH_ALREADY_EXISTS_WITHOUT_THIS_TASK_PUBLICATION_PROVENANCE'
   pub=REPO.parent/f'gvcrt_tcg_data_expansion_publish_{int(time.time())}'
   subprocess.run(['git','worktree','add','--detach','--no-checkout',str(pub),BASELINE],cwd=REPO,check=True)
   subprocess.run(['git','sparse-checkout','set','--cone',REL],cwd=pub,check=True)
   subprocess.run(['git','checkout',BASELINE],cwd=pub,check=True)
   dump(record,dict(path=str(pub),baseline=BASELINE))
  def command(args):return subprocess.check_output(['git',*args],cwd=pub,text=True).strip()
  if existing:
   remote=existing.split()[0];assert command(['merge-base','--is-ancestor',remote,'HEAD'])==''
  copied=[]
  for p in files:
   relative=str(p.relative_to(ROOT));target=pub/REL/relative;target.parent.mkdir(parents=True,exist_ok=True)
   if p.stat().st_size>=49*2**20:
    assert p.suffix=='.json','NON_JSON_TOO_LARGE_FOR_PUBLICATION'
    target=target.with_suffix(target.suffix+'.gz')
    with p.open('rb') as src,gzip.open(target,'wb') as out:shutil.copyfileobj(src,out)
   else:shutil.copyfile(p,target)
   assert target.stat().st_size<50*2**20;copied.append(str(target.relative_to(pub)))
  # Only exact allowlisted task paths; force-add is for bounded logs ignored by
  # the repository-wide logs pattern. Never a force push.
  for start in range(0,len(copied),100):subprocess.run(['git','add','-f','--',*copied[start:start+100]],cwd=pub,check=True)
  staged=command(['diff','--cached','--name-only','-z']).split('\0')
  staged=[p for p in staged if p]
  assert all(p.startswith(REL+'/') and p in copied for p in staged)
  for p in staged:assert (pub/p).stat().st_size<50*2**20 and Path(p).suffix not in ('.pt','.pth','.ckpt','.mp4','.bin','.yuv','.jpg','.pdf')
  subprocess.run(['git','diff','--cached','--check'],cwd=pub,check=True)
  dump(ROOT/'audits/staging_audit.json',dict(status='PASS',files=[dict(file=p,size_bytes=(pub/p).stat().st_size) for p in staged],only_task_allowlist=True,no_weights_media_cache=True))
  # Include the exact staging evidence as another explicitly added file.
  target=pub/REL/'audits/staging_audit.json';target.parent.mkdir(exist_ok=True);shutil.copyfile(ROOT/'audits/staging_audit.json',target)
  subprocess.run(['git','add','--',str(target.relative_to(pub))],cwd=pub,check=True)
  if staged:subprocess.run(['git','commit','-m',('Record blocked TCG source-split audit and completed GT-flow rescore' if blocked else 'Record fixed TCG data expansion paired continuation and GT-flow evaluation')],cwd=pub,check=True)
  data_commit=command(['rev-parse','HEAD'])
  subprocess.run(['git','push','origin',f'HEAD:refs/heads/{BRANCH}'],cwd=pub,check=True)
  remote=command(['ls-remote','origin',f'refs/heads/{BRANCH}']).split()[0];assert remote==data_commit
  core=['REPORT.md','protocol.json','metrics.csv','paired_deltas.csv','final_integrity.json','source_manifest.json']
  for f in core:
   blob=subprocess.check_output(['git','show',f'{remote}:{REL}/{f}'],cwd=pub);assert hashlib.sha256(blob).hexdigest()==sha(ROOT/f)
  state.update(status='PUBLISHED',push_success=True,remote_verified=True,verified_data_commit=data_commit,core_blob_verification='PASS',branch_url=f'https://github.com/ZhangYangyingzi/GVC-RT/tree/{BRANCH}',publication_note='The committed metadata records the already verified data commit. The subsequent metadata commit is independently verified and recorded locally to avoid a self-referential commit hash.')
  dump(ROOT/'publication_status.json',state);shutil.copyfile(ROOT/'publication_status.json',pub/REL/'publication_status.json')
  subprocess.run(['git','add','--',REL+'/publication_status.json'],cwd=pub,check=True);subprocess.run(['git','diff','--cached','--check'],cwd=pub,check=True)
  if command(['diff','--cached','--name-only']):subprocess.run(['git','commit','-m','Record verified TCG paired experiment publication'],cwd=pub,check=True)
  subprocess.run(['git','push','origin',f'HEAD:refs/heads/{BRANCH}'],cwd=pub,check=True)
  final=command(['rev-parse','HEAD']);assert command(['ls-remote','origin',f'refs/heads/{BRANCH}']).split()[0]==final
  for f in core:
   assert hashlib.sha256(subprocess.check_output(['git','show',f'{final}:{REL}/{f}'],cwd=pub)).hexdigest()==sha(ROOT/f)
  state.update(verified_commit=final,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{final}');dump(ROOT/'publication_status.json',state)
  event('PUBLICATION_PASS',commit=final,branch=BRANCH)
if __name__=='__main__':
 try:main()
 except Exception as e:
  old=load(ROOT/'publication_status.json') if (ROOT/'publication_status.json').exists() else {}
  old.update(status='FAILED',error=str(e),traceback=traceback.format_exc(),resume_command=f"{PYTHON} -u {ROOT}/publish.py"+(' --blocked' if '--blocked' in sys.argv else ''));dump(ROOT/'publication_status.json',old);traceback.print_exc();sys.exit(1)
