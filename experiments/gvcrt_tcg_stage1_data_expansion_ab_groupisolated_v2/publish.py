"""Long-term detached publication worktree; normal fast-forward on existing branch."""
from ab_common import *
import traceback
BRANCH='codex/tcg-stage1-data-expansion-ab-20261103'
REL=str(ROOT.relative_to(REPO))
DENY={'.pt','.pth','.ckpt','.bin','.mp4','.yuv','.npz','.npy','.jpg','.jpeg','.pdf','.pid','.lock','.tmp','.log'}
CORE=['REPORT.md','protocol.json','source_manifest.json','split_audit.json','metrics.csv','paired_deltas.csv','warping_error.csv','final_integrity.json','historical_exposure.json']
def eligible(path):
 rel=path.relative_to(ROOT)
 if path.is_symlink() or any(x in rel.parts for x in ('cache','large_artifacts','checkpoints','__pycache__','verified')):return False
 if path.suffix in DENY:return False
 if rel.parts[0]=='logs':return path.name=='summary.txt'
 if path.suffix=='.png':return rel.parts[0]=='figures'
 if path.name.endswith('.local.json'):return False
 return path.name=='.gitignore' or path.suffix in ('.py','.md','.json','.csv','.txt','.gz')
def run(args,cwd=REPO):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()
def remote_head():return run(['ls-remote','origin','refs/heads/'+BRANCH]).split()[0]
def main():
 with lock('publication.lock'):
  assert load(ROOT/'final_integrity.json')['status']=='PASS'
  state=dict(status='PREPARING',branch=BRANCH,protocol_revision=REVISION,experiment_status='PASS',training_complete=True,evaluation_complete=True,push_success=False,remote_verified=False);dump(ROOT/'publication_status.json',state)
  tail=[]
  for f in sorted((ROOT/'logs').glob('*.log')):
   with f.open('rb') as h:h.seek(max(0,f.stat().st_size-4096));text=h.read().decode(errors='replace')
   tail.append('## '+f.name+'\n'+text)
  (ROOT/'logs/summary.txt').write_text('\n'.join(tail)[-128*1024:])
  files=[f for f in sorted(ROOT.rglob('*')) if f.is_file() and eligible(f)]
  dump(ROOT/'SHA256_MANIFEST.json',dict(protocol_revision=REVISION,files=[dict(file=str(f.relative_to(ROOT)),size_bytes=f.stat().st_size,sha256=sha(f)) for f in files if f.name not in ('SHA256_MANIFEST.json','publication_status.json','run_status.json','staging_audit.json','PUBLIC_REASSEMBLY.json')],excludes='self/mutable publication control records; all weights/media/cache artifacts'))
  files=[f for f in sorted(ROOT.rglob('*')) if f.is_file() and eligible(f)]
  current=remote_head();assert current,'EXPECTED_EXISTING_TARGET_BRANCH_MISSING'
  subprocess.run(['git','fetch','origin','refs/heads/'+BRANCH],cwd=REPO,check=True)
  assert run(['merge-base','--is-ancestor',BASELINE,current])=='','REMOTE_BASELINE_CHANGED'
  record=ROOT/'audits/publication_worktree.local.json'
  if record.exists():
   pub=Path(load(record)['path']);assert pub.is_dir() and (pub/'.git').is_file()
   assert run(['merge-base','--is-ancestor',current,'HEAD'],pub)=='','REMOTE_ADVANCED_OUTSIDE_PUBLICATION; preserve work and review'
  else:
   if run(['ls-tree','-r','--name-only',current,'--',REL]):assert json.loads(run(['show',f'{current}:{REL}/task_identity.json']))==load(ROOT/'task_identity.json'),'REMOTE_TASK_ID_CONFLICT'
   pub=REPO.parent/f'tcg_stage1_groupisolated_v2_publication_{int(time.time())}'
   subprocess.run(['git','worktree','add','--detach','--no-checkout',str(pub),current],cwd=REPO,check=True)
   subprocess.run(['git','sparse-checkout','set','--cone',REL],cwd=pub,check=True)
   subprocess.run(['git','checkout',current],cwd=pub,check=True)
   dump(record,dict(path=str(pub),remote_base=current,long_term_project_directory=True))
  allowed=[];transforms=[]
  for f in files:
   dest=pub/REL/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
   if f.stat().st_size>=49*2**20:
    parts=[]
    with f.open('rb') as src:
     k=0
     while chunk:=src.read(45*2**20):
      part=dest.with_name(dest.name+f'.part{k:04d}');part.write_bytes(chunk);allowed.append(str(part.relative_to(pub)));parts.append(dict(file=str(part.relative_to(pub/REL)),size_bytes=len(chunk),sha256=sha(part)));k+=1
    transforms.append(dict(original_file=str(f.relative_to(ROOT)),original_size_bytes=f.stat().st_size,original_sha256=sha(f),parts=parts,reassembly='Concatenate listed parts in order; verify original SHA256. Reassembled GZIP remains GZIP.'))
   else:shutil.copyfile(f,dest);allowed.append(str(dest.relative_to(pub)))
  assembly=pub/REL/'PUBLIC_REASSEMBLY.json';assembly.write_text(json.dumps(dict(records=transforms),indent=2)+'\n');allowed.append(str(assembly.relative_to(pub)))
  for start in range(0,len(allowed),100):subprocess.run(['git','add','-f','--',*allowed[start:start+100]],cwd=pub,check=True)
  staged=[s for s in run(['diff','--cached','--name-only','-z'],pub).split('\0') if s]
  assert all(s.startswith(REL+'/') and s in allowed for s in staged)
  for s in staged:assert (pub/s).stat().st_size<50*2**20 and Path(s).suffix not in DENY
  subprocess.run(['git','diff','--cached','--check'],cwd=pub,check=True)
  dump(ROOT/'audits/staging_audit.json',dict(status='PASS',files=[dict(file=s,size_bytes=(pub/s).stat().st_size) for s in staged],only_explicit_E2_allowlist=True,no_weights_media_cache=True,remote_base=current))
  target=pub/REL/'audits/staging_audit.json';target.parent.mkdir(exist_ok=True);shutil.copyfile(ROOT/'audits/staging_audit.json',target);subprocess.run(['git','add','--',str(target.relative_to(pub))],cwd=pub,check=True)
  subprocess.run(['git','diff','--cached','--check'],cwd=pub,check=True)
  if run(['diff','--cached','--name-only'],pub):subprocess.run(['git','commit','-m','Record group-isolated TCG paired continuation v2 and fixed evaluations'],cwd=pub,check=True)
  data_commit=run(['rev-parse','HEAD'],pub);subprocess.run(['git','push','origin',f'HEAD:refs/heads/{BRANCH}'],cwd=pub,check=True);assert remote_head()==data_commit
  for f in CORE:assert hashlib.sha256(subprocess.check_output(['git','show',f'{data_commit}:{REL}/{f}'],cwd=pub)).hexdigest()==sha(ROOT/f)
  state.update(status='PUBLISHED',push_success=True,remote_verified=True,verified_data_commit=data_commit,core_blob_verification='PASS',branch_url=f'https://github.com/ZhangYangyingzi/GVC-RT/tree/{BRANCH}',note='Committed metadata names the verified data commit; final metadata commit verified locally to avoid self-referential hash.')
  dump(ROOT/'publication_status.json',state);status('complete','PASS',protocol_revision=REVISION,verified_data_commit=data_commit)
  for f in ('publication_status.json','run_status.json'):
   shutil.copyfile(ROOT/f,pub/REL/f);subprocess.run(['git','add','--',REL+'/'+f],cwd=pub,check=True)
  subprocess.run(['git','diff','--cached','--check'],cwd=pub,check=True)
  if run(['diff','--cached','--name-only'],pub):subprocess.run(['git','commit','-m','Record verified group-isolated v2 completion and publication'],cwd=pub,check=True)
  subprocess.run(['git','push','origin',f'HEAD:refs/heads/{BRANCH}'],cwd=pub,check=True);head=run(['rev-parse','HEAD'],pub);assert remote_head()==head
  for f in CORE:assert hashlib.sha256(subprocess.check_output(['git','show',f'{head}:{REL}/{f}'],cwd=pub)).hexdigest()==sha(ROOT/f)
  state.update(verified_commit=head,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{head}',core_blobs_verified=True);dump(ROOT/'publication_status.json',state);event('PUBLICATION_PASS',commit=head,branch=BRANCH)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  state=load(ROOT/'publication_status.json') if (ROOT/'publication_status.json').exists() else {};state.update(status='FAILED',error=str(exc),traceback=traceback.format_exc(),resume_command=f'{PYTHON} -B -u {ROOT}/publish.py');dump(ROOT/'publication_status.json',state);traceback.print_exc();sys.exit(1)
