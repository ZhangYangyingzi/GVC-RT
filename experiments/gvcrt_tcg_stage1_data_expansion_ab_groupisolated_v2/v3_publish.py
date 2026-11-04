"""Normal fast-forward code/protocol publication first, results publication after integrity."""
from v3_common import *
from publish import eligible,BRANCH,REL,DENY
import argparse,traceback
def git_run(args,cwd=REPO,timeout=1800):
 r=subprocess.run(['git',*args],cwd=cwd,text=True,capture_output=True,timeout=timeout)
 if r.returncode:raise RuntimeError(json.dumps(dict(command=['git',*args],returncode=r.returncode,stdout=r.stdout[-4000:],stderr=r.stderr[-8000:])))
 return r.stdout.strip()
def remote_head():
 text=git_run(['ls-remote','origin','refs/heads/'+BRANCH],timeout=120);assert text;return text.split()[0]
def code_file(path):
 rel=path.relative_to(ROOT)
 if len(rel.parts)==1:return path.suffix in ('.py','.md') and path.name!='REPORT.md' or path.name in ('protocol.json','protocol_v3.json','ACTIVE_PROTOCOL.json','source_manifest.json','source_manifest_v3.json','task_identity.json','revision_diff.json','split_audit.json')
 if rel.parts[0] in ('migrations','repairs'):return True
 if rel.parts[0]=='audits':return path.name.startswith(('startup_repair','launcher_initialization','runner_resume_geometry','post_repair_resume','source_and_parent','reuse.','known_source_links','grouping_regression'))
 return rel.parts[0]=='plans' and path.name=='index.json'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('code','results'),required=True);args=ap.parse_args();verify_sources()
 with lock('publication_v3.lock'):
  status_path=ROOT/'publication_status_v3.json';state=load(status_path) if status_path.exists() else dict(branch=BRANCH,protocol_revision=V3,code_commit_remote_verified=False,results_remote_verified=False,core_blobs_verified=False)
  if args.phase=='results':assert load(ROOT/'final_integrity_v3.json')['status']=='PASS'
  else:assert load(MIG/'selftest.json')['status']=='PASS' and (MIG/'runtime_selection.json').exists()
  state.update(status='CODE_PUBLICATION_IN_PROGRESS' if args.phase=='code' else 'RESULT_PUBLICATION_IN_PROGRESS',last_phase=args.phase);dump(status_path,state)
  current=remote_head();git_run(['fetch','origin','refs/heads/'+BRANCH]);git_run(['merge-base','--is-ancestor',BASELINE,current])
  location=ROOT/'audits/publication_v3_worktree.local.json'
  if location.exists():
   pub=Path(load(location)['path']);assert (pub/'.git').exists();git_run(['merge-base','--is-ancestor',current,'HEAD'],pub)
  else:
   pub=REPO.parent/f'tcg_online_v3_publication_{int(time.time())}';git_run(['worktree','add','--detach','--no-checkout',str(pub),current]);git_run(['sparse-checkout','set','--cone',REL],pub);git_run(['checkout',current],pub);dump(location,dict(path=str(pub),initial_remote_HEAD=current,long_term_project_directory=True))
  if args.phase=='results':
   parts=[]
   for log in sorted((ROOT/'logs').glob('*.log')):
    with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2048));parts.append(log.name+'\n'+f.read().decode(errors='replace'))
   (ROOT/'logs/summary.txt').write_text('\n'.join(parts)[-128*1024:])
  files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and eligible(p) and p.name not in ('run_status.json','control_v3.json','publication_status.json','publication_status_v3.json') and (args.phase=='results' or code_file(p))]
  checksum_name='CODE_SHA256_MANIFEST_v3.json' if args.phase=='code' else 'SHA256_MANIFEST_v3.json'
  files=[p for p in files if p.name not in ('CODE_SHA256_MANIFEST_v3.json','SHA256_MANIFEST_v3.json','PUBLIC_REASSEMBLY_V3.json')]
  dump(ROOT/checksum_name,dict(phase=args.phase,protocol_revision=V3,files=[dict(file=str(p.relative_to(ROOT)),size_bytes=p.stat().st_size,sha256=sha(p)) for p in files],exclusions='weights/media/cache/PDF/raw logs and mutable status/control records'))
  files.append(ROOT/checksum_name);allowed=[];assembly=[]
  for source in files:
   destination=pub/REL/source.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
   if source.stat().st_size>=49*2**20:
    pieces=[]
    with source.open('rb') as inp:
     i=0
     while block:=inp.read(45*2**20):
      piece=destination.with_name(destination.name+f'.part{i:04d}');piece.write_bytes(block);allowed.append(str(piece.relative_to(pub)));pieces.append(dict(file=str(piece.relative_to(pub/REL)),sha256=sha(piece),size_bytes=len(block)));i+=1
    assembly.append(dict(file=str(source.relative_to(ROOT)),sha256=sha(source),pieces=pieces,reassembly='Concatenate in listed order and verify original SHA256'))
   else:shutil.copyfile(source,destination);allowed.append(str(destination.relative_to(pub)))
  if assembly:
   target=pub/REL/'PUBLIC_REASSEMBLY_V3.json';target.write_text(json.dumps(dict(files=assembly),indent=2)+'\n');allowed.append(str(target.relative_to(pub)))
  for start in range(0,len(allowed),100):git_run(['add','-f','--',*allowed[start:start+100]],pub)
  staged=[p for p in git_run(['diff','--cached','--name-only','-z'],pub).split('\0') if p]
  assert all(p.startswith(REL+'/') and p in allowed and (pub/p).stat().st_size<50*2**20 and Path(p).suffix not in DENY for p in staged)
  git_run(['diff','--cached','--check'],pub)
  audit=dict(status='PASS',phase=args.phase,explicit_allowlist=True,files=[dict(file=p,size_bytes=(pub/p).stat().st_size) for p in staged],no_weights_media_cache=True,remote_base=current)
  dump(ROOT/f'audits/staging_v3_{args.phase}.json',audit);destination=pub/REL/f'audits/staging_v3_{args.phase}.json';destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f'audits/staging_v3_{args.phase}.json',destination);git_run(['add','--',str(destination.relative_to(pub))],pub);git_run(['diff','--cached','--check'],pub)
  if git_run(['diff','--cached','--name-only'],pub):git_run(['commit','-m','Register TCG online development stopping and paired logical-batch runtime' if args.phase=='code' else 'Record TCG independent stopping results and dynamic final comparisons'],pub)
  commit=git_run(['rev-parse','HEAD'],pub);git_run(['push','origin',f'HEAD:refs/heads/{BRANCH}'],pub);assert remote_head()==commit
  core=['protocol_v3.json','source_manifest_v3.json','PROTOCOL_REVISION_V3.md','v3_train.py','v3_controller.py'] if args.phase=='code' else ['REPORT.md','protocol_v3.json','source_manifest_v3.json','selected_models_v3.json','metrics_v3.csv','paired_deltas_v3.csv','final_integrity_v3.json']
  for name in core:assert hashlib.sha256(subprocess.check_output(['git','show',f'{commit}:{REL}/{name}'],cwd=pub)).hexdigest()==sha(ROOT/name)
  if args.phase=='code':state.update(status='CODE_PROTOCOL_PUBLISHED_RESULTS_PENDING',code_commit=commit,code_commit_remote_verified=True,code_core_blobs_verified=True,code_commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{commit}')
  else:state.update(status='PUBLISHED',results_data_commit=commit,results_commit=commit,results_remote_verified=True,core_blobs_verified=True,results_commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{commit}')
  dump(status_path,state)
  # The metadata commit names the already remotely verified data commit.
  target=pub/REL/'publication_status_v3.json';shutil.copyfile(status_path,target);git_run(['add','--',str(target.relative_to(pub))],pub);git_run(['diff','--cached','--check'],pub)
  if git_run(['diff','--cached','--name-only'],pub):git_run(['commit','-m','Record verified TCG v3 publication status'],pub)
  git_run(['push','origin',f'HEAD:refs/heads/{BRANCH}'],pub);head=git_run(['rev-parse','HEAD'],pub);assert remote_head()==head
  for name in core:assert hashlib.sha256(subprocess.check_output(['git','show',f'{head}:{REL}/{name}'],cwd=pub)).hexdigest()==sha(ROOT/name)
  state['code_metadata_commit' if args.phase=='code' else 'results_commit']=head
  if args.phase=='results':state['results_commit_url']=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{head}'
  dump(status_path,state);event('V3_PUBLICATION_VERIFIED',phase=args.phase,commit=head,branch=BRANCH)
if __name__=='__main__':
 try:main()
 except Exception:
  path=ROOT/'publication_status_v3.json';r=load(path) if path.exists() else {};r.update(status='FAILED',error=traceback.format_exc(),recorded_unix=time.time());dump(path,r);traceback.print_exc();sys.exit(1)
