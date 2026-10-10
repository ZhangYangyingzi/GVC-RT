"""Publish only the two reviewed audit directories using ordinary push."""
from io21 import *
import shutil,traceback
B=ROOT.parent/'gvcrt_neural_wrapper_v6_21b_gvcrt_latent_interface_audit'
MESSAGE='experiment: add v6.21 reliability and latent interface audits'
def run(argv,cwd=REPO):return subprocess.check_output(argv,cwd=cwd,text=True,stderr=subprocess.STDOUT).strip()
def main():
 assert load(ROOT/'final_integrity.json')['status']=='PASS';assert load(B/'final_integrity.json')['status']in ('PASS','BLOCKED');remote=run(['git','remote','get-url','--push','origin']);assert 'ZhangYangyingzi/GVC-RT' in remote
 runtime=load(ROOT/'pipeline_status.json');runtime.update(status='PASS',phase='experiment_complete',active={},publication_pending=True);dump(ROOT/'pipeline_status.json',runtime)
 status=run(['git','status','--short']);(ROOT/'logs/git_status_before_publication.txt').write_text(status+'\n')
 roots=[ROOT,B];ext={'.py','.json','.csv','.png','.md','.txt','.log','.jsonl'};excl={'cache','features','bitstreams','checkpoints','__pycache__'}
 files=[p for root in roots for p in root.rglob('*')if p.is_file()and(p.suffix in ext or p.name=='.gitignore')and not any(x in excl for x in p.relative_to(root).parts)and p.name not in ('publication_status.json','upload_manifest.json')];assert all(p.stat().st_size<90*1024**2 for p in files)
 manifests={str(p.relative_to(REPO)):sha(p)for p in files};dump(ROOT/'upload_manifest.json',dict(files=manifests,checkpoints_uploaded=False,feature_tensors_uploaded=False,bitstreams_uploaded=False));files.append(ROOT/'upload_manifest.json')
 for attempt in range(3):
  run(['git','fetch','origin','main']);base=run(['git','rev-parse','origin/main']);work=REPO.parent/'GVC-RT-publication-worktrees'/f'v621_{time.time_ns()}';run(['git','worktree','add','--detach',str(work),base])
  paths=[]
  for p in files:
   rel=p.relative_to(REPO);dst=work/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);paths.append(str(rel))
  for i in range(0,len(paths),150):run(['git','add','-f','--',*paths[i:i+150]],work)
  staged=run(['git','diff','--cached','--name-only'],work).splitlines();assert staged and all(any(Path(p).is_relative_to(root.relative_to(REPO))for root in roots)for p in staged)
  run(['git','commit','-m',MESSAGE],work);commit=run(['git','rev-parse','HEAD'],work)
  try:push=run(['git','push','origin','HEAD:main'],work)
  except subprocess.CalledProcessError as e:
   dump(ROOT/'publication_status.json',dict(status='FAIL',commit=commit,error=e.output,worktree=str(work)))
   if ('fetch first'in e.output or 'non-fast-forward'in e.output)and attempt<2:continue
   raise
  run(['git','fetch','origin','main']);head=run(['git','rev-parse','origin/main']);run(['git','merge-base','--is-ancestor',commit,head])
  for p in files:
   rel=str(p.relative_to(REPO));remote_blob=run(['git','rev-parse',f'{head}:{rel}']);local_blob=run(['git','hash-object',str(work/rel)]);assert remote_blob==local_blob,rel
  dump(ROOT/'publication_status.json',dict(status='PASS',commit=commit,remote_branch='main',remote_main=head,files=len(files),all_uploaded_file_blobs_verified=True,worktree=str(work),push_output=push,finished_unix=time.time()));print('PUSH PASS',commit,flush=True);return
 raise RuntimeError('remote advanced repeatedly')
if __name__=='__main__':
 try:main()
 except Exception:
  old=load(ROOT/'publication_status.json')if(ROOT/'publication_status.json').exists()else{};dump(ROOT/'publication_status.json',dict(**{k:v for k,v in old.items()if k not in ('status','error')},status='FAIL',error=traceback.format_exc()));raise
