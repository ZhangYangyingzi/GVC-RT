"""Scoped normal publication from latest origin/main, with remote file verification."""
from io32 import *
import subprocess,shutil,traceback,urllib.parse
LIMIT=524288
EXCLUDE={'features','bitstreams','cache','publication','transactions','__pycache__'}
ALLOWED={'.py','.md','.json','.jsonl','.csv','.txt','.sh','.png','.yaml','.yml','.license','.toml','.log'}
def git(*args,cwd=REPO):
 r=subprocess.run(['git',*args],cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if r.returncode:raise RuntimeError('git '+args[0]+': '+r.stderr[-5000:])
 return r.stdout.strip()
def main():
 comp=load(ROOT/'computation_integrity.json');assert comp['status']=='PASS' and comp['evaluation']['completed']==3312
 remote=git('remote','get-url','origin');assert 'ZhangYangyingzi/GVC-RT'.lower() in remote.lower(), 'origin is not the authorized personal repository'
 if remote.startswith('git@'):
  host=remote.split('@',1)[1].split(':',1)[0];cfg=subprocess.check_output(['ssh','-G',host],text=True,stderr=subprocess.DEVNULL);resolved=next(x.split()[1] for x in cfg.splitlines() if x.startswith('hostname '));assert resolved=='github.com'
 else:assert urllib.parse.urlparse(remote).hostname=='github.com'
 export=ROOT/'publication/export';export.mkdir(parents=True,exist_ok=True);index=[];shards=[]
 # Include all raw task receipts and all explicit failure journals; server-only tensors excluded.
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(ROOT)
  if rel.parts[0] in ('text_shards','text_indexes') or any(x in EXCLUDE for x in rel.parts) or p.suffix in ('.pt','.npz','.npy','.bin','.lock','.pid','.tmp'):continue
  if p.suffix not in ALLOWED and p.name!='.gitignore':continue
  if p.name in ('publication_status.json','pipeline_status.json','driver_state.json','geometry_workers.json','file_index.json'):continue
  if p.suffix=='.png' and p.stat().st_size>5*1024*1024:continue
  # Raw logs are retained and line-sharded, including failures; no binary or credentials.
  dst=export/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  if p.stat().st_size<=LIMIT or p.suffix=='.png':
   shutil.copy2(p,dst);
   if p.name!='final_integrity.json':index.append(dict(relative_path=str(rel),server_path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
  else:
   assert p.suffix in ('.txt','.json','.jsonl','.csv','.md','.log'),('oversized non-text',str(p))
   parts=[];buf=bytearray();i=0
   def flush():
    nonlocal buf,i
    if not buf:return
    pp=Path('text_shards')/rel/(f'part_{i:05d}.txt');q=export/pp;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(buf);local=ROOT/pp;local.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,local);parts.append(dict(relative_path=str(pp),bytes=len(buf),sha256=sha(q)));buf=bytearray();i+=1
   with p.open('rb') as f:
    for line in f:
     assert len(line)<=LIMIT,('single line exceeds512KiB',str(p))
     if len(buf)+len(line)>LIMIT:flush()
     buf.extend(line)
   flush();h=hashlib.sha256();size=0
   for part in parts:
    b=(export/part['relative_path']).read_bytes();h.update(b);size+=len(b)
   assert h.hexdigest()==sha(p) and size==p.stat().st_size
   shards.append(dict(original_relative_path=str(rel),server_path=str(p),bytes=size,sha256=sha(p),parts=parts))
 entries=[dict(kind='file',record=r) for r in index]+[dict(kind='sharded_text',record=r) for r in shards];chunks=[];buf=bytearray();ci=0
 def flush_index():
  nonlocal buf,ci
  if not buf:return
  path=Path('text_indexes')/f'index_{ci:04d}.jsonl';q=export/path;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(buf);local=ROOT/path;local.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,local);chunks.append(dict(relative_path=str(path),bytes=len(buf),sha256=sha(q)));ci+=1;buf=bytearray()
 for e in entries:
  line=(json.dumps(e,separators=(',',':'))+'\n').encode();assert len(line)<=LIMIT
  if len(buf)+len(line)>LIMIT:flush_index()
  buf.extend(line)
 flush_index()
 record=dict(schema='v632b',files=[],sharded_texts=[],index_parts=chunks,shards_max_bytes=LIMIT,reassembly_verified=True,binary_index='binary_artifacts_index.json');dump(ROOT/'file_index.json',record);shutil.copy2(ROOT/'file_index.json',export/'file_index.json')
 git('fetch','origin','main');base=git('rev-parse','FETCH_HEAD');wt=ROOT/'publication/worktree';wt.parent.mkdir(exist_ok=True)
 if wt.exists():
  # Never destroy an existing worktree or remote history; use another isolated path.
  wt=ROOT/'publication'/f'worktree_{time.time_ns()}'
 git('worktree','add','--detach','--no-checkout',str(wt),base)
 git('sparse-checkout','set','--cone','--skip-checks',str(ROOT.relative_to(REPO)),cwd=wt);git('read-tree','-mu','HEAD',cwd=wt)
 dest=wt/ROOT.relative_to(REPO);dest.mkdir(parents=True,exist_ok=True)
 for p in export.rglob('*'):
  if p.is_file():q=dest/p.relative_to(export);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 # Publication gitignore excludes no lightweight exported receipts.
 (dest/'.gitignore').write_text('*.pt\n*.npz\n*.npy\n*.bin\n*.lock\n*.pid\n__pycache__/\nfeatures/\nbitstreams/\npublication/\n')
 rel=str(ROOT.relative_to(REPO));git('add','--',rel,cwd=wt)
 staged=git('diff','--cached','--name-only',cwd=wt).splitlines();assert staged and all(p.startswith(rel+'/') for p in staged)
 git('commit','-m','Add v6.32b native-frame FD and CORAL completion results',cwd=wt)

 for attempt in range(4):
  try:git('push','origin','HEAD:main',cwd=wt);break
  except RuntimeError as exc:
   if not any(x in str(exc).lower() for x in ('non-fast-forward','fetch first','rejected')) or attempt==3:raise
   append(ROOT/'publication/retries.jsonl',dict(error=str(exc),attempt=attempt+1,retained_local_commit=git('rev-parse','HEAD',cwd=wt)))
   git('fetch','origin','main');base=git('rev-parse','FETCH_HEAD');wt=ROOT/'publication'/f'worktree_retry_{time.time_ns()}';git('worktree','add','--detach','--no-checkout',str(wt),base);git('sparse-checkout','set','--cone','--skip-checks',rel,cwd=wt);git('read-tree','-mu','HEAD',cwd=wt);dest=wt/rel;dest.mkdir(parents=True,exist_ok=True);shutil.copytree(export,dest,dirs_exist_ok=True);git('add','--',rel,cwd=wt);git('commit','-m','Add v6.32b native-frame FD and CORAL completion results',cwd=wt)
 first=git('rev-parse','HEAD',cwd=wt);actual=git('ls-remote','origin','refs/heads/main').split()[0];assert actual==first
 keys=['config.json','protocol.json','geometry_audit.json','data_isolation_audit.json','calibration/final.json','gradient_integrity.json','computation_integrity.json','REPORT.md','file_index.json']+[f'branches/{b}/gradient_summary.csv' for b in BRANCHES]
 verified=[]
 for k in keys:
  target=rel+'/'+k
  if not (dest/k).exists():
   # Core summaries must stay directly readable, not hidden behind sharding.
   raise RuntimeError('Required direct file absent after export: '+k)
  assert git('rev-parse',first+':'+target,cwd=wt)==git('hash-object',str(dest/k),cwd=wt);verified.append(k)
 final=dict(comp,status='PASS',publication='VERIFIED',publication_verified_commit=first,remote_key_files_verified=verified,file_shards_reassembled_and_hash_verified=True)
 dump(ROOT/'final_integrity.json',final);shutil.copy2(ROOT/'final_integrity.json',dest/'final_integrity.json');git('add','--',rel+'/final_integrity.json',cwd=wt);git('commit','-m','Verify v6.32b completion publication',cwd=wt);git('push','origin','HEAD:main',cwd=wt);last=git('rev-parse','HEAD',cwd=wt);assert git('ls-remote','origin','refs/heads/main').split()[0]==last;assert git('rev-parse',last+':'+rel+'/final_integrity.json',cwd=wt)==git('hash-object',str(dest/'final_integrity.json'),cwd=wt)
 dump(ROOT/'publication_status.json',dict(status='PUBLISHED',commit=last,first_verified_commit=first,remote_branch='origin/main',url=f'https://github.com/ZhangYangyingzi/GVC-RT/tree/{last}/{rel}',verified_key_files=verified+['final_integrity.json'],ordinary_push=True,worktree=str(wt),timestamp=time.time(),server_only_final_receipt=True));print('PUBLISHED',last,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  failure=traceback.format_exc();dump(ROOT/'publication_status.json',dict(status='FAIL',error=failure,timestamp=time.time()))
  if (ROOT/'computation_integrity.json').exists():
   final=dict(load(ROOT/'computation_integrity.json'),status='PUBLICATION_FAILED',calculation='COMPLETE',publication='FAIL',publication_error=failure);dump(ROOT/'final_integrity.json',final)
  error(dict(stage='publish',error=failure));raise
