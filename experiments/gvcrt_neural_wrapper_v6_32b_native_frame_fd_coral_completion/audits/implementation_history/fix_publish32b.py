from pathlib import Path
p=Path('publish32b.py');s=p.read_text().replace("'geometry_workers.json','file_index.json')", "'geometry_workers.json','file_index.json')")
s=s.replace("if any(x in EXCLUDE for x in rel.parts)","if rel.parts[0] in ('text_shards','text_indexes') or any(x in EXCLUDE for x in rel.parts)")
s=s.replace("q.write_bytes(buf);parts.append", "q.write_bytes(buf);local=ROOT/pp;local.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,local);parts.append")
a=s.index(" record=dict(schema='v632b'");b=s.index(" git('fetch','origin','main')",a)
s=s[:a]+''' entries=[dict(kind='file',record=r) for r in index]+[dict(kind='sharded_text',record=r) for r in shards];chunks=[];buf=bytearray();ci=0
 def flush_index():
  nonlocal buf,ci
  if not buf:return
  path=Path('text_indexes')/f'index_{ci:04d}.jsonl';q=export/path;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(buf);local=ROOT/path;local.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,local);chunks.append(dict(relative_path=str(path),bytes=len(buf),sha256=sha(q)));ci+=1;buf=bytearray()
 for e in entries:
  line=(json.dumps(e,separators=(',',':'))+'\\n').encode();assert len(line)<=LIMIT
  if len(buf)+len(line)>LIMIT:flush_index()
  buf.extend(line)
 flush_index()
 record=dict(schema='v632b',files=[],sharded_texts=[],index_parts=chunks,shards_max_bytes=LIMIT,reassembly_verified=True,binary_index='binary_artifacts_index.json');dump(ROOT/'file_index.json',record);shutil.copy2(ROOT/'file_index.json',export/'file_index.json')
''' +s[b:]
# Retry remote advances in new detached worktrees; do not rebase/reset original workspace.
s=s.replace(" git('push','origin','HEAD:main',cwd=wt);first=git('rev-parse','HEAD',cwd=wt);", "\n for attempt in range(4):\n  try:git('push','origin','HEAD:main',cwd=wt);break\n  except RuntimeError as exc:\n   if not any(x in str(exc).lower() for x in ('non-fast-forward','fetch first','rejected')) or attempt==3:raise\n   append(ROOT/'publication/retries.jsonl',dict(error=str(exc),attempt=attempt+1,retained_local_commit=git('rev-parse','HEAD',cwd=wt)))\n   git('fetch','origin','main');base=git('rev-parse','FETCH_HEAD');wt=ROOT/'publication'/f'worktree_retry_{time.time_ns()}';git('worktree','add','--detach','--no-checkout',str(wt),base);git('sparse-checkout','set','--cone','--skip-checks',rel,cwd=wt);git('read-tree','-mu','HEAD',cwd=wt);dest=wt/rel;dest.mkdir(parents=True,exist_ok=True);shutil.copytree(export,dest,dirs_exist_ok=True);git('add','--',rel,cwd=wt);git('commit','-m','Add v6.32b native-frame FD and CORAL completion results',cwd=wt)\n first=git('rev-parse','HEAD',cwd=wt);")
p.write_text(s)
p=Path('reassemble.py');s=p.read_text();s=s.replace("for r in json.loads(idx.read_text())['sharded_texts']:","index=json.loads(idx.read_text());records=index.get('sharded_texts',[])\nfor part in index.get('index_parts',[]):\n b=(root/part['relative_path']).read_bytes();assert len(b)==part['bytes'] and hashlib.sha256(b).hexdigest()==part['sha256']\n for line in b.splitlines():\n  e=json.loads(line)\n  if e['kind']=='sharded_text':records.append(e['record'])\nfor r in records:")
p.write_text(s)
import ast
ast.parse(Path('publish32b.py').read_text());ast.parse(Path('reassemble.py').read_text())
