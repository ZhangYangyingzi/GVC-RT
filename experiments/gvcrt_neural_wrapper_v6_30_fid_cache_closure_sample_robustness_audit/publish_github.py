"""Publish only v6.30 reviewed assets from a latest-main isolated worktree."""
import shutil,traceback
from io21 import *
MESSAGE='Add v6.30 FID cache closure and sample robustness audit results'
EXCLUDED={'label_cache','weights','dependencies','.git','cache','checkpoints','bitstreams','features','verification','decoded','rgb','decoded_frames','rgb_cache','__pycache__','upload_shards','file_index_parts','upload_manifest_parts'}
SNAPSHOT={}
def snapshot_sha(p):return hashlib.sha256(SNAPSHOT[str(p.relative_to(ROOT))]).hexdigest()
ALLOWED={'.LICENSE','.py','.json','.jsonl','.csv','.md','.txt','.log','.png'}
def run(argv,cwd=REPO):return subprocess.check_output(argv,cwd=cwd,text=True,stderr=subprocess.STDOUT,timeout=600).strip()
def files_for_upload():
    files=[];index=[];limit=524288
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or any(t in EXCLUDED for t in p.relative_to(ROOT).parts) or (p.suffix not in ALLOWED and p.name not in ('.gitignore','LICENSE','WEIGHTS_LICENSE')) or p.name in ('publication_status.json','upload_manifest.json','file_index.json'):continue
        data=p.read_bytes();SNAPSHOT[str(p.relative_to(ROOT))]=data
        if len(data)<=limit or p.suffix=='.png':files.append(p);index.append(dict(path=str(p.relative_to(ROOT)),sha256=snapshot_sha(p),bytes=len(data)));continue
        assert p.suffix!='.png'
        text=data.decode('utf-8');lines=text.splitlines(keepends=True);chunks=[];buf='';size=0
        for line in lines:
            if size+len(line.encode())>limit and buf:chunks.append(buf);buf='';size=0
            assert len(line.encode())<=limit,('oversized text line',str(p))
            buf+=line;size+=len(line.encode())
        if buf:chunks.append(buf)
        entries=[]
        for i,content in enumerate(chunks):
            shard=ROOT/'upload_shards'/p.relative_to(ROOT)/f'part_{i:04d}.txt';shard.parent.mkdir(parents=True,exist_ok=True);shard.write_bytes(content.encode('utf-8'));files.append(shard);SNAPSHOT[str(shard.relative_to(ROOT))]=shard.read_bytes();entries.append(dict(path=str(shard.relative_to(ROOT)),sha256=sha(shard),bytes=shard.stat().st_size))
        assert hashlib.sha256(''.join(chunks).encode()).hexdigest()==snapshot_sha(p)
        index.append(dict(path=str(p.relative_to(ROOT)),sha256=snapshot_sha(p),bytes=len(data),shards=entries,reassembly='concatenate UTF8 shards in listed order'))
    def write_parts(name,entries,mapping=False):
        # Index and manifest themselves obey the same text-file size limit.
        folder=name+'_parts';key='manifest_parts' if mapping else 'index_parts'
        parts=[];chunk=[]
        def encoded(items):
            return json.dumps(dict(files=dict(items) if mapping else items),ensure_ascii=False,indent=2).encode('utf-8')+b'\n'
        def save(items):
            path=ROOT/folder/f'part_{len(parts):04d}.json';path.parent.mkdir(exist_ok=True)
            data=encoded(items);assert len(data)<=limit;path.write_bytes(data)
            SNAPSHOT[str(path.relative_to(ROOT))]=data;files.append(path)
            parts.append(dict(path=str(path.relative_to(ROOT)),bytes=len(data),sha256=snapshot_sha(path)))
        for entry in entries:
            if chunk and len(encoded(chunk+[entry]))>limit:save(chunk);chunk=[]
            chunk.append(entry)
        if chunk:save(chunk)
        header=dict(files={} if mapping else [],entry_count=len(entries),max_shard_bytes=limit)
        header[key]=parts
        path=ROOT/(name+'.json');dump(path,header)
        assert path.stat().st_size<=limit
        SNAPSHOT[str(path.relative_to(ROOT))]=path.read_bytes();files.append(path)
    write_parts('file_index',index)
    # Capture the payload plus the index, prior to adding manifest parts.
    manifest=[(str(p.relative_to(ROOT)),snapshot_sha(p)) for p in files]
    write_parts('upload_manifest',manifest,mapping=True)
    return files
def main():
    phase=os.environ.get('V630_PUBLICATION_PHASE','progress')
    remote=run(['git','remote','get-url','--push','origin']);assert 'ZhangYangyingzi/GVC-RT' in remote and 'semcomm' not in remote
    runtime=load(ROOT/'pipeline_status.json');runtime.update(publication_phase=phase,publication_pending=phase not in ('final_integrity_verified','verification_receipt'));dump(ROOT/'pipeline_status.json',runtime)
    dump(ROOT/'audits/publication_worktree_status.json',dict(original_worktree_git_status=run(['git','status','--short']),target_remote=remote,original_worktree_preserved=True))
    files=files_for_upload();relative=ROOT.relative_to(REPO);workbase=REPO.parent/'GVC-RT-publication-worktrees';workbase.mkdir(exist_ok=True)
    paths=[str(relative/p.relative_to(ROOT)) for p in files]
    previous_record=load(ROOT/'publication_status.json') if (ROOT/'publication_status.json').exists() else {}
    owned_previous=previous_record.get('commit') if previous_record.get('status')=='PASS' else previous_record.get('previous_commit')
    for attempt in range(3):
        run(['git','fetch','origin','main']);base=run(['git','rev-parse','origin/main']);work=workbase/f'v630_{time.time_ns()}';run(['git','worktree','add','--detach',str(work),base]);dest=work/relative
        if dest.exists():
            assert owned_previous, 'remote directory exists without owned publication record'
            previous=owned_previous
            # Refuse foreign changes to this experiment since our previous publication.
            diff=run(['git','diff','--name-only',previous,base,'--',str(relative)],work)
            assert not diff, ('remote experiment was modified by another commit',diff)
        for p in files:
            target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(SNAPSHOT[str(p.relative_to(ROOT))])
        for off in range(0,len(paths),100):run(['git','add','-f','--',*paths[off:off+100]],work)
        assert all(p.startswith(str(relative)+'/') for p in run(['git','diff','--cached','--name-only'],work).splitlines())
        run(['git','commit','--allow-empty','-m',MESSAGE+' ('+phase+')'],work);commit=run(['git','rev-parse','HEAD'],work)
        dump(ROOT/'publication_status.json',dict(status='COMMITTED_PENDING_PUSH',previous_commit=owned_previous,commit=commit,worktree=str(work),base=base))
        try:output=run(['git','push','origin','HEAD:main'],work)
        except subprocess.CalledProcessError as e:
            current=run(['git','ls-remote','origin','refs/heads/main']).split()[0]
            if current!=base and attempt<2:continue
            raise RuntimeError('normal push failed: '+e.output)
        run(['git','fetch','origin','main'],work);run(['git','merge-base','--is-ancestor',commit,'origin/main'],work)
        tree=run(['git','ls-tree','-r','origin/main','--',str(relative)],work);objects={line.split('\t',1)[1]:line.split('\t',1)[0].split()[2] for line in tree.splitlines()}
        for p,name in zip(files,paths):
            b=(work/name).read_bytes();h=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();assert objects[name]==h
        current=run(['git','ls-remote','origin','refs/heads/main']).split()[0]
        dump(ROOT/'publication_status.json',dict(status='PASS',commit=commit,remote_main=current,base=base,files=len(paths),all_remote_blobs_verified=True,worktree=str(work),link='https://github.com/ZhangYangyingzi/GVC-RT/tree/main/'+str(relative),push_output=output,finished_unix=time.time()));print('PUBLICATION PASS',commit,flush=True);return
if __name__=='__main__':
    try:main()
    except Exception:
        old=load(ROOT/'publication_status.json') if (ROOT/'publication_status.json').exists() else {};old.update(status='FAIL',error=traceback.format_exc());dump(ROOT/'publication_status.json',old);raise
