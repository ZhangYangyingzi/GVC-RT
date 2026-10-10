"""Publish only v6.21 reviewed assets from a latest-main isolated worktree."""
import shutil,traceback
from io21 import *
MESSAGE='Add v6.21 reference-length loss and GOP control experiment results'
EXCLUDED={'checkpoints','bitstreams','features','verification','decoded_frames','rgb_cache','__pycache__','upload_shards'}
ALLOWED={'.py','.json','.jsonl','.csv','.md','.txt','.log','.png'}
def run(argv,cwd=REPO):return subprocess.check_output(argv,cwd=cwd,text=True,stderr=subprocess.STDOUT,timeout=600).strip()
def files_for_upload():
    files=[];index=[];limit=4_000_000
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or any(t in EXCLUDED for t in p.relative_to(ROOT).parts) or (p.suffix not in ALLOWED and p.name!='.gitignore') or p.name in ('publication_status.json','upload_manifest.json','file_index.json'):continue
        if p.stat().st_size<=limit:files.append(p);index.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size));continue
        assert p.suffix!='.png'
        text=p.read_bytes().decode('utf-8');lines=text.splitlines(keepends=True);chunks=[];buf='';size=0
        for line in lines:
            if size+len(line.encode())>limit and buf:chunks.append(buf);buf='';size=0
            assert len(line.encode())<=limit,('oversized text line',str(p))
            buf+=line;size+=len(line.encode())
        if buf:chunks.append(buf)
        entries=[]
        for i,content in enumerate(chunks):
            shard=ROOT/'upload_shards'/p.relative_to(ROOT)/f'part_{i:04d}.txt';shard.parent.mkdir(parents=True,exist_ok=True);shard.write_bytes(content.encode('utf-8'));files.append(shard);entries.append(dict(path=str(shard.relative_to(ROOT)),sha256=sha(shard),bytes=shard.stat().st_size))
        assert hashlib.sha256(''.join(chunks).encode()).hexdigest()==sha(p)
        index.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size,shards=entries,reassembly='concatenate UTF8 shards in listed order'))
    dump(ROOT/'file_index.json',dict(files=index,large_text_sharded=True,max_shard_bytes=limit));files.append(ROOT/'file_index.json')
    dump(ROOT/'upload_manifest.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in files},only_new_experiment=True,excluded=sorted(EXCLUDED)));files.append(ROOT/'upload_manifest.json')
    return files
def main():
    assert load(ROOT/'final_integrity.json')['status']=='PASS'
    remote=run(['git','remote','get-url','--push','origin']);assert 'ZhangYangyingzi/GVC-RT' in remote and 'semcomm' not in remote
    runtime=load(ROOT/'pipeline_status.json');runtime.update(status='PASS',phase='experiment_complete',active={},publication_pending=True);dump(ROOT/'pipeline_status.json',runtime)
    dump(ROOT/'audits/publication_worktree_status.json',dict(original_worktree_git_status=run(['git','status','--short']),target_remote=remote,original_worktree_preserved=True))
    files=files_for_upload();relative=ROOT.relative_to(REPO);workbase=REPO.parent/'GVC-RT-publication-worktrees';workbase.mkdir(exist_ok=True)
    paths=[str(relative/p.relative_to(ROOT)) for p in files]
    for attempt in range(3):
        run(['git','fetch','origin','main']);base=run(['git','rev-parse','origin/main']);work=workbase/f'v621_{time.time_ns()}';run(['git','worktree','add','--detach',str(work),base]);dest=work/relative
        if dest.exists():
            # A prior successful push can be verified, but never overwritten.
            stable=[p for p in files if p.name not in ('pipeline_status.json','upload_manifest.json','file_index.json','publication_worktree_status.json','commands.jsonl') and p.suffix!='.log']
            for p in stable:assert (dest/p.relative_to(ROOT)).exists() and sha(dest/p.relative_to(ROOT))==sha(p),('remote experiment differs',str(p))
            commit=run(['git','log','-1','--format=%H',base,'--',str(relative)],work)
            dump(ROOT/'publication_status.json',dict(status='PASS',commit=commit,remote_main=base,resumed_remote_verification=True,link='https://github.com/ZhangYangyingzi/GVC-RT/tree/main/'+str(relative)));return
        for p in files:
            target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        for off in range(0,len(paths),100):run(['git','add','-f','--',*paths[off:off+100]],work)
        assert set(run(['git','diff','--cached','--name-only'],work).splitlines())==set(paths)
        run(['git','commit','-m',MESSAGE],work);commit=run(['git','rev-parse','HEAD'],work)
        dump(ROOT/'publication_status.json',dict(status='COMMITTED_PENDING_PUSH',commit=commit,worktree=str(work),base=base))
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

