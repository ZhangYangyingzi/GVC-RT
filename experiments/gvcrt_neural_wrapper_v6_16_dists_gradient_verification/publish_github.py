"""Publish only this new experiment, from a fresh origin/main worktree."""
import shutil,traceback
from io16 import *
MESSAGE='Add v6.16 DISTS gradient verification experiment results'
EXCLUDED={'checkpoints','bitstreams','features','visualizations','decoded_frames','inputs','data_cache','datasets','__pycache__'}
ALLOWED={'.py','.json','.jsonl','.csv','.md','.txt','.log','.png'}
def run(argv,cwd=REPO):return subprocess.check_output(argv,cwd=cwd,text=True,stderr=subprocess.STDOUT,timeout=300).strip()
def main():
    state=load(ROOT/'final_integrity.json');assert state['status'] in ('PASS','NO_FIX_REQUIRED','FAIL')
    runtime=load(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else {}
    runtime.update(status=state['status'],phase='experiment_complete',active={},publication_pending=True)
    dump(ROOT/'pipeline_status.json',runtime)
    relative=ROOT.relative_to(REPO);workbase=REPO.parent/'GVC-RT-publication-worktrees';workbase.mkdir(exist_ok=True)
    files=[p for p in ROOT.rglob('*') if p.is_file() and (p.suffix in ALLOWED or p.name=='.gitignore') and not any(x in EXCLUDED for x in p.relative_to(ROOT).parts) and p.name not in ('publication_status.json','upload_manifest.json')]
    assert files and all(p.stat().st_size<90*1024*1024 for p in files)
    manifest={str(p.relative_to(ROOT)):sha(p) for p in files}
    # Included in the commit; excludes itself to avoid a recursive hash.
    dump(ROOT/'upload_manifest.json',dict(files=manifest,only_experiment_directory=True,large_assets_excluded=True));files=[p for p in files if p.name!='upload_manifest.json']+[ROOT/'upload_manifest.json']
    for attempt in range(3):
        run(['git','fetch','origin','main']);base=run(['git','rev-parse','origin/main']);work=workbase/f'v616_{time.time_ns()}'
        run(['git','worktree','add','--detach',str(work),base]);dest=work/relative
        if dest.exists():raise RuntimeError('Experiment already exists remotely; no overwrite attempted')
        for p in files:
            target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        # Root ignores some logs and numerical outputs: explicitly stage the
        # lightweight allowlist, never git add . or unrelated dependencies.
        paths=[str(relative/p.relative_to(ROOT)) for p in files]
        for offset in range(0,len(paths),150):run(['git','add','-f','--',*paths[offset:offset+150]],work)
        staged=run(['git','diff','--cached','--name-only'],work).splitlines();assert set(staged)==set(paths)
        assert all(Path(p).is_relative_to(relative) for p in staged)
        run(['git','commit','-m',MESSAGE],work);commit=run(['git','rev-parse','HEAD'],work)
        try:push=run(['git','push','origin','HEAD:main'],work)
        except subprocess.CalledProcessError as e:
            remote=run(['git','ls-remote','origin','refs/heads/main']).split()[0]
            if remote!=base and attempt<2:continue
            raise RuntimeError('Ordinary push failed: '+e.output)
        run(['git','fetch','origin','main'],work);run(['git','merge-base','--is-ancestor',commit,'origin/main'],work)
        for p in paths:
            obj=run(['git','rev-parse','origin/main:'+p],work);local=run(['git','hash-object',str(work/p)],work);assert obj==local
        remote=run(['git','ls-remote','origin','refs/heads/main']).split()[0]
        dump(ROOT/'publication_status.json',dict(status='PASS',commit=commit,remote_main=remote,base=base,files=len(paths),all_uploaded_file_blobs_verified=True,worktree=str(work),link='https://github.com/ZhangYangyingzi/GVC-RT/tree/main/'+str(relative),push_output=push,finished_unix=time.time()))
        print('PUBLICATION PASS',commit,flush=True);return
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'publication_status.json',dict(status='FAIL',error=traceback.format_exc()));raise
