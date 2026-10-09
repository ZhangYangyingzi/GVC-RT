"""Complete an owned publication using an explicit lightweight-file allowlist."""
import shutil,traceback
from mixed_io import *
from publish_github import ALLOWED,EXCLUDED,MESSAGE,run

def main():
    assert load(ROOT/'final_integrity.json')['status']=='PASS'
    prior=load(ROOT/'publication_status.json');assert prior['status']=='PASS'
    owned=prior['commit'];relative=ROOT.relative_to(REPO)
    excluded=EXCLUDED|{'decoded_frames','frame_cache'}
    command([PYTHON,'-B','-u',str(Path(__file__).resolve())])
    for attempt in range(3):
        run(['git','fetch','origin','main']);base=run(['git','rev-parse','origin/main'])
        work=REPO.parent/'GVC-RT-publication-worktrees'/f'v615_complete_{time.time_ns()}'
        run(['git','worktree','add','--detach',str(work),base])
        run(['git','merge-base','--is-ancestor',owned,base],work)
        files=[p for p in ROOT.rglob('*') if p.is_file() and (p.suffix in ALLOWED or p.name=='.gitignore') and not any(x in excluded for x in p.relative_to(ROOT).parts) and p.name!='publication_status.json']
        assert all(p.stat().st_size<90*1024*1024 for p in files)
        for p in files:
            rel=p.relative_to(REPO);dest=work/rel
            if dest.exists():
                # Reject any remote edits after our own first publication.
                old=subprocess.run(['git','rev-parse',owned+':'+str(rel)],cwd=work,text=True,capture_output=True)
                assert old.returncode==0 and old.stdout.strip()==run(['git','hash-object',str(dest)],work),('remote conflict',str(rel))
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        paths=[str(p.relative_to(REPO)) for p in files]
        for start in range(0,len(paths),100):run(['git','add','-f','--',*paths[start:start+100]],work)
        staged=run(['git','diff','--cached','--name-only'],work).splitlines()
        assert set(staged)<=set(paths) and all(Path(p).is_relative_to(relative) for p in staged)
        if staged:run(['git','commit','-m',MESSAGE],work)
        commit=run(['git','rev-parse','HEAD'],work)
        try:run(['git','push','origin','HEAD:main'],work)
        except subprocess.CalledProcessError:
            if attempt<2 and run(['git','ls-remote','origin','refs/heads/main']).split()[0]!=base:continue
            raise
        run(['git','fetch','origin','main'],work);run(['git','merge-base','--is-ancestor',commit,'origin/main'],work)
        remote=run(['git','ls-remote','origin','refs/heads/main']).split()[0]
        listing=run(['git','ls-tree','-r','origin/main','--',str(relative)],work)
        entries={line.split('\t',1)[1]:line.split()[2] for line in listing.splitlines()}
        verified={}
        for p in files:
            rel=str(p.relative_to(REPO));assert entries.get(rel)==run(['git','hash-object',str(p)])
            verified[str(p.relative_to(ROOT))]=entries[rel]
        assert len([p for p in entries if p.endswith('.png')])==12
        dump(ROOT/'publication_status.json',dict(status='PASS',commit=commit,previous_commit=owned,remote_main=remote,worktree=str(work),files_verified=len(verified),explicit_allowlist_staging=True,ignored_lightweight_files_included=True,necessary_dependencies_added=prior['necessary_dependencies_added'],remote_file_blobs=verified,link=prior['link'],finished_unix=time.time()))
        print('COMPLETE PUBLICATION PASS',commit,len(verified),flush=True);return

if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'publication_completion_failure.json',dict(status='FAIL',reason=traceback.format_exc()));raise
