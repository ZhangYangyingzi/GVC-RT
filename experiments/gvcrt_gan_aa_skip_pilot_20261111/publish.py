"""Publish task-only reviewable code/audits/scalars/plots; never weights or data."""
from common_s2 import *
import argparse,traceback


def git(*args,check=True):
    return subprocess.run(['git',*args],cwd=WORKTREE,text=True,capture_output=True,check=check)


def main(interim=False):
    env()
    with lock('locks/publication.lock'):
        subprocess.run([PYTHON,'results.py'],cwd=ROOT,check=True)
        assert git('branch','--show-current').stdout.strip()==BRANCH
        excluded={'cache','tmp','locks','checkpoints','__pycache__','.git','training','logs','media','runtime_dependencies'}
        paths=[]
        for p in ROOT.rglob('*'):
            rel=p.relative_to(ROOT)
            if not p.is_file() or set(rel.parts)&excluded:continue
            if p.suffix in ['.pt','.pth','.so','.bin','.mkv','.mp4','.pyc','.html']:continue
            if p.name in ['afldm_cvpr2025_paper.txt']:continue
            if p.stat().st_size>16*2**20:continue
            paths.append(str(p.relative_to(WORKTREE)))
        pathspec=ROOT/'tmp/publication_pathspec';pathspec.write_bytes(b''.join(p.encode()+b'\0' for p in paths))
        git('add','--pathspec-from-file='+str(pathspec),'--pathspec-file-nul')
        staged=git('diff','--cached','--name-only').stdout.splitlines();prefix=str(ROOT.relative_to(WORKTREE))+'/'
        assert all(n.startswith(prefix) for n in staged)
        if staged:
            result=git('commit','-m','Execute B300 GAN x local antialias skip pilot: code, audits and current facts')
            (ROOT/'logs/publication.log').open('a').write(result.stdout+result.stderr)
        commit=git('rev-parse','HEAD').stdout.strip()
        result=git('push','origin',f'HEAD:refs/heads/{BRANCH}',check=False)
        (ROOT/'logs/publication.log').open('a').write(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('git push failed: '+result.stderr[-2000:])
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).stdout.split()[0];assert remote==commit
        record=dict(status='REMOTE_VERIFIED',branch=BRANCH,commit=commit,remote_commit=remote,remote_verified=True,commit_URL=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{commit}',interim=interim,uploaded_files=len(paths),excluded_large_artifacts='cache/, data, checkpoints/, *.pt/*.pth, media/ videos, runtime binaries',time=time.time())
        dump(ROOT/'publication_status.json',record)
        subprocess.run([PYTHON,'results.py'],cwd=ROOT,check=True)
        print(json.dumps(record),flush=True)


if __name__=='__main__':
    env();p=argparse.ArgumentParser();p.add_argument('--interim',action='store_true');a=p.parse_args()
    try:main(a.interim)
    except Exception as error:
        dump(ROOT/'publication_status.json',dict(status='FAILED',branch=BRANCH,error=repr(error),traceback=traceback.format_exc(),time=time.time()))
        if not a.interim:raise
