"""Publish only this completed experiment via a fresh origin/main worktree."""
import shutil,traceback
from mixed_io import *
REMOTE='origin';MESSAGE='Add v6.15 mixed-domain replay experiment results'
EXCLUDED={'checkpoints','bitstreams','features','visualizations','__pycache__','data_cache','datasets'}
ALLOWED={'.py','.json','.jsonl','.csv','.md','.txt','.log','.png'}
def run(args,cwd=REPO):
    return subprocess.check_output(args,cwd=cwd,text=True,stderr=subprocess.STDOUT,timeout=300).strip()
def main():
    assert load(ROOT/'final_integrity.json')['status']=='PASS'
    workbase=REPO.parent/'GVC-RT-publication-worktrees';workbase.mkdir(exist_ok=True)
    relative=ROOT.relative_to(REPO)
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ALLOWED and not any(x in EXCLUDED for x in p.relative_to(ROOT).parts) and p.name!='publication_status.json']
    assert all(p.stat().st_size<90*1024*1024 for p in files)
    for attempt in range(1,4):
        run(['git','fetch',REMOTE,'main']);base=run(['git','rev-parse','origin/main']);work=workbase/f'v615_{time.time_ns()}'
        run(['git','worktree','add','--detach',str(work),base]);dest=work/relative
        if dest.exists():raise RuntimeError('Remote already contains experiment directory; preserve remote and inspect conflict')
        for p in files:
            target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        # Check the executed import closure plus dynamically loaded runtime sources.
        required=load(ROOT/'audits/imported_modules.json')+[
            str(ENGINE/'engine.py'),str(ENGINE/'parallel_utils.py'),
            str(REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py'),
            str(V4/'eval_core.py'),str(V4/'train.py'),str(BASE/'evaluate.py'),str(V62/'fullqp_train.py'),str(V62/'fullqp_evaluate.py')]
        added=[]
        for raw in sorted(set(required)):
            p=Path(raw)
            if p.suffix!='.py' or not p.is_relative_to(REPO) or p.is_relative_to(ROOT):continue
            peer=work/p.relative_to(REPO)
            if peer.exists():
                if sha(peer)!=sha(p):raise RuntimeError('Required executed code differs from origin/main: '+str(p.relative_to(REPO)))
            else:
                peer.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,peer);added.append(str(p.relative_to(REPO)))
        run(['git','add','--',str(relative),*added],work)
        staged=run(['git','diff','--cached','--name-only'],work).splitlines()
        assert staged and all(Path(p).is_relative_to(relative) or p in added for p in staged)
        assert all((not any(x in EXCLUDED for x in Path(p).relative_to(relative).parts) and Path(p).suffix in ALLOWED) if Path(p).is_relative_to(relative) else p.endswith('.py') for p in staged)
        run(['git','commit','-m',MESSAGE],work);commit=run(['git','rev-parse','HEAD'],work)
        try:push=run(['git','push',REMOTE,'HEAD:main'],work)
        except subprocess.CalledProcessError as e:
            latest=run(['git','ls-remote',REMOTE,'refs/heads/main']).split()[0]
            if latest!=base and attempt<3:continue
            raise RuntimeError('Ordinary push failed: '+e.output)
        remote=run(['git','ls-remote',REMOTE,'refs/heads/main']).split()[0]
        run(['git','fetch',REMOTE,'main'],work)
        run(['git','merge-base','--is-ancestor',commit,'origin/main'],work)
        checked={}
        for suffix in ('final_integrity.json','RUNBOOK.md','training_logs/mixed.jsonl','results/ulong/all_qp_summary.csv','results/uvg_holdout/all_qp_summary.csv','results/hevc_b/all_qp_summary.csv'):
            obj=run(['git','rev-parse','origin/main:'+str(relative/ suffix)],work);expected=run(['git','hash-object',str(ROOT/suffix)]);assert obj==expected;checked[suffix]=obj
        dump(ROOT/'publication_status.json',dict(status='PASS',commit=commit,remote_main=remote,base=base,worktree=str(work),files=len(staged),necessary_dependencies_added=added,remote_files_verified=checked,link='https://github.com/ZhangYangyingzi/GVC-RT/tree/main/'+str(relative),push_output=push,finished_unix=time.time()))
        print('PUBLICATION PASS',commit,flush=True);return
if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'publication_status.json',dict(status='FAIL',reason=traceback.format_exc(),unix=time.time()));raise
