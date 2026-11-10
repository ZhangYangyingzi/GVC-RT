"""Explicit task allowlist, normal pushes and fetched-remote core blob checks."""
from common_s2 import *
import argparse,traceback
ROOT_NAMES={'.gitignore','protocol.json','status.json','source_inputs.json','source_manifest.json','initialization_audit.json','trainable_parameters.json','gradient_calibration.json','scheduler_config.json','training_history.csv','development_metrics.csv','development_paired_by_video_qp.csv','development_matched_steps.csv','latent_temporal_diagnostics.csv','equivariance_controls.csv','equivariance_development_controls.csv','edge_strength_by_video_qp.csv','selection.json','training_completion.json','metrics.csv','rd_points_by_dataset.csv','video_manifest.json','checkpoint_manifest.json','final_integrity.json','final_test_status.json','REPORT.md','publication_status.json','task_started.json','visual_ROIs.json'}
CODE_NAMES={'b300_loader.py','cache_codec.py','cache_job.py','calibrate.py','common_s2.py','convlora.py','data_reader.py','eval_data.py','evaluate.py','evaluation.py','evaluation_queue.py','experiment_model.py','fp32_island.py','freeze.py','inherited_reader.py','integrity.py','loss.py','media.py','model_context.py','original_context.py','original_runtime.py','pipeline.py','prepare.py','preflight.py','publication.py','quality.py','results.py','runner.py','selection.py','supervisor.py','temporal.py','train.py','videos.py','test_mechanics.py','verify_checkpoint.py','test_cache_regeneration.py','execution_policy.py','amend_h2.py','verify_h2_launch.py'}
SUBTREES={'audits':{'.json','.csv','.md'},'manifests':{'.json'},'plans':{'.json','.gz'},'receipts':{'.gz'},'runtime_tree':{'.py','.json','.cpp','.h','.cu','.cuh','.md'},'reference_tree':{'.py','.json','.cpp','.h','.cu','.cuh','.md'},'vendor':{'.py','.json','.cpp','.h','.cu','.cuh','.md'}}

def git(args):
    for attempt in range(4):
        try:return subprocess.run(['git',*args],cwd=WORKTREE,check=True,capture_output=True,text=True).stdout.strip()
        except subprocess.CalledProcessError as error:
            network=args[0] in ['push','fetch','ls-remote']
            transient=any(x in (error.stderr or '') for x in ['closed by remote host','Connection reset','Connection timed out','Could not resolve hostname'])
            if not network or not transient or attempt==3:raise
            event('publication_network_retry',operation=args[0],attempt=attempt+1,error=(error.stderr or '')[-500:]);time.sleep(5*(attempt+1))


def allowlist():
    paths=[]
    for n in ROOT_NAMES|CODE_NAMES:
        p=ROOT/n
        if p.is_file() and not p.is_symlink():paths.append(p)
    for name,suffixes in SUBTREES.items():
        for p in (ROOT/name).rglob('*'):
            if p.is_file() and not p.is_symlink() and p.suffix in suffixes and not any(x in p.relative_to(ROOT).parts for x in ['__pycache__','analysis','cache','checkpoints']):paths.append(p)
    for p in (ROOT/'plots').glob('*/**/*.png'):
        if 'per_video' not in p.relative_to(ROOT).parts:paths.append(p)
    for p in [ROOT/'plots/manifest.json',ROOT/'logs/bounded_summary.json']:
        if p.exists():paths.append(p)
    for p in (ROOT/'plots').glob('development_steps_*.png'):paths.append(p)
    for p in (ROOT/'evaluation').glob('*/*/step_*/points/*.json'):paths.append(p)
    for p in (ROOT/'evaluation').glob('*/*/step_*/*/*.csv'):paths.append(p)
    for g in GROUPS:
        for n in ['initial_state.json']:
            p=ROOT/g/'training'/n
            if p.exists():paths.append(p)
    assert all(p.is_relative_to(ROOT) and not p.is_symlink() and p.stat().st_size<45*2**20 and p.suffix not in ['.pt','.pth','.ckpt','.bin','.mp4','.mkv','.log'] for p in paths)
    return sorted({str(p.relative_to(WORKTREE)) for p in paths})

def package():
    for g in GROUPS:
        files=sorted((ROOT/g/'training/receipts').glob('*.json'))
        for i in range(0,len(files),100):
            block=files[i:i+100];dump(ROOT/f'receipts/{g}_{int(block[0].stem):04d}_{int(block[-1].stem):04d}.json.gz',[load(p) for p in block])
    dump(ROOT/'logs/bounded_summary.json',{p.name:p.read_text(errors='replace').splitlines()[-20:] for p in (ROOT/'logs').glob('*.log')})

def publish(final=False):
    env();phase='final_results' if final else 'preparation';log=ROOT/f'logs/publication_{phase}.log'
    with lock('locks/publication.lock',blocking=True):
        try:
            if final:assert load(ROOT/'final_integrity.json')['status']=='PASS' and load(ROOT/'selection.json')['status']=='FROZEN'
            else:
                from results import main as snapshot
                snapshot(True)
            package();assert git(['branch','--show-current'])==BRANCH
            old=git(['ls-remote','--heads','origin','refs/heads/'+BRANCH])
            if old:git(['merge-base','--is-ancestor',old.split()[0],'HEAD'])
            # Never overwrite an unrelated staged change.
            existing=git(['diff','--cached','--name-only']).splitlines()
            assert all(n in allowlist() for n in existing), 'UNRELATED_STAGED_FILES_REQUIRE_INSPECTION'
            record=dict(status='PREPARED',phase=phase,branch=BRANCH,remote_verified=False,core_files_verified=False,time=time.time());dump(ROOT/'publication_status.json',record)
            files=allowlist();dump(ROOT/'audits/publication_allowlist.json',dict(status='PASS',phase=phase,files=files,no_weights_cache_media_credentials=True,time=time.time()));files=allowlist()
            for i in range(0,len(files),100):git(['add','--sparse','--force','--',*files[i:i+100]])
            staged=git(['diff','--cached','--name-only']).splitlines();assert all(n in files for n in staged)
            git(['diff','--cached','--check'])
            if staged:git(['commit','-m',f'Corrected H-series B300: {phase} execution records'])
            commit=git(['rev-parse','HEAD']);git(['push','origin','HEAD:refs/heads/'+BRANCH]);assert git(['ls-remote','--heads','origin','refs/heads/'+BRANCH]).split()[0]==commit
            git(['fetch','--no-tags','origin','refs/heads/'+BRANCH]);fetched=git(['rev-parse','FETCH_HEAD']);assert fetched==commit
            core=['protocol.json','source_manifest.json','initialization_audit.json','trainable_parameters.json','gradient_calibration.json','REPORT.md']
            if final:core+=['selection.json','development_metrics.csv','development_paired_by_video_qp.csv','development_matched_steps.csv','rd_points_by_dataset.csv','video_manifest.json','final_integrity.json']
            verified=[]
            for n in core:
                rel=str((ROOT/n).relative_to(WORKTREE));raw=subprocess.check_output(['git','show',f'{fetched}:{rel}'],cwd=WORKTREE);assert hashlib.sha256(raw).hexdigest()==sha(ROOT/n);verified.append(dict(file=n,SHA256=sha(ROOT/n),git_blob=git(['rev-parse',f'{fetched}:{rel}'])))
            record.update(status='PUBLISHED',core_commit=commit,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{commit}',remote_verified=True,core_files_verified=True,core_files=verified,verified_at=time.time());dump(ROOT/'publication_status.json',record)
            rel=str((ROOT/'publication_status.json').relative_to(WORKTREE));git(['add','--sparse','--force','--',rel]);git(['commit','-m',f'Corrected H-series B300: verified {phase} remote receipt']);receipt=git(['rev-parse','HEAD']);git(['push','origin','HEAD:refs/heads/'+BRANCH]);assert git(['ls-remote','--heads','origin','refs/heads/'+BRANCH]).split()[0]==receipt
            for a in verified:assert git(['rev-parse',receipt+':'+str((ROOT/a['file']).relative_to(WORKTREE))])==a['git_blob']
            record.update(receipt_commit=receipt,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{receipt}');dump(ROOT/'publication_status.json',record)
            dump(ROOT/'publication_remote_receipt.json',dict(status='VERIFIED',phase=phase,branch=BRANCH,commit=receipt,core_commit=commit,commit_url=record['commit_url'],remote_verified=True,core_files_verified=True,time=time.time()))
            log.write_text(json.dumps(record,indent=2)+'\n');return True
        except Exception as e:
            log.write_text(traceback.format_exc()+'\n'+str(getattr(e,'stdout',''))+'\n'+str(getattr(e,'stderr','')))
            dump(ROOT/'publication_status.json',dict(status='FAILED',phase=phase,branch=BRANCH,remote_verified=False,error=repr(e),log=str(log),resume_command=f'{PYTHON} -B {ROOT}/publication.py'+(' --final' if final else ''),time=time.time()));raise
if __name__=='__main__':p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();publish(a.final)
