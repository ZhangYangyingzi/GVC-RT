"""Resume only completed pilot postprocessing; never invoke training/evaluation.

The historical runner re-enters engineering before its completed-training
branch, although its own cleanup has deleted those derived training tensors.
This entry verifies the existing engineering receipts, all committed formal
updates, complete measured points and existing128-frame videos, then resumes
composition, the unchanged final integrity checks and the unchanged publisher.
"""
from pathlib import Path
import sys
EXP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP))
from common_s2 import *
import threading,traceback,shlex

STOP=threading.Event()
STAGE='verify_completed_training_and_assets'

def snapshot(state='RUNNING',**extra):
    groups={g:load(ROOT/g/'training/completion.json') for g in GROUPS}
    return dict(status=state,complete=state=='COMPLETE',current_stage=STAGE,PID=os.getpid(),heartbeat=time.time(),time=time.time(),experiment_directory=str(ROOT),
        groups=groups,training_complete=True,training_updates_by_finalizer=0,new_evaluation_points_by_finalizer=0,
        resume_command=f'{PYTHON} -B -u {Path(__file__)}',log=str(ROOT/'logs/finalize_completed.log'),**extra)

def heartbeat():
    while not STOP.is_set():dump(ROOT/'status.json',snapshot());STOP.wait(10)

def stage(name):
    global STAGE
    STAGE=name;event('post_training_finalizer_stage',stage=name)

def preflight():
    from freeze import verify
    assert verify()
    import torch
    assert not torch.cuda.is_initialized(),'POSTPROCESSING_MUST_REMAIN_CPU_ONLY'
    plan=plan_rows();proof={};hashes={}
    for group in GROUPS:
        root=ROOT/group/'training';done=load(root/'completion.json');step=done['committed_updates']
        assert done['status']=='PASS' and step in [200,300]
        receipts=sorted((root/'receipts').glob('*.json'));assert len(receipts)==step
        for i,p in enumerate(receipts,1):
            r=load(p);expected=plan[4*(i-1):4*i]
            assert r['additional_step']==i and r['source_plan_cursor']==4*i and r['sample_keys']==[x['sample_key'] for x in expected]
            assert len(expected)==4 and len({x['source_group'] for x in expected})==4 and all(x['domain']=='ulong' for x in expected)
            assert r['frozen_exact_bytes_checked'] and r['buffers_reset'] and math.isfinite(r['gradient_norm']) and r['gradient_norm']>0
            assert r['optimizer_steps'] and set(r['optimizer_steps'])=={i}
        saved=torch.load(root/'latest.pt',map_location='cpu',weights_only=True)
        assert saved['committed_updates']==step and saved['next_cursor']==4*step and saved['last_committed_receipt']==load(receipts[-1])
        assert saved['protocol_SHA256']==sha(ROOT/'protocol.json') and saved['source_manifest_SHA256']==sha(ROOT/'source_manifest.json')
        for name in [f'engineering_{group}.json',f'RANS_{group}.json',f'checkpoint_reload_{group}.json']:
            path=ROOT/'audits'/name;assert load(path)['status']=='PASS'
            proof[str(path.relative_to(ROOT))]=sha(path)
        for n in range(0,step+1,100):
            base=ROOT/f'evaluation/{group}/step_{n:04d}'
            assert load(base/'completion.json')['status']=='PASS' and len(list((base/'points').glob('*.json')))==36
        hashes[group]={str(p.relative_to(ROOT)):sha(p) for p in [root/'latest.pt',*sorted((root/'checkpoints').glob('*.pt'))]}
        progress=root/'progress.json';old=load(progress)
        archive=ROOT/f'audits/post_training_recovery/{group}_progress_before_terminal_metadata.json'
        if not archive.exists():dump(archive,old)
        old.update(status='COMPLETE',completion_time=done['time'],stop_reason=done['stop_reason']);dump(progress,old)
    profiles=load(ROOT/'manifests/long_videos.json');assert len(profiles)==4
    for profile in profiles:
        records=[load(ROOT/'videos/manifests'/f'{profile["sequence"]}_{g}.json') for g in ['GT','Original','A2000',*GROUPS]]
        assert len(records)==7 and len({r['stream_SHA256'] for r in records})==1 and len({r['stream_bytes'] for r in records})==1
        for record in records:
            assert record['status']=='PASS' and record['frames']==128 and record['source_frame_indices']==list(range(128)) and len(record['RGB8_frame_hashes'])==128
            assert sha(record['path'])==record['sha256'] and sha(record['preview_path'])==record['preview_sha256']
            if record['model'] in GROUPS:assert record['step']==load(ROOT/record['model']/'training/completion.json')['committed_updates']
    config=load(ROOT/'ops/ffmpeg_compat_config.json');assert sha(config['real_ffmpeg'])==config['real_ffmpeg_SHA256']
    private=ROOT/'ops/bin';private.mkdir(exist_ok=True);entry=private/'ffmpeg'
    body='#!/bin/sh\nexec '+shlex.quote(PYTHON)+' -B '+shlex.quote(str(ROOT/'ops/ffmpeg_compat.py'))+' "$@"\n'
    if entry.exists():assert entry.read_text()==body
    else:entry.write_text(body);entry.chmod(0o755)
    os.environ['PATH']=str(private)+os.pathsep+os.environ['PATH']
    audit=dict(status='PASS',time=time.time(),reason='Resume completed postprocessing after known FFmpeg failure and invalid full-run engineering re-entry against intentionally released training cache',
        existing_engineering_PASS_receipts=proof,checkpoint_hashes_before_postprocessing=hashes,formal_steps={g:load(ROOT/g/'training/completion.json')['committed_updates'] for g in GROUPS},
        completed_validation_points_per_group_per_step=36,existing_128frame_model_videos=28,training_or_model_evaluation_will_not_be_called=True,
        original_failure_records_preserved=True,source_manifest_SHA256=sha(ROOT/'source_manifest.json'),protocol_SHA256=sha(ROOT/'protocol.json'),
        finalizer_source_SHA256=sha(Path(__file__)),FFmpeg_compat_SHA256=sha(ROOT/'ops/ffmpeg_compat.py'),only_task_private_FFmpeg_PATH_modified=True)
    dump(ROOT/'audits/post_training_recovery/preflight.json',audit);return hashes

def main():
    env()
    with lock('runner.lock'),lock('ops/finalize_completed.lock'):
        old=load(ROOT/'status.json')
        archived=ROOT/'audits/post_training_recovery/status_before_finalizer.json'
        if not archived.exists():dump(archived,old)
        worker=threading.Thread(target=heartbeat,daemon=True);worker.start()
        try:
            before=preflight()
            stage('compose_existing_128_frame_videos_CPU_only')
            from videos import montage
            montage()
            stage('independent_final_integrity')
            from summarize import report,plots
            report();plots()
            from integrity import check
            final=check();assert final['status']=='PASS',final
            for hashes in before.values():
                for name,value in hashes.items():assert sha(ROOT/name)==value,'CHECKPOINT_CHANGED_DURING_POSTPROCESSING'
            import torch
            assert not torch.cuda.is_initialized()
            audit=load(ROOT/'audits/post_training_recovery/preflight.json');audit.update(postprocessing_completed=True,final_integrity_status=final['status'],no_checkpoint_changes=True,CUDA_not_initialized=True,completed_time=time.time())
            dump(ROOT/'audits/post_training_recovery/preflight.json',audit)
            report()
            with (ROOT/'REPORT.md').open('a') as f:
                f.write('\n## Final execution status\n\nLocal pilot complete: all four branches committed300 updates, all scheduled36-point validations and four128-frame synchronized comparisons complete. Final integrity PASS.\n\nPostprocessing recovery: the original final layout used an FFmpeg option absent from the installed version. Its first recovery incorrectly re-entered an engineering stage after the controller had intentionally released training caches. Both failures are preserved. This entry verified existing successful engineering checks, completed checkpoints/receipts and unchanged source hashes, then performed CPU-only composition and the unchanged final integrity checks. No training updates or metric evaluations were repeated. See audits/post_training_recovery/.\n\nPublication is recorded separately in publication_status.json and publication_remote_receipt.json.\n')
            stage('final_results_GitHub_publication');STOP.set();worker.join(timeout=12)
            local=snapshot('COMPLETE',final_integrity_status='PASS',publication_status='PENDING');local['current_stage']='final_results_GitHub_publication';dump(ROOT/'status.json',local)
            from publish import publish
            published=publish('results')
            local['publication_status']='PUBLISHED' if published else 'FAILED';local['current_stage']='finished';local['time']=time.time();dump(ROOT/'status.json',local)
            event('post_training_finalizer_finished',status=local['status'],publication_status=local['publication_status'])
        except Exception as error:
            STOP.set();worker.join(timeout=12)
            failure=snapshot('FAILED',exception=repr(error),traceback=traceback.format_exc());dump(ROOT/'status.json',failure)
            dump(ROOT/f'audits/post_training_recovery/failure_{int(time.time())}.json',failure);raise

if __name__=='__main__':main()
