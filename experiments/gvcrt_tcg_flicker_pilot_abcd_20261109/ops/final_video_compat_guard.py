"""Recover only the identified final FFmpeg layout failure, without retraining.

No process is signaled. The existing runner is left untouched. If its final
montage encounters the already reproduced unsupported xstack fill option, this
guard preserves that failure and resumes the unchanged controller once, with
a task-private FFmpeg syntax adapter. Explicit unrelated failures are retained.
"""
from pathlib import Path
import sys
ROOT_PATH=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT_PATH))
from common_s2 import *
import shlex

def held(name):
    try:
        with lock(name):return False
    except BlockingIOError:return True

def main():
    env()
    with lock('ops/final_video_compat_guard.lock'):
        used=False
        while True:
            s=load(ROOT/'status.json')
            r=dict(status='WATCHING',PID=os.getpid(),time=time.time(),task_status=s.get('status'),task_stage=s.get('current_stage'),recovery_used=used,no_processes_signaled=True)
            if s.get('status') in ['COMPLETE','PARTIAL','BLOCKED']:
                r.update(status='STOPPED',reason='Task terminal state; no recovery needed or authorized for this state')
                dump(ROOT/'ops/final_video_compat_guard_status.json',r);return
            if s.get('status')=='FAILED':
                process=ROOT/'audits/process_video_montage.json'
                known=('CHILD_PROCESS_FAILED' in s.get('exception','') and 'video_montage' in s.get('exception','') and process.exists())
                if known:
                    p=load(process);log=Path(p['log']);known=log.exists() and "Option 'fill' not found" in log.read_text(errors='replace')
                if not known or used:
                    r.update(status='STOPPED',reason='Failure is unrelated to the single preverified FFmpeg compatibility repair, or recovery already used')
                    dump(ROOT/'ops/final_video_compat_guard_status.json',r);return
                if held('runner.lock'):
                    dump(ROOT/'ops/final_video_compat_guard_status.json',r);time.sleep(15);continue
                completions=[ROOT/g/'training/completion.json' for g in GROUPS]
                assert all(p.exists() and load(p)['status']=='PASS' for p in completions),'TRAINING_NOT_COMPLETE_NO_COMPAT_RESTART'
                assert not any(held(str(p.relative_to(ROOT))) for p in (ROOT/'locks').glob('*.lock')),'ACTIVE_TASK_WORKER_NO_COMPAT_RESTART'
                from freeze import verify
                assert verify()
                config=load(ROOT/'ops/ffmpeg_compat_config.json');assert sha(config['real_ffmpeg'])==config['real_ffmpeg_SHA256']
                proof=load(ROOT/'preview_step100/manifest.json');assert proof['status']=='COMPLETE' and proof['FFmpeg_compat_script_SHA256']==sha(ROOT/'ops/ffmpeg_compat.py')
                private=ROOT/'ops/bin';private.mkdir(exist_ok=True)
                entry=private/'ffmpeg';body='#!/bin/sh\nexec '+shlex.quote(PYTHON)+' -B '+shlex.quote(str(ROOT/'ops/ffmpeg_compat.py'))+' "$@"\n'
                if entry.exists():assert entry.read_text()==body
                else:entry.write_text(body);entry.chmod(0o755)
                dump(ROOT/'audits/final_montage_failure_before_compat_recovery.json',s)
                environment=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='4,5,6,7',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'),PATH=str(private)+os.pathsep+os.environ['PATH'])
                log=ROOT/'logs'/f'runner_ffmpeg_compat_recovery_{int(time.time())}.log'
                with log.open('ab') as f:
                    child=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'runner.py')],cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                used=True;r.update(status='KNOWN_FFMPEG_FAILURE_RECOVERED',recovery_used=True,replacement_runner_PID=child.pid,log=str(log),wrapper_entry_SHA256=sha(entry),frozen_sources_unchanged=True,training_will_resume_only_committed_completion_states=True)
                dump(ROOT/'audits/final_montage_compat_recovery.json',r)
            dump(ROOT/'ops/final_video_compat_guard_status.json',r);time.sleep(15)

if __name__=='__main__':main()
