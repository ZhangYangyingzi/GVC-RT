"""Resumable E/F controller. Stage completion is durable, independent of publish."""
from common_s2 import *
import traceback,threading
CURRENT='准备';STOP=threading.Event();CHILDREN={}

def stage(name):
    global CURRENT
    CURRENT=name;event('stage',stage=name)

def start(name,args,devices='0,1,2,3'):
    log=ROOT/'logs'/f'{name}.log';e=dict(os.environ,CUDA_VISIBLE_DEVICES=devices)
    with log.open('ab') as f:p=subprocess.Popen([PYTHON,'-B','-u',*map(str,args)],cwd=ROOT,env=e,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
    CHILDREN[name]=dict(pid=p.pid,log=str(log),devices=devices);return p

def wait(jobs):
    while any(p.poll() is None for p in jobs.values()):
        for n,p in jobs.items():
            if p.poll() not in [None,0]:raise RuntimeError(('STAGE_CHILD_FAILED',n,p.returncode,str(ROOT/'logs'/f'{n}.log')))
        time.sleep(2)
    for n,p in jobs.items():
        CHILDREN.pop(n,None);assert p.returncode==0,(n,p.returncode)

def heartbeat():
    while not STOP.is_set():
        groups={}
        for g in GROUPS:
            p=ROOT/g/'training/progress.json';groups[g]=load(p) if p.exists() else dict(committed_updates=0)
        cache=ROOT/'cache/progress.json'
        dump(ROOT/'status.json',dict(status='RUNNING',complete=False,current_stage=CURRENT,heartbeat=time.time(),PID=os.getpid(),groups=groups,cache=load(cache) if cache.exists() else None,children=CHILDREN,experiment_directory=str(ROOT),resume_command=resume_command()))
        STOP.wait(10)

def resume_command():return f'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 {PYTHON} -B -u {ROOT}/runner.py'

def main():
    env();assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1,2,3'
    with lock('runner.lock'):
        if not (ROOT/'launch.json').exists():dump(ROOT/'launch.json',dict(time=int(subprocess.check_output(['git','log','-g','--format=%at','-1','refs/heads/codex/tcg-flicker-ef-b300-20261109'],cwd=WORKTREE,text=True).strip()),clock_definition='branch creation reflog timestamp; beginning of task setup',resume_command=resume_command()))
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            # Adopt an existing task producer; never start a second GPU worker.
            try:
                with lock('locks/cache_producer.lock'):needs_producer=True
            except BlockingIOError:needs_producer=False
            if needs_producer:
                p=ROOT/'cache/STOP'
                if p.exists():p.unlink()
                failure=ROOT/'audits/cache_producer_failure.json'
                if failure.exists():failure.replace(ROOT/f'audits/cache_producer_failure_prior_{time.time_ns()}.json')
                start('pipeline',[ROOT/'pipeline.py'])
            # A controller-only refresh adopts already running task workers.
            prior=ROOT/'audits/controller_refresh_before_formal.json'
            if prior.exists() and not (ROOT/'audits/controller_refresh_adopted.json').exists():
                retained=load(prior)['children_retained']
                for n,a in retained.items():
                    if n=='pipeline':continue
                    while Path(f'/proc/{a["pid"]}').exists():time.sleep(1)
                dump(ROOT/'audits/controller_refresh_adopted.json',dict(status='PASS',time=time.time()))
            stage('准备')
            from pipeline import wait_rows
            from baselines import reuse
            reuse()
            # External preparation processes may already have completed these
            # receipts. Locks prevent repeating completed calibration or step0.
            if not (ROOT/'audits/calibration.json').exists():wait({'calibrate':start('calibrate',[ROOT/'calibrate.py'])})
            if not (ROOT/'audits/step0_consistency.json').exists():
                jobs={g:start('step0_'+g,[ROOT/'step0.py','--group',g,'--rank',1 if g=='E' else 3]) for g in GROUPS if not (ROOT/f'audits/step0_{g}.json').exists()};wait(jobs)
                e=load(ROOT/'audits/step0_E.json');f=load(ROOT/'audits/step0_F.json')
                assert e['status']==f['status']=='PASS' and [x['RGB_SHA256'] for x in e['checks']]==[x['RGB_SHA256'] for x in f['checks']]
                dump(ROOT/'audits/step0_consistency.json',dict(status='PASS',points=6,each_group_points=6,E_F_equal=True,exact_parent_B300_RGB_hashes=True))
            if any(not (ROOT/f'audits/engineering_{g}.json').exists() for g in GROUPS):
                wait_rows(plan_rows()[:16]) #16 DISTINCT fixed windows; never1200
            jobs={g:start('smoke_'+g,[ROOT/'train.py','--group',g,'--smoke'],'0,1' if g=='E' else '2,3') for g in GROUPS if not (ROOT/f'audits/engineering_{g}.json').exists()};wait(jobs)
            wait({g:start('reload_'+g,[ROOT/'checkpoint_probe.py','--group',g,'--rank',0 if g=='E' else 2]) for g in GROUPS if not (ROOT/f'audits/checkpoint_reload_{g}.json').exists()})
            # Sources and calibration become immutable before formal step1.
            if not (ROOT/'source_manifest.json').exists():
                dump(ROOT/'source_manifest.json',dict(status='FROZEN',code={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if not any(x in p.parts for x in ['cache','tmp'])},protocol_SHA256=sha(ROOT/'protocol.json'),plan_SHA256=sha(ROOT/'plans/index.json'),B300_SHA256=sha(B300),calibration_SHA256=sha(ROOT/'audits/calibration.json')))
            else:
                for n,h in load(ROOT/'source_manifest.json')['code'].items():assert sha(ROOT/n)==h,('CODE_CHANGED_AFTER_FREEZE',n)
            # Each chunk starts from the last atomic checkpoint, never smoke.
            for target in [100,200,300]:
                stage('训练')
                jobs={}
                for g in GROUPS:
                    if (ROOT/g/'training/completion.json').exists():continue
                    p=ROOT/g/'training/progress.json';step=load(p)['committed_updates'] if p.exists() else 0
                    if step<target:jobs[g]=start('train_'+g,[ROOT/'train.py','--group',g,'--until',target],'0,1' if g=='E' else '2,3')
                wait(jobs);stage('评测')
                # Baseline detail is required for comparison, but never blocks
                # formal step1. Original36-point scalar results remain reused.
                if len(list((ROOT/'baseline_detail').glob('*.json')))<36:wait({f'baseline_detail_{r}':start(f'baseline_detail_{r}',[ROOT/'baselines.py','--rank',r]) for r in [0,1]})
                jobs={}
                for g in GROUPS:
                    if (ROOT/g/'training/completion.json').exists():continue
                    d=ROOT/f'evaluation/{g}/step_{target:04d}/completion.json'
                    if not d.exists():jobs[g]=start('eval_'+g,[ROOT/'evaluation.py','--group',g,'--step',target,'--rank',0 if g=='E' else 2])
                wait(jobs)
                from summarize import report,should_stop
                report()
                for g in GROUPS:
                    if (ROOT/g/'training/completion.json').exists():continue
                    stopped=target==200 and should_stop(g)
                    if stopped or target==300:dump(ROOT/g/'training/completion.json',dict(status='PASS',committed_updates=target,stop_reason='PRESET_BOTH_100_200_CONDITIONS' if stopped else 'MAX_300_ADDITIONAL_UPDATES',final_checkpoint=str(ROOT/g/f'training/checkpoints/step_{target:04d}.pt'),time=time.time()))
            stage('视频');wait({'videos':start('videos',[ROOT/'deliver.py','--videos'])})
            stage('完整性检查')
            from deliver import integrity
            result=integrity();assert result['status']=='PASS'
            from summarize import report
            report();stage('发布')
            from publish import publish
            assert publish('results')
            STOP.set();thread.join(12)
            dump(ROOT/'status.json',dict(status='COMPLETE',complete=True,current_stage='发布完成',experiment_directory=str(ROOT),groups={g:load(ROOT/g/'training/completion.json') for g in GROUPS},publication=load(ROOT/'publication_remote_receipt.json'),time=time.time()))
        except Exception as error:
            STOP.set();thread.join(12)
            failure=dict(status='FAILED',complete=False,current_stage=CURRENT,exception=repr(error),traceback=traceback.format_exc(),children=CHILDREN,resume_command=resume_command(),time=time.time())
            dump(ROOT/f'audits/runner_failure_{time.time_ns()}.json',failure);dump(ROOT/'status.json',failure)
            from summarize import report
            report();raise
if __name__=='__main__':main()
