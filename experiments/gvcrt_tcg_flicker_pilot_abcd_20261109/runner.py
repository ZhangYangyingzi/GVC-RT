"""Durable task controller. Atomic update receipts are never replayed."""
from common_s2 import *
import traceback,threading,signal

CURRENT='starting';CHILDREN={};STOP=threading.Event()

def held(name):
    try:
        with lock(name):return False
    except BlockingIOError:return True

def progress():
    groups={}
    for g in GROUPS:
        formal=ROOT/g/'training/progress.json';done=ROOT/g/'training/completion.json';smoke=ROOT/g/'smoke/progress.json'
        groups[g]=dict(formal=load(formal) if formal.exists() else dict(committed_updates=0),smoke=load(smoke) if smoke.exists() else dict(committed_updates=0),completion=load(done) if done.exists() else None)
    return groups

def heartbeat():
    while not STOP.is_set():
        record=dict(status='RUNNING',complete=False,current_stage=CURRENT,PID=os.getpid(),heartbeat=time.time(),experiment_directory=str(ROOT),groups=progress(),children=dict(CHILDREN),
            resume_command=f'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 {PYTHON} -B -u {ROOT}/runner.py',log=str(ROOT/'logs/runner.log'))
        dump(ROOT/'status.json',record)
        STOP.wait(10)

def stage(name):
    global CURRENT
    CURRENT=name;event('runner_stage',stage=name)

def start(name,args,extra=None):
    env();e=dict(os.environ);e.pop('MODEL_KIND',None)
    if extra:e.update(extra)
    log=ROOT/'logs'/f'{name}_{int(time.time())}.log'
    with log.open('ab') as f:
        p=subprocess.Popen([PYTHON,'-B','-u',*map(str,args)],cwd=ROOT,env=e,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    CHILDREN[name]=dict(pid=p.pid,args=list(map(str,args)),log=str(log),started=time.time())
    dump(ROOT/f'audits/process_{name}.json',CHILDREN[name]);return p

def wait_processes(jobs,strict=True):
    pending=dict(jobs);codes={}
    while pending:
        for name,p in list(pending.items()):
            rc=p.poll()
            if rc is None:continue
            codes[name]=rc;CHILDREN.pop(name,None);pending.pop(name)
        if pending:time.sleep(5)
    if strict:assert not any(codes.values()),('CHILD_PROCESS_FAILED',codes)
    return codes

def call(name,args,extra=None):return wait_processes([(name,start(name,args,extra))])

def cache(kind):
    stage('prepare_'+kind+'_causal_original_codec_and_RAFT_cache')
    while any(held(f'locks/prepare_{kind}_{i}.lock') for i in range(4)):time.sleep(10)
    return wait_processes([(f'cache_{kind}_{i}',start(f'cache_{kind}_{i}',[ROOT/'prepare_cache.py','--kind',kind,'--rank',i],dict(MODEL_KIND='original'))) for i in range(4)])

def await_gpu():
    from resource_preflight import snapshot
    while True:
        r=snapshot()
        if r['status']=='PASS':return
        stage('WAITING_GPU_other_processes');time.sleep(30)

def adopt_or_start_smoke():
    stage('disposable_20_update_functional_training')
    jobs=[];adopted=[]
    for i,g in enumerate(GROUPS):
        p=ROOT/g/'smoke/completion.json'
        if p.exists():assert load(p)['status']=='PASS';continue
        if held(f'locks/{g}_smoke.lock'):adopted.append(g)
        else:jobs.append((f'smoke_{g}',start(f'smoke_{g}',[ROOT/'train.py','--group',g,'--rank',i,'--smoke'])))
    codes=wait_processes(jobs,strict=False)
    while any(held(f'locks/{g}_smoke.lock') for g in adopted):time.sleep(10)
    healthy=[]
    for g in GROUPS:
        p=ROOT/g/'smoke/completion.json'
        if p.exists() and load(p)['status']=='PASS' and load(p)['committed_updates']==20:healthy.append(g)
    assert healthy,'NO_BRANCH_PASSED_DISPOSABLE_FUNCTIONAL_TRAINING'
    return healthy

def frozen_artifacts(healthy):
    from freeze import create,verify
    if (ROOT/'source_manifest.json').exists():verify();return
    # The shared protocol and source hashes are frozen only after disposable
    # checks. Results of calibration remain immutable separate measured input.
    dump(ROOT/'trainable_parameters.json',{g:load(ROOT/f'audits/trainable_{g}.json') for g in healthy})
    dump(ROOT/'baseline_provenance.json',load(ROOT/'audits/formal_initial_provenance_A.json'))
    dump(ROOT/'gradient_audit.json',dict(status='PASS',calibration=load(ROOT/'audits/calibration.json'),engineering={g:load(ROOT/f'audits/engineering_{g}.json') for g in healthy},
        RANS={g:load(ROOT/f'audits/RANS_{g}.json') for g in healthy},checkpoint_reload={g:load(ROOT/f'audits/checkpoint_reload_{g}.json') for g in healthy},smoke_updates={g:load(ROOT/g/'smoke/completion.json')['committed_updates'] for g in healthy}))
    create();verify()

def initial_development(healthy):
    stage('fixed_36_point_baselines_and_step0')
    # One process per device. A2000 and Original are independently measured.
    wait_processes([(f'baseline_{name}',start(f'baseline_{name}',[ROOT/'evaluation.py','--group',name,'--step',0,'--rank',i])) for i,name in enumerate(['Original','A2000','C','D']) if name in ['Original','A2000'] or name in healthy])
    wait_processes([(f'step0_{g}',start(f'step0_{g}',[ROOT/'evaluation.py','--group',g,'--step',0,'--rank',GROUPS.index(g)])) for g in ['A','B'] if g in healthy])

def main():
    env();assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'
    with lock('runner.lock'):
        if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json').get('status')=='PASS':
            from freeze import verify
            verify()
            if load(ROOT/'publication_status.json').get('status')!='PUBLISHED':
                from publish import publish
                publish('results')
            return
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            (ROOT/'README.md').write_text('TCG flicker pilot A/B/C/D. See protocol.json for frozen training/data/stop/screen rules. Run runner.py with CUDA_VISIBLE_DEVICES=4,5,6,7; the controller resumes only committed atomic checkpoints and verified caches. It does not expand the300-update cap or evaluate the final test set. See status.json for live status; REPORT.md is a timestamped factual snapshot.\n')
            if not (ROOT/'plans/index.json').exists():call('prepare',[ROOT/'prepare.py'])
            formal_already_stopped=all((ROOT/g/'training/completion.json').exists() for g in GROUPS)
            await_gpu()
            repair=ROOT/'audits/native_scoring_repair.json'
            if repair.exists() and load(repair)['status']!='PASS':
                stage('repair_native_GT_reference_cache')
                wait_processes([(f'repair_native_{i}',start(f'repair_native_{i}',[ROOT/'repair_native_cache.py','--rank',i])) for i in range(4)])
            if not formal_already_stopped:cache('engineering')
            stage('implementation_engineering_and_lambda_calibration')
            while held('locks/engineering_batch.lock'):time.sleep(10)
            if not (ROOT/'audits/engineering_batch.json').exists():call('engineering_batch',[ROOT/'engineering_batch.py'])
            assert load(ROOT/'audits/calibration.json')['status']=='PASS'
            healthy=adopt_or_start_smoke()
            wait_processes([(f'current_engineering_{g}',start(f'current_engineering_{g}',[ROOT/'engineering.py','--group',g,'--rank',GROUPS.index(g)])) for g in healthy])
            wait_processes([(f'reload_{g}',start(f'reload_{g}',[ROOT/'checkpoint_probe.py','--group',g,'--rank',GROUPS.index(g)])) for g in healthy])
            # All formal models below are reloaded from original A2000, with
            # new optimizer/RNG/cursor; smoke checkpoints are only reload probes.
            if not formal_already_stopped:
                cache('development')
                stage('full_native_64_frame_engineering_scoring')
                wait_processes([(f'native_probe_{g}',start(f'native_probe_{g}',[ROOT/'evaluation.py','--group',g,'--step',0,'--rank',i,'--index',0,'--probe-native'])) for i,g in enumerate(['Original','A2000','C','D'])])
            repair=ROOT/'audits/native_scoring_repair.json'
            if repair.exists() and not formal_already_stopped:
                item=load(repair);item.update(status='PASS',strict_endpoint_audit=str(ROOT/'audits/historical_A2000_s05_native_endpoint.json'),strict_endpoint_audit_SHA256=sha(ROOT/'audits/historical_A2000_s05_native_endpoint.json'),completed=time.time());dump(repair,item)
            stage('freeze_formal_execution_sources')
            frozen_artifacts(healthy)
            from summarize import report
            report()
            from publish import publish
            publish('preparation') # Publication failure never discards local work.
            if not formal_already_stopped:
                await_gpu();cache('training')
                initial_development(healthy)
            stage('formal_four_independent_branches_with_100_step_validation')
            codes=wait_processes([(f'formal_{g}',start(f'formal_{g}',[ROOT/'train.py','--group',g,'--rank',GROUPS.index(g)])) for g in healthy],strict=False)
            completed=[g for g in healthy if (ROOT/g/'training/completion.json').exists()]
            dump(ROOT/'audits/formal_branch_process_exits.json',dict(exit_codes=codes,completed=completed,time=time.time()))
            stage('summaries_and_predefined_stop_verification')
            from summarize import report,plots
            report();plots()
            if len(completed)==4:
                # Training tensors are task-private and reproducible; retain
                # their hash receipts but release bytes before128-frame assets.
                removed=[]
                from cache_codec import cache_paths
                for r in plan_rows():
                    key,p,rp=cache_paths(r,r['external_qp'],True)
                    if p.exists():removed.append(dict(file=p.name,SHA256=sha(p),bytes=p.stat().st_size));p.unlink()
                dump(ROOT/'audits/released_rebuildable_training_cache.json',dict(only_own_derived_tensors=True,all_formal_branches_stopped=True,files=removed,time=time.time()))
                cache('long');stage('preselected_128_frame_synchronized_videos')
                for batch in [['GT','Original','A2000','A'],['B','C','D']]:
                    wait_processes([(f'video_{g}',start(f'video_{g}',[ROOT/'videos.py','--group',g,'--rank',i])) for i,g in enumerate(batch)])
                call('video_montage',[ROOT/'videos.py','--montage'])
                stage('final_integrity')
                from integrity import check
                integrity=check()
            else:integrity=dict(status='PARTIAL',complete=False,errors=['One or more branches failed; completed branches retained'],completed_branches=completed);dump(ROOT/'final_integrity.json',integrity)
            report();plots()
            stage('final_GitHub_publication')
            STOP.set();thread.join(timeout=12)
            final=dict(status='COMPLETE' if integrity['status']=='PASS' else 'PARTIAL',complete=integrity['status']=='PASS',current_stage='local_experiment_finished',experiment_directory=str(ROOT),
                groups=progress(),final_integrity_status=integrity['status'],training_updates_capped_at=300,final_test_points=0,report_path=str(ROOT/'REPORT.md'),time=time.time())
            dump(ROOT/'status.json',final);published=publish('results')
            # Final on-disk status distinguishes local execution from upload.
            final['publication_status']='PUBLISHED' if published else 'FAILED';final['current_stage']='finished';dump(ROOT/'status.json',final)
            event('runner_finished',status=final['status'],publication=final['publication_status'])
        except Exception as error:
            STOP.set();thread.join(timeout=12)
            failure=dict(status='FAILED',complete=False,current_stage=CURRENT,exception=repr(error),traceback=traceback.format_exc(),groups=progress(),time=time.time(),
                resume_command=f'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 {PYTHON} -B -u {ROOT}/runner.py')
            dump(ROOT/f'audits/runner_failure_{int(time.time())}.json',failure);dump(ROOT/'status.json',failure)
            try:
                from summarize import report
                report()
            except Exception:pass
            raise

if __name__=='__main__':main()
