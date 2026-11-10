"""Resumable stage controller; committed updates, evaluations and publication separate."""
from common_s2 import *
import argparse,traceback
CHILDREN={}

def launch(name,args,gpus='0,1,2,3'):
    log=(ROOT/'logs'/f'{name}.log').open('ab');e=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus)
    p=subprocess.Popen([PYTHON,'-B','-u',*map(str,args)],cwd=ROOT,env=e,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);CHILDREN[name]=(p,log)
    event('stage_child_started',process_name=name,PID=p.pid,GPUs=gpus,args=list(map(str,args)));return p

def ensure_producer():
    if (ROOT/'audits/cache_failure.json').exists():raise RuntimeError(('CACHE_FAILURE',load(ROOT/'audits/cache_failure.json')))
    try:
        with lock('locks/producer.lock'):pass
    except BlockingIOError:return
    launch('producer',['pipeline.py']);time.sleep(1)

def heartbeat(stage):
    ensure_producer();cache=load(ROOT/'cache/progress.json') if (ROOT/'cache/progress.json').exists() else {};groups={}
    for g in GROUPS:
        p=ROOT/g/'training/progress.json';a=load(p) if p.exists() else dict(committed_updates=0,next_cursor=0)
        receipts=[load(p) for p in (ROOT/g/'training/receipts').glob('*.json')]
        a['cumulative_cache_wait_seconds']=sum(r['cache_wait_seconds'] for r in receipts);a['next_window']=plan_rows()[a['next_cursor']]['sample_key'] if a['next_cursor']<1200 else None
        a['first_formal_step']=load(ROOT/f'audits/first_step_{g}.json') if (ROOT/f'audits/first_step_{g}.json').exists() else None;groups[g]=a
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits','-i','0,1,2,3'],text=True).strip().splitlines()
    a=dict(status='RUNNING',complete=False,current_stage=stage,groups=groups,cache=cache,GPUs=gpu,children={n:dict(PID=p.pid,returncode=p.poll()) for n,(p,l) in CHILDREN.items()},time=time.time(),resume_command=f'{PYTHON} -B {ROOT}/supervisor.py --start')
    dump(ROOT/'status.json',a)
    last=ROOT/'readable_status.json'
    if not last.exists() or time.time()-load(last)['time']>=60:dump(last,a);event('heartbeat',stage=stage,committed_updates={g:v['committed_updates'] for g,v in groups.items()},READY=cache.get('READY_training_windows'),cache_bytes=cache.get('cache_bytes'))

def wait(names,stage):
    while True:
        heartbeat(stage)
        failed=[(n,CHILDREN[n][0].poll()) for n in names if CHILDREN[n][0].poll() not in [None,0]]
        if failed:raise RuntimeError(('STAGE_CHILD_FAILED',failed))
        if all(CHILDREN[n][0].poll()==0 for n in names):return
        time.sleep(10)

def audit_phase(script,audit,stage,gpus='0,1,2,3',args=[]):
    if (ROOT/audit).exists():assert load(ROOT/audit)['status']=='PASS';return
    name=stage;launch(name,[script,*args],gpus);wait([name],stage)

def evaluated(models,step,phase,ranks):
    names=[]
    for i,rank in enumerate(ranks):
        name=f'{phase}_{"_".join(models)}_{step}_{i}';launch(name,['evaluation.py','--models',','.join(models),'--step',step,'--phase',phase,'--rank',rank,'--shard',i,'--shards',len(ranks)]);names.append(name)
    return names

def main():
    env()
    with lock('locks/runner.lock'):
        if not (ROOT/'task_started.json').exists():dump(ROOT/'task_started.json',dict(time=load(ROOT/'protocol.json')['time'],source='initial protocol preparation timestamp'))
        ensure_producer()
        audit_phase('initialize.py','initialization_audit.json','initialization');audit_phase('calibrate.py','gradient_calibration.json','calibration');audit_phase('preflight.py','audits/preflight.json','preflight')
        smokes=[]
        for g,gpu in zip(GROUPS,['0,1','2,3']):
            if not (ROOT/f'audits/engineering_{g}.json').exists():name='smoke_'+g;launch(name,['train.py','--group',g,'--smoke'],gpu);smokes.append(name)
        if smokes:wait(smokes,'engineering')
        # Reference development work overlaps first training chunk, not a startup barrier.
        refs=evaluated(['B300_original_reference','B300_noLoRA_step0'],0,'development',[1,3])
        if not (ROOT/'training_completion.json').exists():
            from selection import decide,points,complete
            for step in [50,100,200,300]:
                frozen=ROOT/'selection.json'
                if frozen.exists() and load(frozen)['status']=='FROZEN':break
                names=[]
                for g,gpu in zip(GROUPS,['0,1','2,3']):
                    progress=ROOT/g/'training/progress.json';done=load(progress)['committed_updates'] if progress.exists() else 0
                    if done<step:name=f'train_{g}_{step}';launch(name,['train.py','--group',g,'--until',step],gpu);names.append(name)
                if names:wait(names,'training')
                wait(refs,'reference_development')
                alias_step0()
                jobs=[]
                for g,ranks in zip(GROUPS,[[0,1],[2,3]]):jobs+=evaluated([g],step,'development',ranks)
                wait(jobs,f'development_step_{step}')
                for g in GROUPS:complete(points('development',g,step))
                decision=decide(step)
                if decision['stop']:break
            choice=load(ROOT/'selection.json');assert choice['status']=='FROZEN'
            updates={g:load(ROOT/g/'training/progress.json')['committed_updates'] for g in GROUPS};assert len(set(updates.values()))==1 and next(iter(updates.values()))==choice['last_common_step']
            dump(ROOT/'training_completion.json',dict(status='PASS',updates=updates,stop_reason=choice['stop_reason'],selection_SHA256=sha(ROOT/'selection.json'),time=time.time()))
        else:wait(refs,'reference_development');alias_step0()
        final=evaluated(['Original','B300_original_reference','B300_noLoRA_step0','G0_selected','G1_selected'],0,'final',[0,1,2,3]);wait(final,'final_evaluation')
        audit_phase('results.py','audits/results.json','plots')
        audit_phase('media.py','video_manifest.json','video')
        audit_phase('integrity.py','final_integrity.json','integrity')
        launch('publication_final',['publication.py','--final']);wait(['publication_final'],'publication')
        pub=load(ROOT/'publication_status.json');assert pub['remote_verified'] and pub['phase']=='final_results'
        dump(ROOT/'cache/STOP',dict(time=time.time()))
        dump(ROOT/'status.json',dict(status='COMPLETE',complete=True,current_stage='complete',updates=load(ROOT/'training_completion.json')['updates'],publication=pub,experiment_directory=str(ROOT),time=time.time()))
        event('experiment_COMPLETE',publication=pub)

def alias_step0():
    from selection import points,complete
    complete(points('development','B300_noLoRA_step0',0))
    for g in GROUPS:
        for src in (ROOT/'evaluation/development/B300_noLoRA_step0/step_0000/points').glob('*.json'):
            a=load(src);a['result']['model']=g;a['identity']['model']=g;a['step0_identity_reuse']=dict(path=str(src),SHA256=sha(src),common_initialization_SHA256=sha(ROOT/'checkpoints/B300_noLoRA_step0.pt'),no_training_updates=True)
            dest=ROOT/'evaluation/development'/g/'step_0000/points'/src.name
            if dest.exists():assert load(dest)==a
            else:dump(dest,a)

if __name__=='__main__':
    try:main()
    except Exception as e:
        env();dump(ROOT/'audits/runner_failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
