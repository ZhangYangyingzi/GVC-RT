"""Single-owner persistent ULong continuation, validation, testing and publication."""
from common_s2 import *
import argparse,signal,traceback

ACTIVE=[]
STAGE='starting'

def snapshot(stage=None,final=False,**extra):
    global STAGE
    if stage:STAGE=stage
    record=dict(status='RUNNING',complete=False,current_stage=STAGE,PID=os.getpid(),time=time.time(),
        experiment_directory=str(ROOT),s_train=1.,s_eval=.5,training_updates=0,absolute_LoRA_optimizer_step=2000,
        processes=[dict(PID=p.pid,command=cmd,log=log) for p,cmd,log,_ in ACTIVE if p.poll() is None])
    if (ROOT/'training_status.json').exists():
        tr=load(ROOT/'training_status.json');record['training_updates']=tr['committed_optimizer_updates'];record['absolute_LoRA_optimizer_step']=2000+record['training_updates']
    record.update(extra)
    if not final:dump(ROOT/'status.json',record)
    dump(ROOT/'heartbeat.json',record)
    if not final:
        (ROOT/'REPORT.md').write_text('# ULong-only A2000 continuation — execution snapshot\n\n'
            f"Snapshot UTC: {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}. Status: RUNNING.\n\n"
            f"Stage: `{STAGE}`. Committed additional updates: {record['training_updates']}; absolute LoRA step: {record['absolute_LoRA_optimizer_step']}.\n\n"
            'This snapshot does not assert completion. Final validation and test metrics are pending.\n')

def jobs(specs,stage,final=False):
    global ACTIVE
    snapshot(stage,final=final);ACTIVE=[]
    for label,args in specs:
        path=ROOT/f'logs/{label}.log';path.parent.mkdir(parents=True,exist_ok=True);handle=path.open('a')
        cmd=[PYTHON,'-B','-u',*map(str,args)]
        proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6,7'),stdout=handle,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=(RUN_LOCK.fileno(),))
        ACTIVE.append((proc,cmd,str(path),handle));event('CHILD_STARTED',stage=stage,PID=proc.pid,command=cmd,log=str(path))
    try:
        last=0
        while True:
            failed=[p for p,_,_,_ in ACTIVE if p.poll() not in (None,0)]
            if failed:raise RuntimeError(f'CHILD_EXIT_{failed[0].returncode}_PID_{failed[0].pid}; logs='+str([x[2] for x in ACTIVE]))
            if all(p.poll() is not None for p,_,_,_ in ACTIVE):break
            if time.time()-last>20:snapshot(final=final);last=time.time()
            time.sleep(2)
    except BaseException:
        # Only child process groups created by this runner are signalled.
        for p,_,_,_ in ACTIVE:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p,_,_,_ in ACTIVE:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        raise
    finally:
        for _,_,_,h in ACTIVE:h.close()
        ACTIVE=[]

def one(label,script,*args,final=False):jobs([(label,[ROOT/script,*args])],label,final)

def torchrun(until):
    jobs([(f'train_until_{until:04d}',['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4',ROOT/'train.py',
        '--group','A_control','--until',str(until)])],f'training_to_additional_{until}')

def evaluation(batch,split,model,cp=None):
    specs=[]
    for rank in range(4):
        args=[ROOT/'eval_worker.py','--rank',str(rank),'--batch',batch,'--split',split,'--model',model]
        if cp:args+=['--checkpoint',str(cp)]
        specs.append((f'{batch.replace("/","_")}_rank{rank}',args))
    jobs(specs,f'evaluation_{batch}')
    from evaluation_support import audit_batch
    return audit_batch(batch,split,model,cp)

def engineering():
    one('GPU_preflight','resource_preflight.py')
    from reuse import check_initial_probes,reuse_development_reference,check_MCL_probe
    if not (ROOT/'audits/A2000_s05_initial_reproduction.json').exists():
        specs=[]
        for rank,q in enumerate([0,5,9]):
            specs.append((f'initial_A2000_QP{q}',[ROOT/'eval_worker.py','--rank',rank,'--batch','engineering/A2000_s05_dev',
                '--model','lora','--checkpoint',INIT,'--probe-dev','--probe-qp',q]))
        jobs(specs,'initial_A2000_s05_reproduction');check_initial_probes()
    assert load(ROOT/'audits/A2000_s05_initial_reproduction.json')['status']=='PASS'
    reuse_development_reference()
    if not (ROOT/'audits/MCL_real_IP_probe.json').exists():
        specs=[]
        for rank,model in enumerate(['baseline','original','lora']):
            args=[ROOT/'eval_worker.py','--rank',rank,'--batch',f'engineering/MCL_probe_{model}','--split','final','--model',model,'--probe-mcl']
            if model=='lora':args+=['--checkpoint',INIT]
            specs.append((f'MCL_real_IP_probe_{model}',args))
        jobs(specs,'MCL_real_IP_no_quality_probe');check_MCL_probe()
    assert load(ROOT/'audits/MCL_real_IP_probe.json')['status']=='PASS'
    if not (ROOT/'audits/aggregation_comparison.json').exists():one('aggregation_checks','check_aggregation.py')
    if not (ROOT/'audits/loss_components.json').exists():one('loss_gradient_checks','engineering_gradients.py')
    for name in ['aggregation_comparison','loss_components','runtime_selection','source_isolation']:
        assert load(ROOT/f'audits/{name}.json')['status']=='PASS'
    dump(ROOT/'audits/engineering_complete.json',dict(status='PASS',formal_updates_advanced_by_engineering=0,
        A2000_optimizer_RNG_restored=True,trainable_parameters=59392,training_gain=1.,evaluation_gain=.5,time=time.time()))

def validation(control,step):
    from evaluate import METRICS
    from checkpoint_state import synchronize_validation_metadata
    cp=ROOT/f'checkpoints/step_{step}.pt';assert cp.exists()
    batch=str(eval_dir('development',step,'A_control').relative_to(ROOT))
    rows=evaluation(batch,'development','lora',cp)
    from evaluate import read_csv
    ref=read_csv(eval_dir('development',group='baseline')/'metrics.csv')
    ratios={};means={}
    for q in [0,5,9]:
        r=[x for x in rows if int(x['QP'])==q];b=[x for x in ref if int(x['QP'])==q];assert len(r)==len(b)==32
        means[str(q)]={m:sum(float(x[m]) for x in r)/32 for m in METRICS}
        denom={m:sum(float(x[m]) for x in b)/32 for m in ['FloLPIPS','DISTS']};assert all(v>0 and math.isfinite(v) for v in denom.values())
        ratios[str(q)]={m:means[str(q)][m]/denom[m] for m in denom}
    F={q:x['FloLPIPS'] for q,x in ratios.items()};D={q:x['DISTS'] for q,x in ratios.items()};score=sum(F.values())/3
    qualified=all(d<=1. for d in D.values()) and score<1.
    assert control['last_validation_step']==step-250
    anchor=control['anchor'];improved=qualified and score<anchor-.0005
    item=dict(step=step,absolute_LoRA_optimizer_step=2000+step,S=score,F_q=F,D_q=D,qualified=qualified,
        anchor_before=anchor,anchor_after=score if improved else anchor,bad_check_count=0 if improved else control['bad_check_count']+1,
        meaningful_improvement=improved,means_by_qp=means,checkpoint_SHA256=sha(cp),time=time.time())
    control['checks'].append(item);control.update(last_validation_step=step,anchor=item['anchor_after'],bad_check_count=item['bad_check_count'])
    for field,eligible in [('qualified_best',qualified),('raw_best_trained',True)]:
        if eligible and (control[field] is None or score<control[field]['S']):control[field]=dict(step=step,S=score)
    if score<control['raw_best_including_parent']['S']:control['raw_best_including_parent']=dict(step=step,S=score)
    control['status']='EARLY_STOPPED_PATIENCE' if control['bad_check_count']>=4 else ('MAX_BUDGET_REACHED' if step==3000 else 'RUNNING')
    dump(ROOT/'stopping_state.json',control);synchronize_validation_metadata('A_control',control)
    event('DEVELOPMENT_CHECK_COMMITTED',validation=item)

def freeze_selection(control):
    best=control['qualified_best'];step=best['step'] if best else 0;cp=ROOT/f'checkpoints/step_{step}.pt' if step else INIT
    value=dict(status='FROZEN_BEFORE_FINAL_TEST',selected_additional_step=step,checkpoint=str(cp),checkpoint_SHA256=sha(cp),s_eval=.5,
        qualification='all three D_q<=1 and S<1',qualified_best=best,raw_best_trained=control['raw_best_trained'],raw_best_including_parent=control['raw_best_including_parent'],
        final_selection='trained_ULong_only_continuation' if best else 'A2000_s05_retained',test_used_for_selection=False,stop_reason=control['status'],time=time.time())
    path=ROOT/'selection.json'
    if path.exists():
        previous=load(path);assert all(previous[k]==value[k] for k in value if k!='time');value=previous
    else:dump(path,value)
    states=[dict(name='Original_GVC_RT',label='Original GVC-RT',candidate='original',checkpoint=None,batch='evaluation/final_original'),
        dict(name='TCG_B1500',label='TCG B+1500 matched-FP32',candidate='baseline',checkpoint=None,batch='evaluation/final_TCG_B1500'),
        dict(name='A2000_s05',label='TCG+LoRA A2000 s0.5',candidate='lora',checkpoint=str(INIT),batch='evaluation/final_A2000_s05')]
    if step:states.append(dict(name='ULong_continuation_s05',label='TCG+LoRA ULong-only continuation s0.5',candidate='lora',checkpoint=str(cp),batch='evaluation/final_ULong_s05'))
    dump(ROOT/'final_states.json',dict(states=states,deduplicated_parent_retained=not bool(step),selection_SHA256=sha(path),time=time.time()))
    return states

def main():
    global RUN_LOCK
    env();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='4,5,6,7';verify_sources();disk()
    with lock('runner.lock') as RUN_LOCK:
        dump(ROOT/'runner_identity.json',dict(PID=os.getpid(),proc_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],command=sys.argv,time=time.time()))
        snapshot('preparation_publication')
        one('publish_preparation','publish.py','--phase','preparation')
        engineering()
        control_path=ROOT/'stopping_state.json'
        if control_path.exists():control=load(control_path)
        else:
            control=dict(status='RUNNING',last_validation_step=0,anchor=1.,bad_check_count=0,qualified_best=None,raw_best_trained=None,
                raw_best_including_parent=dict(step=0,S=1.),checks=[])
            dump(control_path,control)
        if not (ROOT/'checkpoints/latest.pt').exists():torchrun(0)
        while control['status']=='RUNNING':
            target=control['last_validation_step']+250;assert target<=3000
            import torch
            latest=torch.load(ROOT/'checkpoints/latest.pt',map_location='cpu',weights_only=True);done=latest['committed_optimizer_updates']
            assert control['last_validation_step']<=done<=target
            if done<2:torchrun(2);done=2
            if done<target:torchrun(target)
            validation(control,target)
        states=freeze_selection(control)
        from integrity import training_integrity
        train_audit,_=training_integrity();dump(ROOT/'audits/training_integrity.json',train_audit)
        from reuse import reuse_TCG_legacy_final
        reuse_TCG_legacy_final()
        # Canonical MCL streams are generated/verified first; all states decode them independently.
        for state in sorted(states,key=lambda s:s['candidate']!='baseline'):
            evaluation(state['batch'],'final',state['candidate'],Path(state['checkpoint']) if state['checkpoint'] else None)
        one('finalize_results','finalize.py')
        one('publish_results','publish.py','--phase','results',final=True)
        result=load(ROOT/'status.json');assert result['complete'] and result['integrity_status']=='PASS'
        event('EXPERIMENT_LOCAL_COMPLETE',publication=load(ROOT/'publication_status.json'))

if __name__=='__main__':
    try:main()
    except BlockingIOError:
        print('EXISTING_RUNNER_LOCK_HELD_NO_DUPLICATE_STARTED',flush=True);sys.exit(2)
    except BaseException as error:
        env();traceback.print_exc();record=dict(status='FAILED',complete=False,current_stage=STAGE,failure_stage=STAGE,error=repr(error),
            PID=os.getpid(),processes_running=False,resume_command=f'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 {PYTHON} -B -u {ROOT}/runner.py',time=time.time())
        dump(ROOT/'status.json',record);dump(ROOT/f'audits/failure_{time.time_ns()}.json',record)
        (ROOT/'REPORT.md').write_text('# Execution failure snapshot\n\n'+json.dumps(record,indent=2)+'\n\nEarlier audits and committed checkpoints are preserved.\n')
        raise
