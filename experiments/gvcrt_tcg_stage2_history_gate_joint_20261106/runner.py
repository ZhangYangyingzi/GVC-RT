"""Locked resumable single joint line, independent precision baseline, evaluation and publication."""
from common_s2 import *
from evaluate import videos
from reuse import read_csv
import argparse,traceback

def committed():
    p=ROOT/'training_status.json';return load(p)['committed_joint_updates'] if p.exists() else 0

def snapshot(stage,state='RUNNING',**extra):
    status(stage,state,PID=os.getpid(),committed_joint_updates=committed(),baseline_cumulative_tcg_step=6500,**extra)
    if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json').get('status')=='PASS':return
    text=['# Internal history gate joint execution snapshot',f'UTC: {time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}',
        f'Status:{state}; stage:{stage}; committed joint updates:{committed()}.',
        'Baseline B+1500 cumulative TCG6500. New seed20261106 Router M0=.2. Only Router+TCG171649 parameters train.',
        'Gate previous_adapter output before concat; current_adapter retained; no outer residual gate. Router/TCG parameters and internals FP32; output returns to incoming dtype.',
        'Fresh AdamW Router1e-4/TCG1e-5; effective batch4=2UVG+2ULong; source cursor1500; inherited loss and VJP4096.',
        'Cap3000, validation250, patience4, min_delta.001; normalization uses independent matched-FP32 B1500.',
        'Final integrity PASS is not claimed until every required point, plot and assertion has completed.']
    if (ROOT/'stopping_state.json').exists():text+=['```json',json.dumps(load(ROOT/'stopping_state.json'),indent=2),'```']
    (ROOT/'REPORT.md').write_text('\n\n'.join(text)+'\n')

def commands(jobs,stage,lease):
    children=[]
    if stage=='publication':complete_experiment_status()
    else:snapshot(stage)
    for label,cmd in jobs:
        f=(ROOT/f'logs/{label}.log').open('a')
        p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6,7',CUDA_DEVICE_ORDER='PCI_BUS_ID'),
            stdout=f,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
        children.append((p,f,label));event('PROCESS_STARTED',stage=stage,label=label,PID=p.pid,command=cmd)
    dump(ROOT/'active_processes.json',dict(stage=stage,runner_PID=os.getpid(),children=[dict(PID=p.pid,label=label) for p,f,label in children],time=time.time()))
    heartbeat=0
    while any(p.poll() is None for p,f,label in children):
        if time.time()-heartbeat>30:
            if stage!='publication':snapshot(stage)
            heartbeat=time.time()
        time.sleep(2)
    for p,f,label in children:f.close();event('PROCESS_EXIT',stage=stage,label=label,exit_code=p.returncode)
    assert all(p.returncode==0 for p,f,label in children),(stage,'CHILD_PROCESS_FAILURE',[(label,p.returncode) for p,f,label in children])

def complete_experiment_status():
    old=load(ROOT/'run_status.json') if (ROOT/'run_status.json').exists() else {}
    if old.get('status')=='COMPLETE' and old.get('committed_joint_updates')==committed():return
    status('experiment_complete','COMPLETE',experiment_complete=True,committed_joint_updates=committed(),
        publication_status_file='publication_status.json',publication_recorded_separately=True)

def train_segment(until,lease,smoke=False):
    label='DDP_engineering_smoke' if smoke else f'train_joint_to_{until:04d}'
    cmd=[PYTHON,'-B','-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4',str(ROOT/'train.py'),'--until',str(until)]
    if smoke:cmd+=['--smoke']
    commands([(label,cmd)],label,lease)

def evaluate_batch(split,lease,step=None,candidate='joint'):
    name=evaluation_id(split,step,candidate);root=ROOT/'evaluation'/name;cp=None if candidate=='matched_baseline' else ROOT/f'checkpoints/step_{step}.pt'
    if cp is not None:assert cp.exists()
    if (root/'integrity.json').exists():
        a=load(root/'integrity.json');assert a['status']=='PASS' and a['candidate']==candidate and a['checkpoint_sha256']==(None if cp is None else sha(cp));return root
    jobs=[]
    for rank in range(4):
        cmd=[PYTHON,'-B','-u',str(ROOT/'evaluate.py'),'--batch',name,'--split',split,'--candidate',candidate,'--worker',str(rank)]
        if cp is not None:cmd+=['--checkpoint',str(cp)]
        jobs.append((name+f'_rank{rank}',cmd))
    commands(jobs,name,lease)
    from integrity import audit_batch
    audit_batch(root,split,cp,candidate);return root

def score(path,baseline):
    rows=read_csv(path);F={};B={}
    for q in [0,5,9]:
        rv=[float(r['FloLPIPS']) for r in rows if int(r['QP'])==q];bv=[float(r['FloLPIPS']) for r in baseline if int(r['QP'])==q]
        assert len(rv)==len(bv)==32 and all(math.isfinite(x) for x in rv+bv)
        F[str(q)]=sum(rv)/32;B[str(q)]=sum(bv)/32;assert B[str(q)]>0
    return sum(F[str(q)]/B[str(q)] for q in [0,5,9])/3,F,B

def run(lease):
    env();disk();verify_sources()
    if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json').get('status')=='PASS':
        if not ((ROOT/'publication_status.json').exists() and load(ROOT/'publication_status.json').get('phase')=='results' and load(ROOT/'publication_status.json').get('status')=='PUBLISHED'):
            commands([('publish_results',[PYTHON,'-B','-u',str(ROOT/'publish.py'),'--phase','results'])],'publication',lease)
        complete_experiment_status();return
    commands([('engineering_driver',[PYTHON,'-B','-u',str(ROOT/'engineering_driver.py')])],'engineering_validation',lease)
    from integrity import audit_rans_probe
    audit_rans_probe()
    if not (ROOT/'audits/reuse_summary.json').exists():commands([('historical_reuse_audit',[PYTHON,'-B','-u',str(ROOT/'reuse.py')])],'historical_precision_reuse_audit',lease)
    if not (ROOT/'audits/DDP_smoke.json').exists():train_segment(2,lease,True)
    if not (ROOT/'checkpoints/step_0.pt').exists():train_segment(0,lease)
    baseline_root=evaluate_batch('development',lease,candidate='matched_baseline');baseline=read_csv(baseline_root/'metrics.csv')
    from precision_comparison import update
    update()
    if not (ROOT/'stopping_state.json').exists():
        init=evaluate_batch('development',lease,0);s,F,B=score(init/'metrics.csv',baseline)
        dump(ROOT/'stopping_state.json',dict(status='RUNNING',anchor=1.,bad_check_count=0,checks=[],raw_best=None,
            enabled0=dict(score=s,F_q=F,metrics_sha256=sha(init/'metrics.csv')),matched_baseline_F_q=B,
            matched_baseline_metrics_sha256=sha(baseline_root/'metrics.csv'),last_validation_step=0))
    from checkpoint_state import synchronize_validation_metadata
    synchronize_validation_metadata(load(ROOT/'stopping_state.json'))
    while True:
        state=load(ROOT/'stopping_state.json')
        if state['status']!='RUNNING':break
        target=min(3000,state['last_validation_step']+250)
        if committed()<target:train_segment(target,lease)
        assert committed()==target,'TRAINING_CURSOR_EXCEEDS_NEXT_UNVALIDATED_NODE'
        out=evaluate_batch('development',lease,target);s,F,B=score(out/'metrics.csv',baseline)
        anchor=state['anchor'];substantive=s<anchor-.001
        if substantive:state['anchor']=s;state['bad_check_count']=0
        else:state['bad_check_count']+=1
        checkpoint=ROOT/f'checkpoints/step_{target}.pt'
        candidate=dict(step=target,score=s,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
        if state['raw_best'] is None or s<state['raw_best']['score']:state['raw_best']=candidate
        check=dict(step=target,score=s,F_q=F,anchor_before=anchor,anchor_after=state['anchor'],substantive_improvement=substantive,
            bad_check_count=state['bad_check_count'],metrics_sha256=sha(out/'metrics.csv'),checkpoint_sha256=sha(checkpoint),time=time.time())
        state['checks'].append(check);state['last_validation_step']=target
        if state['bad_check_count']>=4:state['status']='EARLY_STOPPED_PATIENCE'
        elif target==3000:state['status']='MAX_BUDGET_REACHED'
        dump(ROOT/'stopping_state.json',state);synchronize_validation_metadata(state)
        snapshot('development_validation_complete');event('VALIDATION_COMPLETE',validation_check=check)
    state=load(ROOT/'stopping_state.json')
    if not (ROOT/'selection.json').exists():
        best=state['raw_best'];assert best is not None
        dump(ROOT/'selection.json',dict(status='FROZEN_BEFORE_FINAL_TEST',time=time.time(),baseline_score=1.,
            best_trained_joint=best,raw_better_than_baseline=best['score']<1.,meaningful_improvement_threshold_met=best['score']<.999,
            final_selection='trained_joint' if best['score']<1. else 'baseline',stop_reason=state['status'],actual_committed_joint_updates=committed(),
            protocol_sha256=sha(ROOT/'protocol.json'),source_manifest_sha256=sha(ROOT/'source_manifest.json'),
            stopping_state_sha256=sha(ROOT/'stopping_state.json'),final_test_used_for_selection=False))
    selection=load(ROOT/'selection.json');evaluate_batch('final',lease,candidate='matched_baseline');update()
    evaluate_batch('final',lease,selection['best_trained_joint']['step'])
    commands([('fixed_diagnostics',[PYTHON,'-B','-u',str(ROOT/'diagnostics.py')])],'fixed_diagnostics',lease)
    commands([('finalize',[PYTHON,'-B','-u',str(ROOT/'finalize.py')])],'final_integrity_and_plotting',lease)
    assert load(ROOT/'final_integrity.json')['status']=='PASS'
    complete_experiment_status()
    commands([('publish_results',[PYTHON,'-B','-u',str(ROOT/'publish.py'),'--phase','results'])],'publication',lease)
    complete_experiment_status()

def main():
    argparse.ArgumentParser().parse_args();env()
    with lock('runner.lock') as lease:
        dump(ROOT/'runner_identity.json',dict(PID=os.getpid(),proc_start=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],command=[PYTHON,'-B','-u',str(ROOT/'runner.py')],start_time=time.time()))
        try:run(lease)
        except Exception:
            active=load(ROOT/'active_processes.json') if (ROOT/'active_processes.json').exists() else {}
            stage=active.get('stage',load(ROOT/'run_status.json').get('current_stage'));error=traceback.format_exc();event('RUNNER_FAILED',stage=stage,error=error)
            if stage=='publication' and (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json')['status']=='PASS':
                status(stage,'LOCAL_COMPLETE_PUBLICATION_FAILED',experiment_complete=True,committed_joint_updates=committed(),failure_stage=stage,error=error)
            else:snapshot(stage,'FAILED',failure_stage=stage,error=error,resume_command=f'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 {PYTHON} -B -u {ROOT}/runner.py')
            raise
if __name__=='__main__':main()
