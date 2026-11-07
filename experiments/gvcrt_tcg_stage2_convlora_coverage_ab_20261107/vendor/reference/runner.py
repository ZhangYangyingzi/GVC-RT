"""Persistent lock, atomic checkpoints, complete validation, stopping and publication."""
from common_s2 import *
from evaluate import read_csv
import argparse,traceback

def committed():
    p=ROOT/'checkpoints/latest.pt'
    if p.exists():
        import torch
        return torch.load(p,map_location='cpu',weights_only=True)['committed_optimizer_updates']
    return 0

def snapshot(stage,state='RUNNING',**extra):
    status(stage,state,PID=os.getpid(),committed_optimizer_updates=committed(),runner_running=state=='RUNNING',**extra)
    if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json').get('status')=='PASS':return
    text=['# Generator Conv-LoRA execution snapshot',
        'UTC: '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),f'Status:{state}; stage:{stage}; committed optimizer updates:{committed()}.',
        'One LoRA-only line; rank4, alpha/r1,59392 parameters, learning rate5e-5. Original generator/TCG/codec/Bridge and quality networks frozen. No Router.',
        'Baseline is strict original I/P + pilot receiver + B1500 TCG overlay, cumulative frozen TCG6500. It is not original GVC-RT weights.',
        'Source cursor1500, effective batch4 (2UVG+2ULong), inherited loss and VJP4096. Max3000, validation250, patience4, min_delta.001.',
        'Every baseline/candidate point independently decodes the same verified pre-encoded stream. Exact bpp/hash checks precede scoring.',
        'Full acceptance and final test are pending; no completion claim. Historical SHA blocker and explicit user correction are preserved in audits/sha_correction_20261106/.']
    if (ROOT/'stopping_state.json').exists():text+=['```json',json.dumps(load(ROOT/'stopping_state.json'),indent=2),'```']
    (ROOT/'REPORT.md').write_text('\n\n'.join(text)+'\n')

def complete_status():
    s=load(ROOT/'selection.json');p=Path(s['best_trained']['checkpoint'])
    wanted=dict(status='COMPLETE',experiment_complete=True,current_stage='experiment_complete',experiment_directory=str(ROOT),
        committed_optimizer_updates=committed(),best_dev_step=s['best_trained']['step'],best_trained_dev_score=s['best_trained']['score'],
        final_selection=s['final_selection'],stop_reason=s['stop_reason'],completed_test_points=135,expected_test_points=135,
        bpp_hash_check='PASS',checkpoint_path=str(p),checkpoint_sha256=sha(p),runner_running=False,
        result_files={n:str(ROOT/n) for n in ['REPORT.md','protocol.json','metrics.csv','development_metrics.csv','training_history.csv',
            'training_coverage.json','rd_points_by_dataset.csv','stream_identity_checks.csv','selection.json','checkpoints_manifest.json',
            'final_integrity.json','publication_status.json','plots/datasets']},publication_recorded_separately=True)
    current=load(ROOT/'status.json')
    if current.get('status')!='COMPLETE':wanted['time']=time.time();dump(ROOT/'status.json',wanted);dump(ROOT/'run_status.json',wanted)

def commands(jobs,stage,lease):
    children=[]
    if stage=='publication':complete_status()
    else:snapshot(stage)
    for label,cmd in jobs:
        f=(ROOT/f'logs/{label}.log').open('a')
        p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6,7',CUDA_DEVICE_ORDER='PCI_BUS_ID'),stdout=f,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
        children.append((p,f,label));event('PROCESS_STARTED',stage=stage,label=label,PID=p.pid,command=cmd)
    dump(ROOT/'active_processes.json',dict(stage=stage,runner_PID=os.getpid(),children=[dict(PID=p.pid,label=n) for p,f,n in children],time=time.time()))
    heartbeat=0
    while any(p.poll() is None for p,f,label in children):
        if time.time()-heartbeat>30:
            if stage!='publication':snapshot(stage)
            heartbeat=time.time()
        time.sleep(2)
    for p,f,label in children:f.close();event('PROCESS_EXIT',stage=stage,label=label,exit_code=p.returncode)
    assert all(p.returncode==0 for p,f,label in children),('CHILD_PROCESS_FAILURE',stage,[(n,p.returncode) for p,f,n in children])

def train_segment(until,lease,smoke=False):
    name='DDP_engineering_smoke' if smoke else f'train_to_{until:04d}'
    cmd=[PYTHON,'-B','-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4',str(ROOT/'train.py'),'--until',str(until)]
    if smoke:cmd+=['--smoke']
    commands([(name,cmd)],name,lease)

def evaluate_batch(split,lease,step=None,candidate='lora'):
    root=ROOT/'evaluation'/evaluation_id(split,step,candidate)
    cp=None if candidate=='baseline' else ROOT/f'checkpoints/step_{step}.pt'
    if cp is not None:assert cp.exists()
    if (root/'integrity.json').exists():
        a=load(root/'integrity.json');assert a['status']=='PASS' and a['candidate']==candidate and a['checkpoint_sha256']==(None if cp is None else sha(cp));return root
    jobs=[]
    for rank in range(4):
        name=root.name+f'_rank{rank}'
        cmd=[PYTHON,'-B','-u',str(ROOT/'evaluate.py'),'--batch',root.name,'--split',split,'--candidate',candidate,'--worker',str(rank)]
        if cp is not None:cmd+=['--checkpoint',str(cp)]
        jobs.append((name,cmd))
    commands(jobs,root.name,lease)
    from integrity import audit_batch,paired_streams
    audit_batch(root,split,cp,candidate)
    if candidate=='lora':paired_streams(split,step)
    return root

def score(path,baseline):
    rows=read_csv(path);F={};B={}
    for q in [0,5,9]:
        a=[float(r['FloLPIPS']) for r in rows if int(r['QP'])==q];b=[float(r['FloLPIPS']) for r in baseline if int(r['QP'])==q]
        assert len(a)==len(b)==32 and all(math.isfinite(x) for x in a+b)
        F[str(q)]=sum(a)/32;B[str(q)]=sum(b)/32;assert B[str(q)]>0
    return sum(F[str(q)]/B[str(q)] for q in [0,5,9])/3,F,B

def run(lease):
    env();disk();verify_sources()
    if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json').get('status')=='PASS':
        complete_status()
        p=load(ROOT/'publication_status.json')
        if p.get('phase')!='results' or p.get('status')!='PUBLISHED':commands([('publish_results',[PYTHON,'-B','-u',str(ROOT/'publish.py'),'--phase','results'])],'publication',lease)
        return
    for name in ['full_model_load_and_dispatch','real_loss_gradients','RANS_identity']:
        assert load(ROOT/f'audits/{name}.json')['status']=='PASS'
    if not (ROOT/'audits/DDP_smoke.json').exists():train_segment(2,lease,True)
    if not (ROOT/'checkpoints/step_0.pt').exists():train_segment(0,lease)
    base_root=evaluate_batch('development',lease,candidate='baseline');baseline=read_csv(base_root/'metrics.csv')
    if not (ROOT/'stopping_state.json').exists():
        initial=evaluate_batch('development',lease,0);s,F,B=score(initial/'metrics.csv',baseline);assert s==1.,'ZERO_LORA_SCORE_NOT_BASELINE'
        dump(ROOT/'stopping_state.json',dict(status='RUNNING',anchor=1.,bad_check_count=0,checks=[],raw_best=None,last_validation_step=0,
            initialization=dict(score=s,metrics_sha256=sha(initial/'metrics.csv')),baseline_F_q=B,baseline_metrics_sha256=sha(base_root/'metrics.csv')))
    from checkpoint_state import synchronize_validation_metadata
    synchronize_validation_metadata(load(ROOT/'stopping_state.json'))
    while True:
        state=load(ROOT/'stopping_state.json')
        if state['status']!='RUNNING':break
        target=min(3000,state['last_validation_step']+250)
        if committed()<target:train_segment(target,lease)
        assert committed()==target,'COMMITTED_CURSOR_EXCEEDS_UNVALIDATED_NODE'
        out=evaluate_batch('development',lease,target);s,F,B=score(out/'metrics.csv',baseline)
        anchor=state['anchor'];good=s<anchor-.001
        if good:state['anchor']=s;state['bad_check_count']=0
        else:state['bad_check_count']+=1
        cp=ROOT/f'checkpoints/step_{target}.pt';candidate=dict(step=target,score=s,checkpoint=str(cp),checkpoint_sha256=sha(cp))
        if state['raw_best'] is None or s<state['raw_best']['score']:state['raw_best']=candidate
        check=dict(step=target,score=s,F_q=F,anchor_before=anchor,anchor_after=state['anchor'],substantive_improvement=good,
            bad_check_count=state['bad_check_count'],checkpoint_sha256=sha(cp),metrics_sha256=sha(out/'metrics.csv'),time=time.time())
        state['checks'].append(check);state['last_validation_step']=target
        if state['bad_check_count']>=4:state['status']='EARLY_STOPPED_PATIENCE'
        elif target==3000:state['status']='MAX_BUDGET_REACHED'
        dump(ROOT/'stopping_state.json',state);synchronize_validation_metadata(state)
        snapshot('development_validation_complete');event('VALIDATION_COMPLETE',validation_check=check)
    state=load(ROOT/'stopping_state.json')
    if not (ROOT/'selection.json').exists():
        best=state['raw_best'];assert best is not None
        dump(ROOT/'selection.json',dict(status='FROZEN_BEFORE_FINAL_TEST',time=time.time(),baseline_score=1.,best_trained=best,
            raw_better_than_baseline=best['score']<1.,preset_min_delta_threshold_met=best['score']<.999,
            final_selection='trained_LoRA' if best['score']<1. else 'baseline',stop_reason=state['status'],committed_optimizer_updates=committed(),
            final_test_used_for_selection=False,protocol_sha256=sha(ROOT/'protocol.json'),source_manifest_sha256=sha(ROOT/'source_manifest.json')))
    evaluate_batch('final',lease,candidate='baseline')
    evaluate_batch('final',lease,load(ROOT/'selection.json')['best_trained']['step'])
    commands([('finalize',[PYTHON,'-B','-u',str(ROOT/'finalize.py')])],'final_integrity_and_plotting',lease)
    assert load(ROOT/'final_integrity.json')['status']=='PASS';complete_status()
    commands([('publish_results',[PYTHON,'-B','-u',str(ROOT/'publish.py'),'--phase','results'])],'publication',lease)

def main():
    argparse.ArgumentParser().parse_args();env()
    with lock('runner.lock') as lease:
        dump(ROOT/'runner_identity.json',dict(PID=os.getpid(),proc_start=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
            command=[PYTHON,'-B','-u',str(ROOT/'runner.py')],start_time=time.time()))
        try:run(lease)
        except Exception:
            stage=load(ROOT/'active_processes.json').get('stage') if (ROOT/'active_processes.json').exists() else 'preflight'
            error=traceback.format_exc();event('RUNNER_FAILED',stage=stage,error=error)
            if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json').get('status')=='PASS':
                status(stage,'LOCAL_COMPLETE_PUBLICATION_FAILED',experiment_complete=True,committed_optimizer_updates=committed(),error=error)
            else:snapshot(stage,'FAILED',failure_stage=stage,error=error,resume_command=f'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 {PYTHON} -B -u {ROOT}/runner.py')
            raise

if __name__=='__main__':main()
