"""Persistent A/B scheduler: paired samples, independent stopping, tests after both selections."""
from common_s2 import *
from evaluate import read_csv
import argparse,traceback,signal

def committed(group):
    p=group_dir(group)/'checkpoints/latest.pt'
    if not p.exists():return 0
    import torch
    return torch.load(p,map_location='cpu',weights_only=True)['committed_optimizer_updates']

def snapshot(stage,state='RUNNING',**extra):
    status(stage,state,PID=os.getpid(),runner_running=state=='RUNNING',committed_updates={g:committed(g) for g in GROUPS},**extra)
    if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json').get('status')=='PASS':return
    text=['# Generator Conv-LoRA coverage A/B execution snapshot','UTC: '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        f'Status:{state}; stage:{stage}.', 'A: stage3+head59392 parameters. B: A plus upsample.conv1(512→2048), total135168. No Router. All original/TCG weights frozen.',
        'Both start fresh from strict pilot+B1500 overlay,TCG cumulative6500,source cursor1500. Effective batch4, same four sample keys per update. Seed20261106,lr5e-5.',
        'Separate min_delta0.0005,patience4,validation250,cap3000. No quality-dependent configuration changes.',
        'Existing test videos are reused; no new unseen confirmation-set claim. Final quality evaluation starts after both selections are frozen.',
        'Full test/report/plot/publication acceptance is pending. Historical baseline reuse is separately audited.']
    for g in GROUPS:
        text += [f'{g} committed optimizer updates:{committed(g)}.']
        p=group_dir(g)/'stopping_state.json'
        if p.exists():text+=['```json',json.dumps(load(p),indent=2),'```']
    (ROOT/'REPORT.md').write_text('\n\n'.join(text)+'\n')

def commands(jobs,stage,lease):
    # Each job has its own session so a failure can terminate only this experiment's children.
    if stage!='publication':snapshot(stage)
    children=[]
    for label,cmd,gpus,group in jobs:
        home=ROOT if group is None else group_dir(group);(home/'logs').mkdir(exist_ok=True)
        logfile=home/f'logs/{label}.log';f=logfile.open('a');start=time.time()
        p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,CUDA_DEVICE_ORDER='PCI_BUS_ID'),
            stdout=f,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),),start_new_session=True)
        children.append(dict(p=p,f=f,label=label,group=group,gpus=gpus,log=str(logfile),start=start))
        event('PROCESS_STARTED',stage=stage,label=label,group=group,PID=p.pid,GPUs=gpus,log=str(logfile),command=cmd)
    dump(ROOT/'active_processes.json',dict(stage=stage,runner_PID=os.getpid(),children=[dict(PID=x['p'].pid,label=x['label'],group=x['group'],GPUs=x['gpus'],log=x['log']) for x in children],time=time.time()))
    heartbeat=0
    while any(x['p'].poll() is None for x in children):
        if any(x['p'].poll() is not None and x['p'].returncode!=0 for x in children):
            for x in children:
                if x['p'].poll() is None:
                    try:os.killpg(x['p'].pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for x in children:
                if x['p'].poll() is None:
                    try:x['p'].wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(x['p'].pid,signal.SIGKILL)
            break
        if time.time()-heartbeat>30:
            if stage!='publication':snapshot(stage)
            heartbeat=time.time()
        time.sleep(2)
    for x in children:
        code=x['p'].wait();x['f'].close();elapsed=time.time()-x['start']
        event('PROCESS_EXIT',stage=stage,label=x['label'],group=x['group'],exit_code=code,elapsed_wall_seconds=elapsed)
        if x['group'] is not None and x['label'].startswith('train_'):
            dump(group_dir(x['group'])/f'audits/{x["label"]}.json',dict(status='PASS' if code==0 else 'FAILED',
                elapsed_wall_seconds=elapsed,GPUs=x['gpus'],PID=x['p'].pid,exit_code=code,start_time=x['start'],end_time=time.time()))
    assert all(x['p'].returncode==0 for x in children),('EXPERIMENT_CHILD_FAILED',stage,[(x['label'],x['p'].returncode) for x in children])

def train_job(group,target,mode):
    world=2 if mode=='two_card_parallel' else 4
    gpu=('4,5' if group=='A_control' else '6,7') if world==2 else '4,5,6,7'
    cmd=[PYTHON,'-B','-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node='+str(world),str(ROOT/'train.py'),
        '--group',group,'--until',str(target)]
    return (f'train_{group}_to_{target:04d}',cmd,gpu,group)

def evaluate_group(group,split,step,lease):
    root=eval_dir(split,step,group);cp=group_dir(group)/f'checkpoints/step_{step}.pt'
    if (root/'integrity.json').exists():
        a=load(root/'integrity.json');assert a['status']=='PASS' and a['checkpoint_sha256']==sha(cp)
        from integrity import paired_streams
        paired_streams(split,step,group);return root
    jobs=[]
    for rank in range(4):
        cmd=[PYTHON,'-B','-u',str(ROOT/'evaluate.py'),'--batch',str(root.relative_to(ROOT)),'--split',split,'--candidate','lora',
            '--group',group,'--checkpoint',str(cp),'--worker',str(rank)]
        jobs.append((f'eval_{split}_{group}_{step:04d}_rank{rank}',cmd,'4,5,6,7',group))
    commands(jobs,f'evaluation_{split}_{group}_{step:04d}',lease)
    from integrity import audit_batch,paired_streams
    audit_batch(root,split,cp,'lora',group);paired_streams(split,step,group);return root

def score(path,baseline):
    rows=read_csv(path);F={};B={}
    for q in [0,5,9]:
        a=[float(r['FloLPIPS']) for r in rows if int(r['QP'])==q];b=[float(r['FloLPIPS']) for r in baseline if int(r['QP'])==q]
        assert len(a)==len(b)==32 and all(math.isfinite(v) for v in a+b)
        F[str(q)]=sum(a)/32;B[str(q)]=sum(b)/32;assert B[str(q)]>0
    return sum(F[str(q)]/B[str(q)] for q in [0,5,9])/3,F,B

def completed_status():
    integrity=load(ROOT/'final_integrity.json');assert integrity['status']=='PASS'
    p=ROOT/'status.json'
    if p.exists() and load(p).get('status')=='COMPLETE':return
    result=dict(status='COMPLETE',experiment_complete=True,current_stage='experiment_complete',experiment_directory=str(ROOT),runner_running=False,
        groups=integrity['groups'],common_validation_steps=integrity['common_validation_steps'],final_test_points=integrity['final_points'],
        bpp_hash_frozen_state_checks='PASS',result_paths={n:str(ROOT/n) for n in ['REPORT.md','protocol.json','source_manifest.json','baseline_provenance.json',
            'metrics.csv','paired_deltas_by_dataset_qp.csv','matched_step_comparison.csv','stream_identity_checks.csv','checkpoints_manifest.json','decode_timing.csv',
            'plots/datasets','plots/development_curves.png','plots/DISTS_changes.png','final_integrity.json','publication_status.json']},time=time.time())
    dump(p,result)

def run(lease):
    env();disk();verify_sources()
    if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json').get('status')=='PASS':
        completed_status()
        pub=load(ROOT/'publication_status.json')
        if pub.get('phase')!='results' or pub.get('status')!='PUBLISHED':commands([('publish_results',[PYTHON,'-B','-u',str(ROOT/'publish.py'),'--phase','results'],'4,5,6,7',None)],'publication',lease)
        return
    mode=load(ROOT/'audits/aggregation_comparison.json')['selected_mode']
    from baseline_reuse import reuse
    snapshot('baseline_development_reuse_audit');baseline_root=reuse('development');baseline=read_csv(baseline_root/'metrics.csv')
    init=[train_job(g,0,mode) for g in GROUPS if not (group_dir(g)/'checkpoints/step_0.pt').exists()]
    if mode=='two_card_parallel' and init:commands(init,'fresh_A_B_initialization',lease)
    elif init:
        for job in init:commands([job],'fresh_'+job[3]+'_initialization',lease)
    from checkpoint_state import synchronize_validation_metadata
    for g in GROUPS:
        p=group_dir(g)/'stopping_state.json'
        if not p.exists():
            _,_,B=score(baseline_root/'metrics.csv',baseline)
            dump(p,dict(status='RUNNING',anchor=1.,bad_check_count=0,last_validation_step=0,checks=[],raw_best=None,
                baseline_F_q=B,baseline_metrics_sha256=sha(baseline_root/'metrics.csv'),initialization_engineering_checks='PASS; no full96point candidate initialization validation claimed'))
        synchronize_validation_metadata(g,load(p))
    while True:
        active=[g for g in GROUPS if load(group_dir(g)/'stopping_state.json')['status']=='RUNNING']
        if not active:break
        targets={g:min(3000,load(group_dir(g)/'stopping_state.json')['last_validation_step']+250) for g in active}
        jobs=[train_job(g,targets[g],mode) for g in active if committed(g)<targets[g]]
        if mode=='two_card_parallel' and jobs:commands(jobs,'parallel_training_round',lease)
        elif jobs:
            for job in jobs:commands([job],'four_card_training_'+job[3],lease)
        for g in active:
            target=targets[g];assert committed(g)==target,'UNVALIDATED_CURSOR_MISMATCH'
            out=evaluate_group(g,'development',target,lease);s,F,B=score(out/'metrics.csv',baseline)
            control=load(group_dir(g)/'stopping_state.json');anchor=control['anchor'];good=s<anchor-.0005
            if good:control['anchor']=s;control['bad_check_count']=0
            else:control['bad_check_count']+=1
            cp=group_dir(g)/f'checkpoints/step_{target}.pt';candidate=dict(step=target,score=s,checkpoint=str(cp),checkpoint_sha256=sha(cp))
            if control['raw_best'] is None or s<control['raw_best']['score']:control['raw_best']=candidate
            check=dict(step=target,score=s,F_q=F,anchor_before=anchor,anchor_after=control['anchor'],bad_check_count=control['bad_check_count'],
                threshold_improvement=good,metrics_sha256=sha(out/'metrics.csv'),checkpoint_sha256=sha(cp),time=time.time())
            control['checks'].append(check);control['last_validation_step']=target
            if control['bad_check_count']>=4:control['status']='EARLY_STOPPED_PATIENCE'
            elif target==3000:control['status']='MAX_BUDGET_REACHED'
            dump(group_dir(g)/'stopping_state.json',control);synchronize_validation_metadata(g,control)
            event('VALIDATION_COMPLETE',group=g,validation_check=check)
    for g in GROUPS:
        p=group_dir(g)/'selection.json';control=load(group_dir(g)/'stopping_state.json');best=control['raw_best'];assert best
        if not p.exists():dump(p,dict(status='FROZEN_BEFORE_FINAL_TEST',group=g,best_trained=best,baseline_retained=best['score']>=1.,
            raw_better_than_baseline=best['score']<1.,preset_min_delta_threshold_met=best['score']<.9995,stop_reason=control['status'],
            committed_optimizer_updates=committed(g),test_used_for_selection=False,protocol_sha256=sha(ROOT/'protocol.json'),time=time.time()))
    if not (ROOT/'selection_manifest.json').exists():dump(ROOT/'selection_manifest.json',dict(status='FROZEN_BOTH_BEFORE_FINAL_TEST',
        selections={g:sha(group_dir(g)/'selection.json') for g in GROUPS},time=time.time()))
    snapshot('baseline_final_reuse_audit');reuse('final')
    for g in GROUPS:evaluate_group(g,'final',load(group_dir(g)/'selection.json')['best_trained']['step'],lease)
    commands([('decode_timing',[PYTHON,'-B','-u',str(ROOT/'decode_timing.py')],'4,5,6,7',None)],'decode_timing',lease)
    commands([('finalize',[PYTHON,'-B','-u',str(ROOT/'finalize.py')],'4,5,6,7',None)],'final_integrity_report_plotting',lease)
    assert load(ROOT/'final_integrity.json')['status']=='PASS';completed_status()
    commands([('publish_results',[PYTHON,'-B','-u',str(ROOT/'publish.py'),'--phase','results'],'4,5,6,7',None)],'publication',lease)

def main():
    argparse.ArgumentParser().parse_args();env()
    with lock('runner.lock') as lease:
        dump(ROOT/'runner_identity.json',dict(PID=os.getpid(),proc_start=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
            command=[PYTHON,'-B','-u',str(ROOT/'runner.py')],start_time=time.time()))
        try:run(lease)
        except Exception:
            error=traceback.format_exc();stage=load(ROOT/'active_processes.json').get('stage') if (ROOT/'active_processes.json').exists() else 'preflight'
            event('RUNNER_FAILED',stage=stage,error=error)
            if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json').get('status')=='PASS':
                status(stage,'LOCAL_COMPLETE_PUBLICATION_FAILED',experiment_complete=True,error=error)
            else:snapshot(stage,'FAILED',failure_stage=stage,error=error,resume_command=f'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 {PYTHON} -B -u {ROOT}/runner.py')
            raise

if __name__=='__main__':main()
