"""Persistent, locked, resumable engineering -> train/validate -> select -> test -> publish."""
from common_s2 import *
import argparse,traceback,socket
from evaluate import videos,METRICS
from reuse import read_csv

def committed():
    p=ROOT/'training_status.json'
    return load(p)['committed_router_updates'] if p.exists() else 0

def snapshot(stage,state='RUNNING',**extra):
    count=committed();status(stage,state,PID=os.getpid(),committed_router_updates=count,baseline_cumulative_tcg_step=6500,**extra)
    if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json').get('status')=='PASS':return
    rows=[f'# Stage2 Router-only execution record\n',f'Snapshot UTC: {time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}',
        f'Status: {state}; stage: {stage}; committed Router updates: {count}.',
        'Baseline: B+1500, cumulative TCG step6500; only the new 33889-parameter Router is trainable.',
        'Frozen data: ULong pool8470, quarantined source groups236; source-plan cursor starts at1500.',
        'Cap3000; validation every250 updates; FloLPIPS normalized to B+1500; patience4; min_delta0.001.',
        'Codec FP16; training Bridge/generator/Router FP32; evaluation Bridge/generator FP16 and Router FP32.',
        'No final PASS is claimed before full integrity validation.']
    if (ROOT/'stopping_state.json').exists():rows+=['```json',json.dumps(load(ROOT/'stopping_state.json'),indent=2),'```']
    (ROOT/'REPORT.md').write_text('\n\n'.join(rows)+'\n')

def commands(jobs,stage,lease):
    children=[];snapshot(stage)
    for label,cmd in jobs:
        handle=(ROOT/f'logs/{label}.log').open('a')
        p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='4,5,6,7'),
                           stdout=handle,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
        children.append((p,handle,label));event('PROCESS_STARTED',stage=stage,label=label,PID=p.pid,command=cmd)
    dump(ROOT/'active_processes.json',dict(stage=stage,runner_PID=os.getpid(),children=[dict(PID=p.pid,label=label) for p,h,label in children],time=time.time()))
    last=0
    while any(p.poll() is None for p,h,label in children):
        if time.time()-last>30:snapshot(stage);last=time.time()
        time.sleep(2)
    codes=[]
    for p,h,label in children:h.close();codes.append(p.returncode);event('PROCESS_EXIT',stage=stage,label=label,PID=p.pid,exit_code=p.returncode)
    assert not any(codes),(stage,'CHILD_PROCESS_FAILURE',[(label,p.returncode) for p,h,label in children])

def train_segment(target,lease,smoke=False):
    label=('DDP_engineering_smoke' if smoke else f'train_to_{target:04d}')
    cmd=[PYTHON,'-B','-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4',str(ROOT/'train.py'),'--until',str(target)]
    if smoke:cmd+=['--smoke']
    commands([(label,cmd)],label,lease)

def evaluate_batch(step,split,lease):
    name=f'{split}_router_{step:04d}';cp=ROOT/f'checkpoints/step_{step}.pt';assert cp.exists()
    root=ROOT/'evaluation'/name
    if (root/'integrity.json').exists():
        audit=load(root/'integrity.json');assert audit['status']=='PASS' and audit['checkpoint_sha256']==sha(cp);return root
    commands([(name+f'_rank{r}',[PYTHON,'-B','-u',str(ROOT/'evaluate.py'),'--batch',name,'--split',split,'--checkpoint',str(cp),'--worker',str(r)]) for r in range(4)],name,lease)
    points=[load(p) for p in sorted((root/'points').glob('*.json'))];expected={(v['sequence'],q) for v in videos(split) for q in (0,5,9)}
    observed=[(p['result']['sequence'],p['result']['QP']) for p in points];assert len(observed)==len(expected) and set(observed)==expected
    for p in points:
        r=p['result'];key=Path(r['bitstream_file']).stem
        assert p['status']=='PASS' and p['Router_checkpoint_sha256']==sha(cp) and p['baseline_sha256']==BASELINE_SHA and p['fresh_encode'] and p['independent_decode'] and p['TCG_and_Router_reset']
        frames=read_csv(root/f'frames/{key}.csv');trans=read_csv(root/f'transitions/{key}.csv')
        assert len(frames)==64 and len(trans)==63 and sum(int(f['real_bits']) for f in frames)==8*r['real_bytes']
        assert all(math.isfinite(float(r[k])) for k in METRICS)
        for k in ['MS_SSIM','LPIPS','DISTS']:assert abs(sum(float(f[k]) for f in frames)/64-r[k])<1e-10
        for k in ['FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']:assert abs(sum(float(t[k]) for t in trans)/63-r[k])<1e-10
    write_csv(root/'metrics.csv',[p['result'] for p in points]);dump(root/'integrity.json',dict(status='PASS',points=len(points),checkpoint_sha256=sha(cp),Cartesian_product=True,independent_decode=True,finite_metrics=True,aggregation_checked=True))
    return root

def score(path,base):
    rows=read_csv(path);F={};B={}
    for q in [0,5,9]:
        rv=[float(r['FloLPIPS']) for r in rows if int(r['QP'])==q];bv=[float(r['FloLPIPS']) for r in base if int(r['QP'])==q]
        assert len(rv)==len(bv)==32 and all(math.isfinite(x) for x in rv+bv)
        F[str(q)]=sum(rv)/32;B[str(q)]=sum(bv)/32;assert B[str(q)]>0
    return sum(F[str(q)]/B[str(q)] for q in [0,5,9])/3,F,B

def run(lease):
    env();disk();verify_sources()
    # Resume engineering artifacts only after checking their source identity manifest.
    commands([('engineering_driver',[PYTHON,'-B','-u',str(ROOT/'engineering_driver.py')])],'engineering_forward_gradient',lease)
    if not (ROOT/'audits/RANS_identity.json').exists():
        commands([(mode,[PYTHON,'-B','-u',str(ROOT/'engineering.py'),'--mode',mode,'--rank',str(rank)]) for mode,rank in [('rans_reference',0),('rans_enabled',1),('rans_bypass',2)]],'engineering_real_RANS',lease)
        from integrity import audit_rans_probe
        audit_rans_probe()
    if not (ROOT/'audits/DDP_smoke.json').exists():train_segment(2,lease,True)
    if not (ROOT/'audits/reuse_summary.json').exists():commands([('reuse_audit',[PYTHON,'-B','-u',str(ROOT/'reuse.py')])],'reuse_identity_audit',lease)
    if not (ROOT/'checkpoints/step_0.pt').exists():train_segment(0,lease)
    base=read_csv(ROOT/'evaluation/reused/B_development.csv')
    if not (ROOT/'stopping_state.json').exists():
        out=evaluate_batch(0,'development',lease);s,f,b=score(out/'metrics.csv',base)
        state=dict(status='RUNNING',anchor=1.,bad_check_count=0,checks=[],enabled0=dict(score=s,F_q=f,path=str(out/'metrics.csv')),
            baseline_F_q=b,raw_best=dict(kind='baseline_bypass',step=None,score=1.),best_trained=None,last_validation_step=0)
        if s<1:state['raw_best']=dict(kind='enabled0_ablation',step=0,score=s)
        dump(ROOT/'stopping_state.json',state)
    from checkpoint_state import synchronize_validation_metadata
    synchronize_validation_metadata(load(ROOT/'stopping_state.json'))
    while True:
        state=load(ROOT/'stopping_state.json')
        if state['status']!='RUNNING':break
        target=min(3000,state['last_validation_step']+250)
        if committed()<target:train_segment(target,lease)
        assert committed()==target,'TRAINING_CURSOR_EXCEEDS_UNVALIDATED_TARGET'
        out=evaluate_batch(target,'development',lease);s,f,b=score(out/'metrics.csv',base)
        anchor=state['anchor'];substantive=s<anchor-.001
        if substantive:state['anchor']=s;state['bad_check_count']=0
        else:state['bad_check_count']+=1
        candidate=dict(kind='trained_router',step=target,score=s,checkpoint=str(ROOT/f'checkpoints/step_{target}.pt'),checkpoint_sha256=sha(ROOT/f'checkpoints/step_{target}.pt'))
        if state['best_trained'] is None or s<state['best_trained']['score']:state['best_trained']=candidate
        if s<state['raw_best']['score']:state['raw_best']=candidate
        state['checks'].append(dict(step=target,score=s,F_q=f,anchor_before=anchor,anchor_after=state['anchor'],substantive_improvement=substantive,
            bad_check_count=state['bad_check_count'],metrics_sha256=sha(out/'metrics.csv'),checkpoint_sha256=candidate['checkpoint_sha256'],time=time.time()))
        state['last_validation_step']=target
        if state['bad_check_count']>=4:state['status']='EARLY_STOPPED_PATIENCE'
        elif target==3000:state['status']='MAX_BUDGET_REACHED'
        dump(ROOT/'stopping_state.json',state);synchronize_validation_metadata(state)
        snapshot('development_validation_complete');event('VALIDATION_COMPLETE',**state['checks'][-1])
    state=load(ROOT/'stopping_state.json')
    if not (ROOT/'selection.json').exists():
        best=state['best_trained'];assert best is not None
        dump(ROOT/'selection.json',dict(status='FROZEN_BEFORE_FINAL_TEST',time=time.time(),final_selection='trained_router' if best['score']<1 else 'baseline_bypass',
            best_trained_router=best,raw_best=state['raw_best'],baseline_score=1.,stop_reason=state['status'],actual_committed_router_updates=committed(),
            protocol_sha256=sha(ROOT/'protocol.json'),stopping_state_sha256=sha(ROOT/'stopping_state.json'),final_test_used_for_selection=False))
    selection=load(ROOT/'selection.json');evaluate_batch(selection['best_trained_router']['step'],'final',lease)
    commands([('diagnostics',[PYTHON,'-B','-u',str(ROOT/'diagnostics.py')])],'fixed_mask_diagnostics',lease)
    commands([('finalize',[PYTHON,'-B','-u',str(ROOT/'finalize.py')])],'final_integrity_and_plotting',lease)
    commands([('publish',[PYTHON,'-B','-u',str(ROOT/'publish.py')])],'publication',lease)
    snapshot('complete','COMPLETE',integrity_status='PASS',publication_status=load(ROOT/'publication_status.json')['status'])

def main():
    ap=argparse.ArgumentParser();ap.parse_args();env()
    with lock('runner.lock') as lease:
        dump(ROOT/'runner_identity.json',dict(PID=os.getpid(),proc_start=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],command=[PYTHON,'-B','-u',str(ROOT/'runner.py')],start_time=time.time()))
        try:run(lease)
        except Exception:
            current=load(ROOT/'run_status.json').get('current_stage');error=traceback.format_exc();event('RUNNER_FAILED',stage=current,error=error)
            snapshot(current,'FAILED',failure_stage=current,error=error,resume_command=f'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 {PYTHON} -B -u {ROOT}/runner.py');raise
if __name__=='__main__':main()
