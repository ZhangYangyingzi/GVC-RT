"""Resume-safe v3 queue. Only the selected allowed physical GPUs are used."""
import csv
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

P=Path(__file__).resolve().parent
PYTHON=Path('/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python')
sys.path.insert(0,str(P))
import task


def validate_gate():
    d=json.loads((P/'preflight.json').read_text())
    assert d['status']=='PASS'
    assert d['checks']['identity_A_decoded_state']['status']=='PASS'
    assert task.sha(P/'task.py')==d['checks']['task.py_sha256']
    assert task.sha(P/'ablation.py')==d['checks']['ablation.py_sha256']
    assert task.sha(task.V2/'runtime.py')==d['checks']['v2_runtime_sha256']
    assert task.sha(task.V2/'evaluation_sources.json')==d['checks']['v2_sources_sha256']
    for branch,root in [('A',task.V1),('H11',task.V2)]:
        assert task.sha(root/'server_artifacts'/branch/'step_300.pt')==d['checks']['weights_sha256'][branch]


def gpu_available(gpu):
    try:
        line=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
            '--format=csv,noheader,nounits'],text=True,stderr=subprocess.DEVNULL,timeout=10)
        used={int(a.strip()):int(b.strip()) for a,b in (x.split(',') for x in line.splitlines())}
        return used[gpu]<1024
    except (OSError,subprocess.SubprocessError,KeyError,ValueError):return False


def launch(spec,gpu):
    mode,source,qp,alpha,rate_only=spec
    key=task.key(mode,source,qp,alpha)
    sub='rate_only' if rate_only else 'evaluation'
    result=P/'server_artifacts'/sub/f'{key}.json'
    if result.exists():
        data=json.loads(result.read_text())
        stream=Path(data['stream_path'])
        if data['status']=='PASS' and stream.exists() and task.sha(stream)==data['stream_sha256']:
            assert (data['mode'],data['sequence'],data['external_qp'],data['alpha_index'])==(mode,source,qp,alpha)
            assert data['source_sha256']==task.source_record(source)['source_sha256']
            branch='A' if mode=='A' else 'H11'
            gate=json.loads((P/'preflight.json').read_text())
            assert data['checkpoint_sha256']==gate['checks']['weights_sha256'][branch]
            for suffix,code in [('task_code_sha256','task.py'),('ablation_code_sha256','ablation.py')]:
                if suffix in data:assert data[suffix]==gate['checks'][code+'_sha256']
            return data
    # Wait for the assigned physical card; never preempt another process.
    while not gpu_available(gpu):
        print('WAIT_GPU',gpu,key,flush=True)
        time.sleep(20)
    args=[str(PYTHON),str(P/'task.py'),'--mode',mode,'--source',source,'--qp',str(qp),
          '--alpha-index',str(alpha),'--gpu',str(gpu)]
    if rate_only:args.append('--rate-only')
    log=P/'server_artifacts/logs'/f'{key}_{sub}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as f:
        finished=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,cwd=P.parents[1])
    if finished.returncode:
        raise RuntimeError(f'{key} failed on GPU {gpu}; see {log}')
    assert result.exists()
    data=json.loads(result.read_text())
    assert data['status']=='PASS' and task.sha(Path(data['stream_path']))==data['stream_sha256']
    return data


def run_queue(specs,gpus=(4,5,6,7)):
    work=queue.Queue()
    for s in specs:work.put(s)
    errors=[]; completed=[]
    def worker(gpu):
        while not errors:
            if not gpu_available(gpu):
                if work.empty():return
                time.sleep(20)
                continue
            try:s=work.get_nowait()
            except queue.Empty:return
            try:
                row=launch(s,gpu)
                completed.append((s,row))
                print('PASS',s,'gpu',gpu,'bpp',row['actual_bpp'],flush=True)
            except Exception as exc:
                errors.append(str(exc));print('FAIL',str(exc),flush=True)
            finally:work.task_done()
    threads=[threading.Thread(target=worker,args=(gpu,)) for gpu in gpus]
    for t in threads:t.start()
    for t in threads:t.join()
    if errors:raise RuntimeError(errors)
    assert len(completed)==len(specs)
    return completed


def main():
    validate_gate()
    import reuse_anchors
    for mode in ('A','MAP'):
        for source in ('UVal0','UVal1','ParkScene','Jockey'):
            for qp in (2,7): reuse_anchors.wrap(mode,source,qp)
    for source in ('UVal0','UVal1','ParkScene','Jockey'):
        reuse_anchors.wrap('A',source,9)
    reuse_anchors.wrap('MAP','UVal0',9)
    videos=('UVal0','UVal1','ParkScene','Jockey')
    phase1=[('A',source,qp,2,False) for source in videos for qp in (3,4,5,6)]
    phase1 += [('MAP',source,qp,2,False) for source in videos for qp in (4,6)]
    assert len(phase1)==24
    run_queue(phase1)
    print('PHASE1_COMPLETE',24,flush=True)
    phase2=[(mode,source,qp,2,False) for source in ('UVal1','Jockey') for qp in (2,7)
            for mode in ('MEAN','ROLL','IDENTITY')]
    assert len(phase2)==12
    run_queue(phase2)
    print('PHASE2_ALPHA1_COMPLETE',12,flush=True)
    search=[];selected=[]
    for source in ('UVal1','Jockey'):
        for qp in (2,7):
            target=json.loads((P/'server_artifacts/evaluation'/f'{task.key("MAP",source,qp,2)}.json').read_text())
            for mode in ('MEAN','ROLL'):
                alpha1=json.loads((P/'server_artifacts/evaluation'/f'{task.key(mode,source,qp,2)}.json').read_text())
                candidates={2:alpha1}
                if abs(alpha1['real_total_bytes']-target['real_total_bytes'])/target['real_total_bytes']>.02:
                    specs=[(mode,source,qp,i,True) for i in (0,1,3,4)]
                    candidates.update({s[3]:row for s,row in run_queue(specs)})
                winner=min(candidates,key=lambda i:(abs(candidates[i]['real_total_bytes']-target['real_total_bytes']),i))
                relative=abs(candidates[winner]['real_total_bytes']-target['real_total_bytes'])/target['real_total_bytes']
                for i,row in sorted(candidates.items()):
                    search.append({'sequence':source,'external_qp':qp,'mode':mode,'alpha_index':i,
                        'alpha':(0.8,0.9,1.0,1.1,1.25)[i],'real_total_bytes':row['real_total_bytes'],
                        'actual_bpp':row['actual_bpp'],'target_MAP_total_bytes':target['real_total_bytes'],
                        'relative_rate_error':abs(row['real_total_bytes']-target['real_total_bytes'])/target['real_total_bytes'],
                        'selected':i==winner,'selection_basis':'minimum_absolute_total_byte_distance_only',
                        'rate_only':row['rate_only']})
                selected.append({'sequence':source,'external_qp':qp,'mode':mode,'alpha_index':winner,
                    'relative_rate_error':relative,'status':'CLOSE_RATE_MATCH' if relative<=.02 else 'NO_CLOSE_RATE_MATCH'})
    task.write_csv(P/'rate_match_search.csv',search)
    task.write_csv(P/'rate_match_selection.csv',selected)
    additions=[(r['mode'],r['sequence'],r['external_qp'],r['alpha_index'],False)
               for r in selected if r['alpha_index']!=2]
    assert len(additions)<=8 and len(phase1)+len(phase2)+len(additions)<=44
    run_queue(additions)
    (P/'completion.json').write_text(json.dumps({'status':'EVALUATIONS_COMPLETE',
        'reused_native_anchors':16,'phase1_full':24,'phase2_alpha1_full':12,
        'reused_existing_supplements':5,'preflight_full_metric_runs':3,'preflight_rate_only_encodes':2,
        'additional_selected_full':len(additions),'rate_only_candidates':sum(r['rate_only'] for r in search)},indent=2)+'\n')
    print('ALL_EVALUATIONS_COMPLETE',flush=True)


if __name__=='__main__':main()
