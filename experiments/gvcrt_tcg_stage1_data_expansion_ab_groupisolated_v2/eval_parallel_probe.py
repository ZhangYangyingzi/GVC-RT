"""Paired 4/8/4 worker engineering benchmark on already evaluated parent development data."""
from eval_parallel_common import *
import traceback

def compare_points(reference,candidate):
    from gtflow import METRICS
    ref={p.name:load(p) for p in (reference/'points').glob('*.json')}
    new={p.name:load(p) for p in (candidate/'points').glob('*.json')};assert ref.keys()==new.keys() and len(ref)==24
    largest=0.;comparisons=[]
    for key,a in sorted(ref.items()):
        b=new[key]
        for field in ('reconstruction_sha256','source_rgb_sha256','source_sha256','receiver_module_hashes','actual_qps'):
            assert a[field]==b[field],(key,field)
        assert a['result']['bitstream_sha256']==b['result']['bitstream_sha256'] and a['result']['real_bytes']==b['result']['real_bytes']
        for metric in METRICS:
            error=abs(a['result'][metric]-b['result'][metric]);assert error<1e-6,(key,metric,error);largest=max(largest,error)
        stem=Path(a['result']['bitstream_file']).stem
        for folder in ('frames','transitions','encoder_reference'):
            assert sha(reference/folder/(stem+'.csv'))==sha(candidate/folder/(stem+'.csv')),(key,folder)
        comparisons.append(dict(point=key,bitstream_exact=True,reconstruction_exact=True,source_and_receiver_exact=True,frame_transition_and_reference_CSV_exact=True))
    return dict(status='PASS',points=24,max_absolute_metric_error=largest,metric_absolute_tolerance=1e-6,comparisons=comparisons)

def main():
    verify_parallel_sources();protocol=load(EP/'protocol.json');runs=[]
    for label,w in [('reference_before',1),('candidate',2),('reference_after',1)]:
        path=EP/f'probe_manifests/{label}.json';batch=load(path);out=output_root(batch);done=out/'run_status.json'
        if not done.exists() or load(done)['status']!='PASS':
            handle=(ROOT/f'logs/eval_parallel_probe_{label}.log').open('a');cmd=[PYTHON,'-B','-u',str(ROOT/'eval_parallel.py'),'--batch',str(path),'--workers-per-gpu',str(w)]
            process=subprocess.Popen(cmd,cwd=REPO,stdout=handle,stderr=subprocess.STDOUT);dump(EP/'probe_progress.json',dict(status='RUNNING',phase=label,PID=process.pid,workers_per_gpu=w,log=f'logs/eval_parallel_probe_{label}.log',started_unix=time.time()))
            code=process.wait();handle.close();assert code==0,(label,'ENGINEERING_EVALUATION_FAILED')
        r=load(done);assert r['status']=='PASS' and r['completed_points']==24
        runs.append(dict(label=label,workers_per_gpu=w,seconds=r['elapsed_seconds'],out=str(out.relative_to(ROOT)),metrics_sha256=r['metrics_sha256']))
    reference,candidate,again=[ROOT/r['out'] for r in runs]
    checks=[compare_points(reference,candidate),compare_points(again,candidate)]
    # Compare against both references, including the warmed second reference.
    fastest_reference=min(runs[0]['seconds'],runs[2]['seconds']);speedup=fastest_reference/runs[1]['seconds']
    adopt=speedup>=protocol['minimum_speedup_to_adopt']
    dump(EP/'benchmark.json',dict(status='PASS',correctness_status='PASS',checks=checks,runs=runs,speedup_vs_fastest_reference=speedup,minimum_speedup_to_adopt=protocol['minimum_speedup_to_adopt'],selected_workers_per_gpu=2 if adopt else 1,selected_evaluator_processes=8 if adopt else 4,decision='TWO_WORKERS_PER_GPU' if adopt else 'KEEP_ONE_WORKER_PER_GPU_NO_CONFIRMED_SPEEDUP',training_model_protocol_unchanged=True,formal_optimizer_updates_added=0,stopping_or_selection_decisions_added=0,only_already_evaluated_parent_development_data=True,addendum_sha256=sha(EP/'protocol.json')))
    dump(EP/'probe_progress.json',dict(status='COMPLETE',selected_workers_per_gpu=2 if adopt else 1))

if __name__=='__main__':
    try:main()
    except Exception:
        dump(EP/'benchmark.json',dict(status='FAILED',correctness_status='FAILED',error=traceback.format_exc(),formal_optimizer_updates_added=0));traceback.print_exc();sys.exit(1)
