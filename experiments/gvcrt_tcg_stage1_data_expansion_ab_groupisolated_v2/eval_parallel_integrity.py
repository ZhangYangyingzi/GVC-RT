"""Compose unchanged final scientific-protocol assertions with evaluator scheduling checks."""
from eval_parallel_common import *
import traceback

def main():
    verify_parallel_sources();import v3_integrity
    original_dump=v3_integrity.dump
    def preserve_base(path,record):
        if Path(path)==ROOT/'final_integrity_v3.json':path=EP/'base_final_integrity.json'
        original_dump(path,record)
    v3_integrity.dump=preserve_base
    v3_integrity.main()
    base=load(EP/'base_final_integrity.json');assert base['status']=='PASS'
    benchmark=load(EP/'benchmark.json');assert benchmark['correctness_status']=='PASS'
    selected=benchmark['selected_workers_per_gpu'];checked=0
    for manifest in sorted((ROOT/'evaluation_v3/batches').glob('*.json')):
        batch=load(manifest);out=output_root(batch);run=load(out/'run_status.json')
        if 'addendum_sha256' not in run:continue
        assert run['addendum_sha256']==sha(EP/'protocol.json') and run['workers_per_gpu']==selected and run['evaluator_processes']==4*selected
        assert audit_parallel_batch(batch,out)['status']=='PASS'
        for path in (out/'points').glob('*.json'):
            point=load(path);execution=point.get('parallel_execution')
            if execution:
                assert execution['addendum_sha256']==sha(EP/'protocol.json') and execution['workers_per_gpu']==selected
                worker=execution['worker'];assert execution['physical_GPU']==4+worker//selected
                runtime=load(EP/f'worker_audits/{batch["id"]}/worker{worker}/runtime.json')
                assert runtime['status']=='PASS' and runtime['GPU_mapping']['physical_GPU']==execution['physical_GPU'] and runtime['input_decode_concurrency_cap']==4
            checked+=1
    result=dict(status='PASS',addendum=ADDENDUM,source_manifest_sha256=sha(EP/'source_manifest.json'),addendum_sha256=sha(EP/'protocol.json'),benchmark_correctness=True,selected_workers_per_GPU=selected,formal_points_checked=checked,training_protocol_and_source_unchanged=True,legacy_evaluation_audits_preserved=True)
    dump(EP/'final_integrity.json',result);base['evaluation_parallelism']=result;dump(ROOT/'final_integrity_v3.json',base)

if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'final_integrity_v3.json',dict(status='FAILED',stage='evaluation_parallel_integrity',error=traceback.format_exc()));traceback.print_exc();sys.exit(1)
