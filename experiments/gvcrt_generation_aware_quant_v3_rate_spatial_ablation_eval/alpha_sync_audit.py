"""Check every supported nonunit alpha using persisted real streams."""
import argparse
import json
from pathlib import Path

import task
import driver

P=Path(__file__).resolve().parent


def main(ensure=False):
    driver.validate_gate()
    gate=json.loads((P/'preflight.json').read_text())
    specs=[(mode,'UVal1',2,alpha,True) for mode in ('MEAN','ROLL') for alpha in (0,1,3,4)]
    if ensure:
        missing=[]
        for spec in specs:
            mode,source,qp,alpha,_=spec
            path=P/'server_artifacts/rate_only'/f'{task.key(mode,source,qp,alpha)}.json'
            row=json.loads(path.read_text()) if path.exists() else {}
            if not (row.get('status')=='PASS' and Path(row['stream_path']).exists()
                    and task.sha(row['stream_path'])==row['stream_sha256']):missing.append(spec)
        if missing:driver.run_queue(missing)
    results=[]
    for mode,source,qp,alpha,_ in specs:
        path=P/'server_artifacts/rate_only'/f'{task.key(mode,source,qp,alpha)}.json'
        row=json.loads(path.read_text())
        assert row['status']=='PASS' and row['rate_only'] is True
        assert row['all_64_reference_states_synchronized'] is True
        assert row['all_P_latents_qmaps_exact'] is True
        assert row['alpha_index']==alpha and row['mode']==mode and row['frames']==64
        assert row['checkpoint_sha256']==gate['checks']['weights_sha256']['H11']
        # Initial alpha=.8 checks predate the final runner version.
        # Keep their recorded runner SHA; require the identical codec/ablation
        # implementations, weight and imported runtime rather than relabelling it.
        initial_runner='8292f0de9150a3a625f54f268dd1766b184bd1d2d8cb38014c7cea59cc413ad3'
        allowed_runner={gate['checks']['task.py_sha256']}|({initial_runner} if alpha==0 else set())
        assert row['task_code_sha256'] in allowed_runner
        assert row['ablation_code_sha256']==gate['checks']['ablation.py_sha256']
        assert row['v2_runtime_sha256']==gate['checks']['v2_runtime_sha256']
        assert row['source_sha256']==task.source_record(source)['source_sha256']
        assert row['imported_runtime_path']==str(task.V2/'runtime.py')
        assert task.sha(row['stream_path'])==row['stream_sha256']
        assert task.read_header(Path(row['stream_path']).read_bytes())[:2]==(mode,alpha)
        results.append({'mode':mode,'source':source,'QP':qp,'alpha_index':alpha,'alpha':row['alpha'],
            'status':'PASS','all_64_references_exact':True,'all_P_latents_qmaps_exact':True,
            'stream_sha256':row['stream_sha256'],'result_sha256':task.sha(path),
            'runner_code_sha256':row['task_code_sha256'],
            'ablation_code_sha256':row['ablation_code_sha256'],'runtime_code_sha256':row['v2_runtime_sha256'],
            'real_total_bytes':row['real_total_bytes'],'used_in_rate_selection':False})
    rejected=0
    for mode in ('A','MAP','IDENTITY'):
        for alpha in (0,1,3,4):
            try:task.read_header(task.encode_header(mode,alpha))
            except AssertionError:rejected+=1
            else:raise AssertionError((mode,alpha,'nonunit alpha must be rejected'))
    assert rejected==12
    result={'status':'PASS','nonunit_alpha_stream_checks':results,'unsupported_headers_rejected':rejected,
        'tensor_tolerance':0,'quality_metrics_computed':False,'used_for_rate_selection':False,
        'initial_gate_coverage':'MEAN/ROLL alpha=0.8 before protocol evaluation',
        'supplementary_coverage':'MEAN/ROLL alpha=0.9,1.1,1.25 after protocol evaluation; inference code unchanged',
        'initial_runner_provenance':'Original alpha=.8 runner SHA retained from initial configs; ablation and v2 runtime SHAs are identical to the final gate',
        'code_sha256':task.sha(P/'alpha_sync_audit.py')}
    (P/'alpha_synchronization_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ALL_FIXED_NONUNIT_ALPHAS_SYNC_PASS',len(results),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ensure',action='store_true')
    main(parser.parse_args().ensure)
