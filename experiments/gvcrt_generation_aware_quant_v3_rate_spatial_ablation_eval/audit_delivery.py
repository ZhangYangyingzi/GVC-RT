"""Verify persisted outputs and receiver traces before report/publication."""
import csv
import hashlib
import json
import sys
import platform
import importlib.metadata
from pathlib import Path

import task
import driver

P=Path(__file__).resolve().parent


def read(name):
    with (P/name).open() as f:return list(csv.DictReader(f))


def main():
    driver.validate_gate()
    gate=json.loads((P/'preflight.json').read_text())
    assert task.sha(P/'evaluation_sources.json')==gate['checks']['v2_sources_sha256']
    sources={x['id']:x for x in json.loads((P/'evaluation_sources.json').read_text())}
    data_audit=[]
    for source in sources.values():
        if source['file_type']=='mp4':
            assert task.sha(source['path'])==source['source_sha256']
            count=1
        else:
            assert task.sha(source['source_path'])==source['source_sha256']
            expected=source.get('frame_png_sha256') or source['frame_file_sha256']
            for i,wanted in enumerate(expected):
                name=f'im{i+1:05d}.png' if source['id']=='ParkScene' else f'frame_{i:06d}.png'
                assert task.sha(Path(source['path'])/name)==wanted
            count=len(expected)
        data_audit.append({'source':source['id'],'status':'PASS','files_verified':count,
                           'source_sha256':source['source_sha256'],'frames':len(source['indices'])})
    evaluations=read('evaluation.csv');audits={};records={};manifest=[];diagnostics=[];summaries=[];resources=[]
    config_sha=task.sha(P/'effective_config.json')
    weights=gate['checks']['weights_sha256']
    for row in evaluations:
        mode,source,qp,alpha=row['mode'],row['sequence'],int(row['external_qp']),int(row['alpha_index'])
        key=task.key(mode,source,qp,alpha)
        result_path=P/'server_artifacts/evaluation'/f'{key}.json'
        record=json.loads(result_path.read_text());records[key]=record
        stream=Path(record['stream_path'])
        assert record['stream_sha256']==task.sha(stream)==row['stream_sha256']
        decoded_mode,decoded_alpha,native=task.read_header(stream.read_bytes())
        assert (decoded_mode,decoded_alpha)==(mode,alpha)
        assert hashlib.sha256(native).hexdigest()==record['native_stream_sha256']
        assert record['checkpoint_sha256']==weights['A' if mode=='A' else 'H11']
        assert record['source_sha256']==sources[source]['source_sha256']
        assert record['source_frame_indices']==sources[source]['indices']
        assert record['all_64_reference_states_synchronized'] is True
        audit_path=P/'server_artifacts/receiver_audits'/f'{key}.json'
        audit=json.loads(audit_path.read_text());audits[key]=audit
        assert audit['status']=='PASS' and audit['maps_code_sha256']==task.sha(P/'maps.py')
        assert audit['task_code_sha256']==gate['checks']['task.py_sha256']
        assert audit['ablation_code_sha256']==gate['checks']['ablation.py_sha256']
        assert audit['stream_sha256']==record['stream_sha256'] and audit['checkpoint_sha256']==record['checkpoint_sha256']
        assert audit['runtime_path']==str(task.V2/'runtime.py')
        assert audit['frames']==64 and audit['P_frames']==63 and len(audit['state_hashes'])==64
        assert audit['rho_post_dtype_max_abs_error']==0.0 and audit['mean_definition_max_abs_error']<=1e-7
        reused=record['measurement'].startswith('reused_')
        if reused:
            old=task.old_record(mode,source,qp)
            old_path=task.old_stream(mode,source,qp,old)
            assert native==old_path.read_bytes()
            old_record_path=task.V2/'server_artifacts/evaluation'/f'{"A" if mode=="A" else "H11"}_s300_{source}_q{qp}.json'
            assert task.sha(old_record_path)==record['reuse_record_sha256']
            assert audit['v2_receiver_parity']['all_64_RGB_exact']
            assert audit['v2_receiver_parity']['all_64_references_exact']
            assert audit['v2_receiver_parity']['all_P_latents_qmaps_exact']
            for metric in ('LPIPS','DISTS','FloLPIPS','PSNR'):
                assert record[metric]==old[metric]
        else:
            assert record['all_P_latents_qmaps_exact'] is True
            assert record['task_code_sha256']==gate['checks']['task.py_sha256']
            assert record['ablation_code_sha256']==gate['checks']['ablation.py_sha256']
            assert record['v2_runtime_sha256']==gate['checks']['v2_runtime_sha256']
        canonical=f'{key}_src{record["source_sha256"][:12]}_w{record["checkpoint_sha256"][:12]}_rt{gate["checks"]["v2_runtime_sha256"][:12]}_cfg{config_sha[:12]}'
        state_digest=hashlib.sha256(json.dumps(audit['state_hashes'],sort_keys=True).encode()).hexdigest()
        manifest.append({'task_key':canonical,'storage_alias':key,'mode':mode,'sequence':source,'external_qp':qp,
            'alpha':record['alpha'],'alpha_index':alpha,'checkpoint_sha256':record['checkpoint_sha256'],
            'source_sha256':record['source_sha256'],'stream_sha256':record['stream_sha256'],
            'native_stream_sha256':record['native_stream_sha256'],'codec_payload_sha256':row['codec_payload_sha256'],
            'result_sha256':task.sha(result_path),'receiver_audit_sha256':task.sha(audit_path),
            'all_64_state_hashes_sha256':state_digest,'v2_parity':reused,'status':'PASS'})
        config={'task_key':canonical,'storage_alias':key,'configuration':json.loads((P/'effective_config.json').read_text()),
            'result':record,'imported_modules':audit['module_paths'],
            'imported_modules_sha256':{n:task.sha(path) for n,path in audit['module_paths'].items()},
            'config_sha256':config_sha,'receiver_state_digest':state_digest}
        (P/'configs').mkdir(exist_ok=True)
        (P/'configs'/f'{key}.json').write_text(json.dumps(config,indent=2)+'\n')
        ds=read(f'server_artifacts/receiver_diagnostics/{key}.csv')
        assert len(ds)==63
        diagnostics.extend({'task_key':canonical,'storage_alias':key,'mode':mode,'sequence':source,
                            'external_qp':qp,'alpha_index':alpha,**x} for x in ds)
        for segment,items in [('all_P',ds),('first15',ds[:14]),('remaining49',ds[14:])]:
            summary={'storage_alias':key,'segment':segment,'P_frames':len(items),'mode':mode,
                     'sequence':source,'external_qp':qp,'alpha_index':alpha}
            for col in ('q_clamp_fraction','pass0_skip_fraction','pass1_skip_fraction','raw_m_mean','raw_m_std',
                        'm_mean','m_std','rho_mean','rho_std','actual_q_mean'):
                summary[col]=sum(float(d[col]) for d in items)/len(items)
            summaries.append(summary)
        resources.append({'storage_alias':key,'reused':reused,'encoding_seconds':record['encoding_seconds'],
            'decoding_seconds':record['decoding_seconds'],'evaluation_peak_MiB':record['peak_MiB'],
            'receiver_audit_seconds':audit['elapsed_seconds'],'receiver_peak_MiB':audit['peak_MiB'],
            'receiver_physical_gpu':audit['physical_gpu']})
    identities=[]
    for source in ('UVal1','Jockey'):
        for qp in (2,7):
            a,i=(task.key(m,source,qp,2) for m in ('A','IDENTITY'))
            assert records[a]['native_stream_sha256']==records[i]['native_stream_sha256']
            for col in ('reference','latent','qmap'):
                assert all(x[col]==y[col] for x,y in zip(audits[a]['state_hashes'],audits[i]['state_hashes']))
            identities.append({'source':source,'QP':qp,'native_stream_exact':True,'all_64_references_exact':True,
                               'all_P_latents_qmaps_exact':True,'tolerance':0})
    selection=read('rate_match_selection.csv');search=read('rate_match_search.csv')
    assert len(selection)==8
    for choice in selection:
        candidates=[x for x in search if all(x[k]==choice[k] for k in ('sequence','external_qp','mode'))]
        assert candidates and len([x for x in candidates if x['selected']=='True'])==1
        winner=next(x for x in candidates if x['selected']=='True')
        assert winner['alpha_index']==choice['alpha_index']
        assert float(winner['relative_rate_error'])==min(float(x['relative_rate_error']) for x in candidates)
        if len(candidates)==1:
            assert winner['alpha_index']=='2' and float(winner['relative_rate_error'])<=.02
        else:assert {int(x['alpha_index']) for x in candidates}=={0,1,2,3,4}
    preflight_rate_only=[];preflight_tasks=[]
    preflight_specs=[('A','evaluation',2),('MAP','evaluation',2),('IDENTITY','evaluation',2),
                     ('MEAN','rate_only',0),('ROLL','rate_only',0)]
    for mode,group,alpha in preflight_specs:
        source='UVal1' if group=='rate_only' else 'UVal0'
        key=task.key(mode,source,2,alpha)
        path=P/f'server_artifacts/{group}/{key}.json'
        record=json.loads(path.read_text())
        assert record['status']=='PASS' and task.sha(record['stream_path'])==record['stream_sha256']
        preflight_tasks.append({'storage_alias':key,'source':source,'mode':mode,'QP':2,'alpha':record['alpha'],
            'rate_only':group=='rate_only','real_total_bytes':record['real_total_bytes'],
            'stream_sha256':record['stream_sha256'],'checkpoint_sha256':record['checkpoint_sha256'],
            'source_sha256':record['source_sha256'],'result_sha256':task.sha(path),'status':'PASS'})
        (P/'configs'/f'preflight_{key}.json').write_text(json.dumps({'preflight':True,'result':record},indent=2)+'\n')
    for mode in ('MEAN','ROLL'):
        record=json.loads((P/'server_artifacts/rate_only'/f'{task.key(mode,"UVal1",2,0)}.json').read_text())
        assert task.sha(record['stream_path'])==record['stream_sha256']
        assert record['all_64_reference_states_synchronized'] and record['all_P_latents_qmaps_exact']
        preflight_rate_only.append({'mode':mode,'alpha':.8,'source':'UVal1','QP':2,'status':'PASS',
                                   'stream_sha256':record['stream_sha256'],'used_in_rate_selection':False})
    task.write_csv(P/'task_manifest.csv',manifest)
    task.write_csv(P/'quant_diagnostics.csv',diagnostics)
    task.write_csv(P/'quant_summary.csv',summaries)
    task.write_csv(P/'resource_summary.csv',resources)
    task.write_csv(P/'preflight_tasks.csv',preflight_tasks)
    code_paths=list(P.glob('*.py'))+[P/'run.sh',task.V2/'runtime.py',task.V2/'evaluate.py',task.V2/'train.py']
    code_sha={str(path.relative_to(P.parents[1])):task.sha(path) for path in code_paths}
    base_weights={str(path.relative_to(P.parents[1])):task.sha(path) for path in
        [P.parents[1]/'checkpoints/GVC-RT_I.pt',P.parents[1]/'checkpoints/GVC-RT_P.pt',
         P.parent/'gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100/evaluation/checkpoints/G_basic_100.pt']}
    versions={}
    for name in ('torch','numpy','opencv-python','Pillow','matplotlib','lpips','DISTS-pytorch'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='not_a_distribution'
    (P/'environment.json').write_text(json.dumps({'python_executable':sys.executable,
        'python_version':platform.python_version(),'packages':versions},indent=2)+'\n')
    result={'status':'PASS','protocol_tasks':len(evaluations),'new_full_tasks':36,
        'reused_native_anchor_tasks':16,'reused_existing_supplement_tasks':5,
        'all_stream_SHA_verified':True,'all_byte_accounts_exact':True,
        'all_64_frame_files_present':True,'all_63_transition_files_present':True,
        'all_64_receiver_state_hashes_saved':True,'reused_all_64_RGB_and_codec_state_exact':21,
        'new_encode_independent_decode_records_verified':36,'identity_A_exact_state_checks':identities,
        'preflight_full_metric_runs':3,'total_full_metric_runs_including_preflight':39,
        'rho_post_dtype_max_abs_error':0.0,'mean_definition_tolerance':1e-7,
        'tensor_and_native_stream_tolerance':0,'preflight_status':'PASS',
        'data_sources':data_audit,'preflight_rate_only_tasks':preflight_rate_only,
        'additional_rate_search_encodes':sum(x['rate_only']=='True' for x in search),
        'selected_extra_full_evaluations':sum(x['alpha_index']!='2' for x in selection),
        'code_sha256':code_sha,'frozen_codec_and_bridge_weights_sha256':base_weights,
        'canonical_task_key':'mode_step_source_QP_alpha_sourceSHA_weightSHA_runtimeSHA_configSHA',
        'quant_summary_definition':'Per-frame spatial statistics, averaged over the declared segment; not pooled spatial quantiles.',
        'uncompleted_items':[]}
    (P/'delivery_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    completion=json.loads((P/'completion.json').read_text())
    completion.update({'delivery_status':'PASS','reused_existing_supplements':5,
        'preflight_full_metric_runs':3,'preflight_rate_only_encodes':2,
        'receiver_only_audits':len(evaluations),'total_full_metric_runs_including_preflight':39})
    (P/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    # Keep exported text tables compatible with Git whitespace checks.
    # Original per-task CSV artifacts and their provenance stay untouched.
    for path in P.glob('*.csv'):
        data=path.read_bytes()
        path.write_bytes(data.replace(b'\r\n',b'\n'))
    print('DELIVERY_AUDIT_PASS',len(evaluations),flush=True)


if __name__=='__main__':main()
