"""CPU delivery audit: saved updates, raw metrics and native decoder receipts."""
import csv
import math
from pathlib import Path

from common_run import HERE, dump, load, sha, state_hash


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def run():
    import torch
    torch.set_num_threads(2)
    assert load(HERE / 'preflight.json')['status'] == 'PASS'
    plan_sha = sha(HERE / 'train_plan.json')
    training = {}
    initial_rng = {}
    paired_streams = None
    inventory = {x['path']: x for x in load(HERE / 'large_file_manifest.json')}
    for b in ('G', 'R', 'RQ'):
        rows = read_csv(HERE / f'training_{b}.csv')
        assert len(rows) == 300 and [int(r['step']) for r in rows] == list(range(1,301))
        assert all(math.isfinite(float(r[k])) for r in rows for k in
                   ('loss','LPIPS','RGB_L1','gradient_norm_preclip','delta_u_RMS','relative_delta_u_norm'))
        streams = [r['stream_sha256'] for r in rows]
        if paired_streams is None: paired_streams = streams
        else: assert streams == paired_streams
        checkpoints = []
        for step in (0,100,300):
            path = HERE / 'local_artifacts/checkpoints' / b / f'step{step}.pt'
            st = torch.load(path, map_location='cpu', weights_only=False)
            assert st['branch'] == b and st['step'] == st['cursor'] == st['effective_updates'] == step
            assert st['plan_sha256'] == plan_sha
            assert len(st['training_rows']) == step
            assert 'optimizer' in st and 'rng' in st
            assert all(int(v['step']) == step for v in st['optimizer']['state'].values())
            if step == 0:
                assert state_hash(st['model']) == st['initial_model_hash']
                initial_rng[b] = st['rng']
            if step == 300:
                assert state_hash(st['model']) != st['initial_model_hash']
                assert st['frozen_components'] == load(HERE / f'training_status_{b}.json')['frozen_components']
            digest = sha(path)
            assert inventory[str(path)]['sha256'] == digest
            checkpoints.append(dict(step=step, path=str(path), sha256=digest, bytes=path.stat().st_size))
        training[b] = dict(updates=300,checkpoint_audit=checkpoints,
            trainable_parameters=load(HERE / f'training_status_{b}.json')['trainable_parameters'],
            parameters_with_nonzero_gradient={r['step']:int(r['parameters_with_nonzero_gradient'])
                for r in rows if r['parameters_with_nonzero_gradient']},
            summed_update_seconds=sum(float(r['seconds']) for r in rows),
            max_reserved_MiB=max(float(r['peak_memory_MiB']) for r in rows))
    assert initial_rng['R']['python'] == initial_rng['RQ']['python']
    for key in ('torch','cuda'):
        assert torch.equal(initial_rng['R'][key], initial_rng['RQ'][key])
    r,n = initial_rng['R']['numpy'],initial_rng['RQ']['numpy']
    assert r[0] == n[0] and (r[1] == n[1]).all() and r[2:] == n[2:]
    assert load(HERE/'training_status_R.json')['initial_model_hash'] == load(HERE/'training_status_RQ.json')['initial_model_hash']
    frame_rows = read_csv(HERE/'frame_metrics.csv')
    transition_rows = read_csv(HERE/'flolpips_transitions.csv')
    assert len(frame_rows) == 2048 and len(transition_rows) == 2016
    assert len(read_csv(HERE/'results_summary.csv')) == 64
    summary_rows=read_csv(HERE/'results_summary.csv')
    differences=read_csv(HERE/'paired_differences.csv')
    assert len(differences) == 96
    summary_index={(r['sequence'],r['external_qp'],r['branch'],r['scope']):r for r in summary_rows}
    for r in differences:
        first,second=r['comparison'].split('-')
        a=summary_index[(r['sequence'],r['external_qp'],first,r['scope'])]
        b=summary_index[(r['sequence'],r['external_qp'],second,r['scope'])]
        for metric in ('LPIPS','DISTS','FloLPIPS','PSNR'):
            assert math.isclose(float(r[metric]),float(a[metric])-float(b[metric]),abs_tol=1e-10)

    export = HERE / 'decode_receipts'; export.mkdir(exist_ok=True)
    checks = []
    for stream in load(HERE/'stream_manifest.json'):
        seq,qp = stream['sequence'],stream['external_qp']
        assert sha(stream['stream_path']) == stream['stream_sha256']
        base = load(HERE/'local_artifacts/decoded'/f'{seq}_q{qp}'/'A/receipt.json')
        for b in ('A','G','R','RQ'):
            receipt = load(HERE/'local_artifacts/decoded'/f'{seq}_q{qp}'/b/'receipt.json')
            dump(export/f'{seq}_q{qp}_{b}.json',receipt)
            assert receipt['stream_sha256'] == base['stream_sha256'] == stream['stream_sha256']
            assert receipt['consumed_bytes'] == stream['total_native_bytes']
            assert receipt['real_total_bytes'] == base['real_total_bytes']
            assert receipt['bpp'] == base['bpp']
            assert len(receipt['states']) == 64
            for a,c in zip(base['states'], receipt['states']):
                for k in ('reference','reference_rgb','reference_poc','y_hat','f','u','q_actual','q_recon','actual_qp','payload_bytes','consumed_bytes'):
                    assert a[k] == c[k],(seq,qp,b,k)
            assert receipt['states'][0]['output_rgb'] == base['states'][0]['output_rgb']
            summary = load(HERE/'local_artifacts/quality'/f'{seq}_q{qp}'/b/'summary.json')
            fr = [r for r in frame_rows if r['sequence']==seq and int(r['external_qp'])==qp and r['branch']==b]
            tr = [r for r in transition_rows if r['sequence']==seq and int(r['external_qp'])==qp and r['branch']==b]
            assert [int(r['frame']) for r in fr] == list(range(64))
            assert [(int(r['from_frame']),int(r['to_frame'])) for r in tr] == [(i,i+1) for i in range(63)]
            for scope, selected in (('all64',fr),('P63',fr[1:])):
                cfg=load(HERE/'public_decode_config.json')[seq]
                psnr=-10*math.log10(sum(float(r['pixel_SSE']) for r in selected)/(len(selected)*cfg['height']*cfg['width']*3))
                assert math.isclose(psnr,summary[scope]['PSNR'],abs_tol=1e-10)
                for metric in ('LPIPS','DISTS'):
                    assert math.isclose(sum(float(r[metric]) for r in selected)/len(selected),summary[scope][metric],abs_tol=1e-10)
                assert math.isclose(sum(float(r['FloLPIPS']) for r in tr)/63, summary[scope]['FloLPIPS'],abs_tol=1e-10)
            checks.append(dict(sequence=seq,external_qp=qp,branch=b,frames=64,transitions=63,status='PASS'))
    for item in inventory.values():
        path=Path(item['path']); assert path.exists() and path.stat().st_size == item['bytes'],str(path)
    assert len([p for p in (HERE/'figures').glob('*_frame[0-9][0-9].png') if '_overview_' not in p.name]) == 48
    assert len(list((HERE/'figures').glob('*_overview_frame[0-9][0-9].png'))) == 24
    dump(HERE/'delivery_audit.json',dict(status='PASS_LOCAL',training=training,
         R_RQ_initial_weights_and_rng_identical=True,shared_training_streams=True,
         quality_points=checks,raw_frame_metric_rows=2048,raw_transition_rows=2016,
         summary_rows=64,paired_difference_rows=96,
         inventory_file_count=len(inventory),inventory_sizes_verified=True,
         inventory_hashes='All artifacts hashed by finalize.py; step0/100/300 checkpoints and evaluation streams rehashed here.',
         publication='Not established by this CPU audit'))
    print('DELIVERY_AUDIT_PASS',flush=True)


if __name__ == '__main__':
    run()
