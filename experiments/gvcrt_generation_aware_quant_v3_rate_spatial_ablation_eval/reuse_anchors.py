"""Wrap v2 native A/H11 streams without claiming a new measurement."""
import csv
import json
import shutil
import sys
from pathlib import Path

P=Path(__file__).resolve().parent
V2=P.parent/'gvcrt_generation_aware_quant_v2_skip_domain_trend_300'
V1=P.parent/'gvcrt_generation_aware_quant_visual_condition_v1_trend_300'
sys.path.insert(0,str(V2))
import task


def wrap(mode, source, qp):
    assert mode in ('A','MAP') and qp in (2,7,9)
    branch='A' if mode=='A' else 'H11'
    old=task.old_record(mode,source,qp)
    assert old and old['frames']==64 and old['independent_decode'] and old['all_64_reference_states_synchronized']
    native=task.old_stream(mode,source,qp,old)
    checkpoint=(V1 if mode=='A' else V2)/'server_artifacts'/branch/'step_300.pt'
    assert old['checkpoint_sha256']==task.sha(checkpoint)
    seq=task.source_record(source)
    assert seq['indices']==old['source_frame_indices']
    assert task.sha(seq['path'])==seq['source_sha256'] if seq['file_type']=='mp4' else True
    key=task.key(mode,source,qp,2)
    output=P/'server_artifacts/streams'/f'{key}.bin'
    output.parent.mkdir(parents=True,exist_ok=True)
    raw=task.encode_header(mode,2)+native.read_bytes()
    if output.exists():
        assert output.read_bytes()==raw
    else:
        output.write_bytes(raw)
    assert task.read_header(output.read_bytes())==(mode,2,native.read_bytes())
    result={k:v for k,v in old.items() if k not in ('stream_path','stream_sha256','actual_bpp','real_total_bytes','header_bytes')}
    result.update({'mode':mode,'alpha_index':2,'alpha':1.0,'source_sha256':seq['source_sha256'],
        'real_total_bytes':len(raw),'actual_bpp':len(raw)*8/(64*seq['height']*seq['width']),
        'codec_payload_bytes':old['payload_bytes'],'native_header_bytes':old['header_bytes'],
        'experiment_header_bytes':task.HEADER.size,'native_stream_bytes':len(raw)-task.HEADER.size,
        'native_stream_sha256':task.sha(native),'stream_path':str(output),'stream_sha256':task.sha(output),
        'payload_origin':str(native),'checkpoint_sha256':task.sha(checkpoint),
        'measurement':'reused_v2_native_stream_and_quality_rewrapped_v3',
        'reuse_record_sha256':task.sha(V2/'server_artifacts/evaluation'/f'{branch}_s300_{source}_q{qp}.json'),
        'all_P_latents_qmaps_exact':old.get('all_P_latents_qmaps_exact','v2_verified'),
        'v3_runtime_parity_basis':'preflight A/MAP UVal0 QP2 full independent decode and exact per-frame metric comparison'})
    path=P/'server_artifacts/evaluation'/f'{key}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        existing=json.loads(path.read_text())
        assert existing['stream_sha256']==task.sha(output)
        return existing
    path.write_text(json.dumps(result,indent=2)+'\n')
    for group,old_group in [('frames','evaluation_frames'),('transitions','evaluation_transitions')]:
        old_csv=V2/'server_artifacts'/old_group/f'{branch}_s300_{source}_q{qp}.csv'
        if not old_csv.exists() and mode=='A':
            old_csv=V1/'server_artifacts'/old_group/f'{branch}_s300_{source}_q{qp}.csv'
        assert old_csv.exists(),old_csv
        dest=P/'server_artifacts'/group/f'{key}.csv'
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(old_csv,dest)
    return result


if __name__=='__main__':
    for mode in ('A','MAP'):
        for source in ('UVal0','UVal1','ParkScene','Jockey'):
            for qp in (2,7):
                wrap(mode,source,qp)
    for source in ('UVal0','UVal1','ParkScene','Jockey'):
        wrap('A',source,9)
    wrap('MAP','UVal0',9)
    print('ANCHORS_AND_SUPPLEMENTS_REWRAPPED',21)
