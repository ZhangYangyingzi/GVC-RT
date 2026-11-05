"""Strict point/stream/frame identity audit before reusing independently measured scores."""
from common_s2 import *
from evaluate import videos,METRICS
import collections

def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def audit_point(point_path,v,large_root,checkpoint_sha,frames_root):
    p=load(point_path);r=p['result'];q=int(r['QP']);key=Path(r['bitstream_file']).stem
    assert p['status']=='PASS' and p['real_RANS'] and p['independent_decode'] and p['model_hashes_unchanged']
    assert r['source_RGB_sha256']==v['rgb_sha256'] and r['source_sha256']==v['source_sha256']
    assert p['source_frame_indices']==v['source_frame_indices'] and p['actual_qps'][0]==q
    if checkpoint_sha is not None:
        assert r['checkpoint_sha256']==checkpoint_sha and p['state_sync_pass'] and all(x['cleared'] for x in p['TCG_reset_audits'])
        expected=load(ROOT/'audits/load_probe.json')['receiver_hashes']
        assert p['receiver_module_hashes']==expected
    stream=large_root/r['bitstream_file'];assert sha(stream)==r['bitstream_sha256'] and stream.stat().st_size==r['real_bytes']==p['bytes_consumed']
    fr=read_csv(frames_root/f'frames/{key}.csv');tr=read_csv(frames_root/f'transitions/{key}.csv')
    assert len(fr)==64 and len(tr)==63 and [int(x['frame']) for x in fr]==list(range(64)) and [int(x['to_frame']) for x in tr]==list(range(1,64))
    assert all(f['source_frame_RGB_sha256']==v['frame_rgb_sha256'][i] for i,f in enumerate(fr))
    assert sum(int(f['real_bits']) for f in fr)==8*r['real_bytes']
    assert abs(r['bpp']-8*r['real_bytes']/(64*v['height']*v['width']))<1e-12
    assert abs(r['kbps']-8*r['real_bytes']*v['fps']/64/1000)<1e-9
    assert hashlib.sha256(''.join(f['reconstruction_tensor_sha256'] for f in fr).encode()).hexdigest()==p['reconstruction_sha256']
    for m in ['MS_SSIM','LPIPS','DISTS']:assert abs(sum(float(f[m]) for f in fr)/64-r[m])<1e-10
    for m in ['FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']:assert abs(sum(float(t[m]) for t in tr)/63-r[m])<1e-10
    mse=sum(float(f['pixel_SSE']) for f in fr)/(64*v['height']*v['width']*3);assert abs(-10*math.log10(max(mse,1e-15))-r['PSNR'])<1e-9
    assert all(math.isfinite(float(r[m])) for m in METRICS)
    record=dict(point_file=str(point_path),point_sha256=sha(point_path),stream_sha256=r['bitstream_sha256'],stream_bytes=r['real_bytes'],frames_sha256=sha(frames_root/f'frames/{key}.csv'),transitions_sha256=sha(frames_root/f'transitions/{key}.csv'),
        checkpoint_sha256=checkpoint_sha,sequence=v['sequence'],QP=q,source_sha256=v['source_sha256'],RGB_sha256=v['rgb_sha256'],reconstruction_sha256=p['reconstruction_sha256'],
        original_fresh_encode=p['fresh_encode'],metrics_reused=True,fresh_encode_in_Stage2=False,input_stream_reused_for_rescoring=False)
    return dict(r),record

def audit_reuse():
    from evaluate import videos
    original_protocol=load(ORIGINAL_EVAL/'protocol.json');assert original_protocol['protocol_revision']=='matched_rgb_fp32_v2'
    # Preserve exact scoring source and independently evaluated original provenance.
    assert load(ORIGINAL_EVAL/'final_integrity.json')['status']=='PASS'
    actual=ROOT/'runtime_tree/experiments/gvcrt_neural_wrapper_v2_joint/core.py'
    previous=HIST.parents[1]/'experiments/gvcrt_neural_wrapper_v2_joint/core.py'
    assert sha(actual)==sha(previous),'SCORING_CORE_CHANGED'
    sets=[('B_development','development',HIST/'evaluation_v3/development_c640d16978902780663b','B_expanded_ulong_add1500',HIST/'large_artifacts/evaluation_v3/development_c640d16978902780663b/bitstreams',BASELINE_SHA),
          ('B_final','final',HIST/'evaluation_v3/final_d2b5efc4d5756f3604d7','B_common',HIST/'large_artifacts/evaluation_v3/final_d2b5efc4d5756f3604d7/bitstreams',BASELINE_SHA),
          ('original_final','final',ORIGINAL_EVAL,'GVC_RT_original',ORIGINAL_EVAL/'bitstreams',None)]
    allrecords=[]
    for label,split,old,model,large,cp in sets:
        rows=[];records=[]
        for v in videos(split):
            for q in [0,5,9]:
                key=f'{model}_{v["domain"]}_{v["sequence"]}_qp{q}'
                r,record=audit_point(old/f'points/{key}.json',v,large,cp,old);r['model_state']='TCG_B1500' if cp else 'GVC_RT_original';r['metrics_reused']=True;r['reuse_source']=str(old)
                rows.append(r);records.append(record)
        assert len(rows)==(96 if split=='development' else 135)
        write_csv(ROOT/f'evaluation/reused/{label}.csv',rows);dump(ROOT/f'audits/reuse_{label}.json',dict(status='PASS',points=len(rows),records=records,RGB_protocol='matched_rgb_fp32_v2'))
        allrecords.extend(records)
    dump(ROOT/'audits/reuse_summary.json',dict(status='PASS',points=len(allrecords),B_development=96,B_final=135,original_final=135,records_sha256=digest(allrecords)))

if __name__=='__main__':env();audit_reuse()
