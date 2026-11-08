"""Real I/P stream identities, complete record validation and canonical MCL registration."""
from common_s2 import *
from evaluate import read_csv,METRICS

def key(v,qp):return f'{v["domain"]}_{v["sequence"]}_qp{qp}'

def mcl_input(v,qp):
    p=load(ROOT/f'canonical_MCL/points/{key(v,qp)}.json')
    assert p['status']=='PASS' and p['real_RANS'] and p['independent_decode'] and not p.get('native_RGB_bypass',False)
    r=p['result'];stream=Path(p['stream_path']);assert sha(stream)==r['bitstream_sha256'] and stream.stat().st_size==r['real_bytes']
    assert p['source_frame_indices']==v['source_frame_indices'] and r['source_RGB_sha256']==v['rgb_sha256']
    assert r['QP']==qp and r['source_sha256']==v['source_sha256'] and p['actual_qps'][0]==qp
    trace=read_csv(p['canonical_codec_trace_path'])
    assert len(trace)==64
    for row in trace:row['reference_feature_sha256']=row['codec_feature_sha256']
    p['audit_source_path']=str(ROOT/f'canonical_MCL/points/{key(v,qp)}.json')
    return p,stream,trace

def register_mcl(v,qp,p,batch):
    assert v['domain']=='mcljcv' and p['real_RANS'] and p['independent_decode'] and p['actual_qps'][0]==qp
    assert p['fresh_encode'] or p['input_stream_reused']
    row=dict(p,canonical_codec_trace_path=str(ROOT/batch/f'codec_trace/{key(v,qp)}.csv'),native_RGB_bypass=False,
        I_frame_bits_included=True,source_native_width=1920,source_native_height=1080,source_fps=v['fps'])
    dest=ROOT/f'canonical_MCL/points/{key(v,qp)}.json'
    if dest.exists():
        old=load(dest)
        assert old['result']['bitstream_sha256']==row['result']['bitstream_sha256'] and old['result']['real_bytes']==row['result']['real_bytes']
        assert read_csv(old['canonical_codec_trace_path'])==read_csv(row['canonical_codec_trace_path'])
    else:dump(dest,row)

def validate_point(root,v,qp,candidate,checkpoint=None,scoring=True):
    path=root/f'points/{key(v,qp)}.json';p=load(path);r=p['result']
    if p.get('measurement_reused'):
        from reuse import validate_reused
        return validate_reused(root,v,qp,candidate,checkpoint)
    assert p['status']=='PASS' and r['sequence']==v['sequence'] and r['domain']==v['domain'] and r['QP']==qp
    assert r['split']==v['split'] and p['candidate']==candidate
    assert p['lora_checkpoint_sha256']==(sha(checkpoint) if checkpoint else None)
    assert p['protocol_sha256']==sha(ROOT/'protocol.json') and p['source_manifest_sha256']==sha(ROOT/'source_manifest.json')
    assert p['point_identity']==dict(model_kind=candidate,checkpoint_SHA256=sha(checkpoint) if checkpoint else None,s_eval=.5 if candidate=='lora' else None,
        domain=v['domain'],sequence=v['sequence'],split=v['split'],external_QP=qp,source_SHA256=v['source_sha256'],RGB_SHA256=v['rgb_sha256'],
        frame_indices=v['source_frame_indices'],protocol_SHA256=sha(ROOT/'protocol.json'),code_SHA256=sha(ROOT/'source_manifest.json'))
    assert p['model_hashes_unchanged'] and p['independent_decode'] and p['real_RANS'] and p['TCG_reset']
    assert p['source_frame_indices']==v['source_frame_indices'] and r['source_sha256']==v['source_sha256'] and r['source_RGB_sha256']==v['rgb_sha256']
    stream=Path(p['stream_path']);assert sha(stream)==r['bitstream_sha256'] and stream.stat().st_size==r['real_bytes']==p['bytes_consumed']
    assert p['record_file_SHA256']=={n:sha(root/n/(key(v,qp)+'.csv')) for n in ['frames','transitions','codec_trace','masks'] if (root/n/(key(v,qp)+'.csv')).exists()}
    fr=read_csv(root/f'frames/{key(v,qp)}.csv');tr=read_csv(root/f'transitions/{key(v,qp)}.csv') if scoring else [];ct=read_csv(root/f'codec_trace/{key(v,qp)}.csv')
    assert len(fr)==len(ct)==64 and [int(x['frame']) for x in fr]==list(range(64))
    assert [int(x['actual_qp']) for x in fr]==p['actual_qps'] and [x['source_frame_RGB_sha256'] for x in fr]==v['frame_rgb_sha256']
    assert sum(int(x['real_bits']) for x in fr)==r['real_bytes']*8 and int(fr[0]['real_bits'])>0
    assert hashlib.sha256(''.join(x['reconstruction_tensor_sha256'] for x in fr).encode()).hexdigest()==p['reconstruction_sha256']
    assert r['bpp']==8*r['real_bytes']/(64*v['height']*v['width']) and abs(r['kbps']-8*r['real_bytes']*v['fps']/64000)<1e-9
    if candidate=='original':assert not p['TCG_modules_present'] and not p['LoRA_modules_present'] and p['baseline_sha256'] is None and p['pilot_sha256'] is None
    elif candidate=='lora':
        assert p['s_eval']==.5 and not p['optimizer_created']
        stats=read_csv(root/f'masks/{key(v,qp)}.csv');gainkeys=[k for k in stats[-1] if k.endswith('_inference_gain')]
        assert len(stats)==63 and len(gainkeys)==5 and all(float(row[k])==.5 for row in stats for k in gainkeys)
    if scoring:
        assert len(tr)==63 and [int(x['to_frame']) for x in tr]==list(range(1,64))
        mse=sum(float(x['pixel_SSE']) for x in fr)/(64*v['height']*v['width']*3)
        assert abs(-10*math.log10(max(mse,1e-15))-r['PSNR'])<1e-9
        for m in ['LPIPS','DISTS','MS_SSIM']:assert abs(sum(float(x[m]) for x in fr)/64-r[m])<1e-10
        for m in ['FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']:assert abs(sum(float(x[m]) for x in tr)/63-r[m])<1e-10
        assert all(math.isfinite(float(r[m])) for m in METRICS)
    return p,fr,tr,ct

def audit_batch(batch,split,candidate,checkpoint=None,scoring=True,limited=None):
    from evaluate import videos
    root=ROOT/batch;vs=videos(split) if limited is None else limited
    expected={(v['sequence'],q) for v in vs for q in [0,5,9]};rows=[]
    for v in vs:
        for q in [0,5,9]:rows.append(validate_point(root,v,q,candidate,checkpoint,scoring)[0]['result'])
    assert len(rows)==len(expected) and {(r['sequence'],r['QP']) for r in rows}==expected
    write_csv(root/'metrics.csv',rows)
    dump(root/'integrity.json',dict(status='PASS',points=len(rows),model_kind=candidate,checkpoint_SHA256=sha(checkpoint) if checkpoint else None,
        all_source_QP_bytes_SHA_codec_features_checked=True,full64frames63transitions=scoring,scoring=scoring,time=time.time()))
    return rows
