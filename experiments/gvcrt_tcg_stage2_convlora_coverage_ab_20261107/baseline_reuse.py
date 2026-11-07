"""Audited reuse of the identical frozen baseline only; never reuse A/B quality."""
from common_s2 import *
from evaluate import videos,historical_input,read_csv,METRICS
import ast

def source_segment(path,name):
    text=path.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    return ast.get_source_segment(text,node)

def compatibility():
    old=load(REFERENCE/'source_manifest.json');assert old['status']=='FROZEN'
    for n,h in old['code'].items():assert sha(REFERENCE/n)==h,('REFERENCE_CODE_CHANGED',n)
    for n,h in old['immutable_manifests'].items():assert sha(REFERENCE/n)==h,('REFERENCE_MANIFEST_CHANGED',n)
    assert old['pilot_sha256']==PILOT_SHA and old['baseline_sha256']==BASELINE_SHA
    assert sha(PILOT)==PILOT_SHA and sha(BASELINE)==BASELINE_SHA
    for p in (REFERENCE/'runtime_tree').rglob('*.py'):
        assert sha(p)==sha(ROOT/p.relative_to(REFERENCE))
    for name in ['configure_safe_dispatch','_safe_block_forward','_safe_generator_cuda']:
        assert source_segment(ROOT/'convlora.py',name)==source_segment(REFERENCE/'convlora.py',name)
    for name in ['videos','arrays_for','metrics_setup']:
        assert source_segment(ROOT/'evaluate.py',name)==source_segment(REFERENCE/'evaluate.py',name)
    new=(ROOT/'model_context.py').read_text();ref=(REFERENCE/'model_context.py').read_text()
    begin="    torch,core,eco,tr,qp,tree=backend(reference,rank)";end='    from convlora import'
    assert new[new.index(begin):new.index(end)]==ref[ref.index(begin):ref.index(end)],'BASELINE_LOAD_PREFIX_CHANGED'
    protocol=load(ROOT/'protocol.json');rp=load(REFERENCE/'protocol.json')
    assert protocol['training']==rp['training'] and protocol['data']==rp['data']
    for key in ['RGB','metrics','inherited_PWC_sampler']:
        assert protocol['evaluation'][key]==rp['evaluation'][key]
    proof=load(ROOT/'audits/RANS_coverage_identity.json');assert proof['status']=='PASS' and proof['reference_baseline_reproduced']
    return dict(reference_protocol_sha256=sha(REFERENCE/'protocol.json'),reference_source_manifest_sha256=sha(REFERENCE/'source_manifest.json'),
        strict_loader_prefix_equal=True,runtime_tree_equal=True,safe_dispatch_equal=True,metric_setup_equal=True,
        reference_baseline_native_RANS_reconstruction_reproduced=True,quality_candidate_reuse=False)

def reuse(split):
    env();proof=compatibility();dest=eval_dir(split,group='baseline')
    if (dest/'integrity.json').exists():
        a=load(dest/'integrity.json');assert a['status']=='PASS' and a['reference_source_manifest_sha256']==proof['reference_source_manifest_sha256'];return dest
    if split=='final':
        assert all(load(group_dir(g)/'selection.json')['status']=='FROZEN_BEFORE_FINAL_TEST' for g in GROUPS)
    source=REFERENCE/'evaluation'/evaluation_id(split,candidate='baseline');expected=len(videos(split))*3
    assert load(source/'integrity.json')['status']=='PASS' and load(source/'integrity.json')['points']==expected
    weight_hashes=load(ORIGINAL_EVAL/'metric_runtime_rank0.json')['hashes'];metrics=[];records=[];flows={}
    for v in videos(split):
        for qp in [0,5,9]:
            key=f'{v["domain"]}_{v["sequence"]}_qp{qp}';path=source/f'points/{key}.json';p=load(path);r=p['result']
            assert p['status']=='PASS' and p['candidate']=='baseline' and p['lora_checkpoint_sha256'] is None
            assert p['baseline_sha256']==BASELINE_SHA and p['pilot_sha256']==PILOT_SHA
            assert p['source_manifest_sha256']==proof['reference_source_manifest_sha256'] and p['protocol_sha256']==proof['reference_protocol_sha256']
            assert p['metric_hashes']==weight_hashes and p['model_hashes_unchanged'] and p['independent_decode'] and p['real_RANS']
            assert p['source_frame_indices']==v['source_frame_indices'] and r['source_RGB_sha256']==v['rgb_sha256'] and r['source_sha256']==v['source_sha256']
            canonical,stream,trace=historical_input(v,qp)
            assert trace is not None and sha(stream)==r['bitstream_sha256'] and stream.stat().st_size==r['real_bytes']==p['bytes_consumed']
            assert r['bpp']==8*r['real_bytes']/(64*v['height']*v['width'])
            frames=read_csv(source/f'frames/{key}.csv');trans=read_csv(source/f'transitions/{key}.csv')
            assert len(frames)==64 and len(trans)==63
            assert sum(int(x['real_bits']) for x in frames)==8*r['real_bytes']
            assert [int(x['actual_qp']) for x in frames]==p['actual_qps']
            assert [x['source_frame_RGB_sha256'] for x in frames]==v['frame_rgb_sha256']
            assert hashlib.sha256(''.join(x['reconstruction_tensor_sha256'] for x in frames).encode()).hexdigest()==p['reconstruction_sha256']
            mse=sum(float(x['pixel_SSE']) for x in frames)/(64*v['height']*v['width']*3)
            assert abs(-10*math.log10(max(mse,1e-15))-r['PSNR'])<1e-9
            for name in ['LPIPS','DISTS','MS_SSIM']:assert abs(sum(float(x[name]) for x in frames)/64-r[name])<1e-10
            for name in ['FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']:assert abs(sum(float(x[name]) for x in trans)/63-r[name])<1e-10
            assert all(math.isfinite(float(r[m])) for m in METRICS)
            for row in trans:flows.setdefault((v['sequence'],int(row['to_frame'])),set()).add(row['GT_flow_sha256'])
            copied=[]
            for folder,filename in [('points',key+'.json')]+[(n,key+'.csv') for n in ['frames','transitions','codec_trace','masks']]:
                s=source/folder/filename;d=dest/folder/filename;d.parent.mkdir(parents=True,exist_ok=True)
                if d.exists():assert sha(d)==sha(s),'CONFLICTING_REUSED_BASELINE_ARTIFACT'
                else:shutil.copyfile(s,d)
                copied.append(dict(path=str(d.relative_to(ROOT)),source_path=str(s),sha256=sha(s)))
            metrics.append(r);records.append(dict(sequence=v['sequence'],QP=qp,files=copied,stream_sha256=sha(stream),
                old_measurement_reused=True,new_decode_performed=False))
    assert all(len(x)==1 for x in flows.values()) and len(metrics)==expected
    write_csv(dest/'metrics.csv',metrics)
    dump(dest/'reuse_manifest.json',dict(status='PASS',proof=proof,records=records,time=time.time()))
    dump(dest/'integrity.json',dict(status='PASS',points=expected,candidate='baseline',checkpoint_sha256=None,
        historical_measurement_reused=True,new_decode_performed=False,independent_decode_evidence='historical audited measurement',
        reference_source_manifest_sha256=proof['reference_source_manifest_sha256'],same_windows_weights_precision_metrics_and_streams=True))
    return dest
