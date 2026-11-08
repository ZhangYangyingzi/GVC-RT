"""Read-only, individually hashed historical measurements; no mislabeled gain reuse."""
from common_s2 import *
from evaluate import videos,read_csv,METRICS
from evaluation_support import key

def parent_root(qp):
    return Path(load(GAIN_SOURCE/'protocol.json')['QP9_SOURCE'])/'evaluation/s05' if qp==9 else GAIN_SOURCE/'evaluation/s05'

def validate_legacy(root,v,qp,expected_lora,gain=None):
    p=load(root/f'points/{key(v,qp)}.json');r=p['result']
    assert p['status']=='PASS' and p['lora_checkpoint_sha256']==expected_lora and p['baseline_sha256']==BASELINE_SHA and p['pilot_sha256']==PILOT_SHA
    assert (r['domain'],r['sequence'],r['split'],int(r['QP']))==(v['domain'],v['sequence'],v['split'],qp)
    assert r['source_RGB_sha256']==v['rgb_sha256'] and r['source_sha256']==v['source_sha256'] and p['source_frame_indices']==v['source_frame_indices']
    stream=Path(p['stream_path']);assert sha(stream)==r['bitstream_sha256'] and stream.stat().st_size==r['real_bytes']==p['bytes_consumed']
    assert p['independent_decode'] and p['real_RANS'] and p['model_hashes_unchanged'] and p['TCG_reset']
    fr=read_csv(root/f'frames/{key(v,qp)}.csv');tr=read_csv(root/f'transitions/{key(v,qp)}.csv');ct=read_csv(root/f'codec_trace/{key(v,qp)}.csv')
    assert len(fr)==len(ct)==64 and len(tr)==63 and [int(x['frame']) for x in fr]==list(range(64))
    assert [int(x['actual_qp']) for x in fr]==p['actual_qps'] and [x['source_frame_RGB_sha256'] for x in fr]==v['frame_rgb_sha256']
    assert sum(int(x['real_bits']) for x in fr)==8*r['real_bytes'] and int(fr[0]['real_bits'])>0
    assert hashlib.sha256(''.join(x['reconstruction_tensor_sha256'] for x in fr).encode()).hexdigest()==p['reconstruction_sha256']
    assert r['bpp']==8*r['real_bytes']/(64*v['width']*v['height'])
    mse=sum(float(x['pixel_SSE']) for x in fr)/(64*v['height']*v['width']*3)
    assert abs(-10*math.log10(max(mse,1e-15))-r['PSNR'])<1e-9
    for m in ['LPIPS','DISTS','MS_SSIM']:assert abs(sum(float(x[m]) for x in fr)/64-r[m])<1e-10
    for m in ['FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']:assert abs(sum(float(x[m]) for x in tr)/63-r[m])<1e-10
    assert all(math.isfinite(float(r[m])) for m in METRICS)
    assert p['metric_hashes']==load(ORIGINAL_EVAL/'metric_runtime_rank0.json')['hashes']
    if gain is not None:
        assert p['inference_gain']==gain
        stats=read_csv(root/f'masks/{key(v,qp)}.csv');ks=[k for k in stats[-1] if k.endswith('_inference_gain')]
        assert len(ks)==5 and len(stats)==63 and all(float(x[k])==gain for x in stats for k in ks)
    return p,fr,tr,ct

def validate_reused(root,v,qp,candidate,checkpoint):
    p=load(root/f'points/{key(v,qp)}.json');pro=p['reuse_provenance'];source=Path(pro['source_point'])
    assert sha(source)==pro['source_point_SHA256']
    original=load(source)
    assert {k:v for k,v in p.items() if k not in ['measurement_reused','reuse_provenance']}==original
    for name,h in pro['record_file_SHA256'].items():assert sha(root/name/(key(v,qp)+'.csv'))==h==sha(source.parents[1]/name/(key(v,qp)+'.csv'))
    expected=INIT_SHA if candidate=='lora' else None
    result=validate_legacy(source.parents[1],v,qp,expected,.5 if candidate=='lora' else None)
    assert sha(checkpoint)==INIT_SHA if candidate=='lora' else checkpoint is None
    return p,result[1],result[2],result[3]

def copy_point(src,root,v,qp):
    record=load(src/f'points/{key(v,qp)}.json');files={}
    for folder in ['frames','transitions','codec_trace','masks']:
        source=src/folder/(key(v,qp)+'.csv');dest=root/folder/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(dest)==sha(source)
        else:shutil.copyfile(source,dest)
        files[folder]=sha(source)
    record['measurement_reused']=True
    record['reuse_provenance']=dict(source_point=str(src/f'points/{key(v,qp)}.json'),source_point_SHA256=sha(src/f'points/{key(v,qp)}.json'),
        record_file_SHA256=files,new_decode_performed=False,checkpoint_sha256=record['lora_checkpoint_sha256'],
        source_protocol_sha256=record['protocol_sha256'],source_code_sha256=record['source_manifest_sha256'])
    dump(root/f'points/{key(v,qp)}.json',record);return record

def check_initial_probes():
    from evaluation_support import validate_point
    v=sorted(videos('development'),key=lambda x:x['sequence'])[0];records=[]
    for qp in [0,5,9]:
        new=validate_point(ROOT/'engineering/A2000_s05_dev',v,qp,'lora',INIT);old=validate_legacy(parent_root(qp),v,qp,INIT_SHA,.5)
        assert new[0]['reconstruction_sha256']==old[0]['reconstruction_sha256']
        for field in ['reconstruction_tensor_sha256','decoded_RGB_tensor_sha256']:assert [r[field] for r in new[1]]==[r[field] for r in old[1]]
        assert new[3]==old[3] and new[0]['actual_qps']==old[0]['actual_qps'] and new[0]['metric_hashes']==old[0]['metric_hashes']
        error={m:abs(new[0]['result'][m]-old[0]['result'][m]) for m in METRICS}
        assert all(e<(1e-9 if m in ['PSNR','kbps'] else 1e-10) for m,e in error.items())
        records.append(dict(QP=qp,sequence=v['sequence'],reconstruction_exact=True,RGB_exact=True,metric_absolute_errors=error))
    dump(ROOT/'audits/A2000_s05_initial_reproduction.json',dict(status='PASS',additional_step=0,source_A2000_SHA256=INIT_SHA,s_eval=.5,full64frames_each=True,records=records,time=time.time()))

def reuse_development_reference():
    verify_sources();assert load(ROOT/'audits/A2000_s05_initial_reproduction.json')['status']=='PASS'
    root=eval_dir('development',group='baseline');expected={}
    for r in read_csv(GAIN_SOURCE/'metrics_s05_all_qps.csv'):expected[r['sequence'],int(r['QP'])]=r
    assert len(expected)==96
    rows=[]
    for v in videos('development'):
        for qp in [0,5,9]:
            item=validate_legacy(parent_root(qp),v,qp,INIT_SHA,.5)[0]
            assert all(float(expected[v['sequence'],qp][m])==item['result'][m] for m in METRICS)
            rows.append(copy_point(parent_root(qp),root,v,qp)['result'])
    write_csv(root/'metrics.csv',rows)
    dump(root/'integrity.json',dict(status='PASS',points=96,measurement_reused=True,new_decode_performed=False,reference='A2000 s_eval0.5',source_A2000_SHA256=INIT_SHA,time=time.time()))

def check_MCL_probe():
    from evaluation_support import validate_point
    v=load(ROOT/'manifests/MCL_JCV_30.json')[0];items={}
    for model,cp in [('baseline',None),('original',None),('lora',INIT)]:items[model]=validate_point(ROOT/f'engineering/MCL_probe_{model}',v,5,model,cp,False)
    base=items['baseline']
    for name,item in items.items():
        assert item[0]['fresh_encode'] and item[0]['codec_encoder_sync_independent_features_equal']
        assert item[0]['result']['bitstream_sha256']==base[0]['result']['bitstream_sha256'] and item[0]['result']['real_bytes']==base[0]['result']['real_bytes']
        assert item[3]==base[3]
    vdev=sorted(videos('development'),key=lambda x:x['sequence'])[0]
    parent_components=load(parent_root(9)/f'points/{key(vdev,9)}.json')['component_hashes']
    assert items['baseline'][0]['component_hashes']==items['lora'][0]['component_hashes']==parent_components
    dump(ROOT/'audits/MCL_real_IP_probe.json',dict(status='PASS',sequence=v['sequence'],external_QP=5,frames=64,first_I_real_bits=int(base[1][0]['real_bits']),
        Original_TCG_A2000_independent_fresh_encodes=True,all_stream_SHA_bytes_bpp_codec_traces_equal=True,native_RGB_bypass=False,
        quality_scoring_performed=False,frozen_components_match_audited_parent=True,time=time.time()))

def reuse_TCG_legacy_final():
    assert load(ROOT/'selection.json')['status']=='FROZEN_BEFORE_FINAL_TEST' and load(ROOT/'audits/MCL_real_IP_probe.json')['status']=='PASS'
    source=SOURCE/'common_baseline/evaluation/final_baseline';root=ROOT/'evaluation/final_TCG_B1500';rows=[]
    assert load(source/'integrity.json')['status']=='PASS'
    for v in load(ROOT/'manifests/final_videos.json'):
        for qp in [0,5,9]:
            validate_legacy(source,v,qp,None)
            rows.append(copy_point(source,root,v,qp)['result'])
    dump(ROOT/'audits/TCG_legacy_final_reuse.json',dict(status='PASS',points=135,source=str(source),model='B+1500 FP32-TCG no LoRA',matched_native_RGB_precision_code_and_streams=True,time=time.time()))
