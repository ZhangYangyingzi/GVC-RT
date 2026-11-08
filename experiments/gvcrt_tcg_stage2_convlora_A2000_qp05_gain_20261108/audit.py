"""Per-video/per-QP identities, strict endpoint checks and audited QP9 reuse."""
from common_s2 import *
from evaluate_core import videos,read_csv,METRICS
BASE=SOURCE/'common_baseline/evaluation/development_baseline'
FULL=SOURCE/'A_control/evaluation/development_lora_2000'
TARGETS=sorted([f'stage3.blocks.{i}.dc.2' for i in range(4)]+['head'])
QUALITY=['PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']
def video_list():return sorted(videos('development'),key=lambda v:v['sequence'])
def key(v,qp):return f'{v["domain"]}_{v["sequence"]}_qp{qp}'
def paired_key(r):return (r['domain'],r['sequence'],r['split'],int(r['QP']))
def gain_batch(g,qp):
    assert qp in (0,5)
    return 'evaluation/s05' if g==.5 else f'audits/smoke_qp{qp}_s{g:g}'
def task_list():return [(v,q) for v in video_list() for q in [0,5]]

def validate_point(root,v,qp,gain=None,identity_root=None):
    assert qp in (0,5,9)
    identity_root=ROOT if identity_root is None else Path(identity_root)
    p=load(root/f'points/{key(v,qp)}.json');r=p['result']
    assert p['status']=='PASS' and paired_key(r)==(v['domain'],v['sequence'],'development',qp)
    assert p['baseline_sha256']==BASELINE_SHA and p['pilot_sha256']==PILOT_SHA
    assert p['source_frame_indices']==v['source_frame_indices'] and r['source_RGB_sha256']==v['rgb_sha256'] and r['source_sha256']==v['source_sha256']
    stream=Path(p['stream_path']);assert sha(stream)==r['bitstream_sha256'] and stream.stat().st_size==r['real_bytes']==p['bytes_consumed']
    assert p['independent_decode'] and p['real_RANS'] and p['model_hashes_unchanged'] and p['TCG_reset']
    assert not p['fresh_encode'] and p['input_stream_reused'] and p['historical_codec_feature_trace_checked']
    fr=read_csv(root/f'frames/{key(v,qp)}.csv');tr=read_csv(root/f'transitions/{key(v,qp)}.csv');ct=read_csv(root/f'codec_trace/{key(v,qp)}.csv')
    assert len(fr)==len(ct)==64 and len(tr)==63
    assert [int(x['frame']) for x in fr]==list(range(64))
    assert [int(x['to_frame']) for x in tr]==list(range(1,64))
    assert [x['source_frame_RGB_sha256'] for x in fr]==v['frame_rgb_sha256']
    assert [int(x['actual_qp']) for x in fr]==p['actual_qps']
    assert sum(int(x['real_bits']) for x in fr)==8*r['real_bytes']
    assert r['bpp']==8*r['real_bytes']/(64*v['height']*v['width'])
    assert abs(r['kbps']-8*r['real_bytes']*v['fps']/64000)<1e-9
    assert hashlib.sha256(''.join(x['reconstruction_tensor_sha256'] for x in fr).encode()).hexdigest()==p['reconstruction_sha256']
    mse=sum(float(x['pixel_SSE']) for x in fr)/(64*v['height']*v['width']*3)
    assert abs(-10*math.log10(max(mse,1e-15))-r['PSNR'])<1e-9
    for m in ['MS_SSIM','LPIPS','DISTS']:assert abs(sum(float(x[m]) for x in fr)/64-r[m])<1e-10
    for m in ['FloLPIPS','WarpingError','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']:assert abs(sum(float(x[m]) for x in tr)/63-r[m])<1e-10
    assert r['WarpingError']==r['WarpingError_ReconFlow_PWC'] and all(math.isfinite(float(r[m])) for m in METRICS)
    if gain is not None:
        assert p['inference_gain']==gain and p['lora_checkpoint_sha256']==CHECKPOINT_SHA
        assert p['cache_identity']==dict(checkpoint_sha256=CHECKPOINT_SHA,inference_gain=gain,external_QP=qp,domain=v['domain'],sequence=v['sequence'],split='development',
            frame_indices=v['source_frame_indices'],source_sha256=v['source_sha256'],source_RGB_sha256=v['rgb_sha256'],
            protocol_sha256=sha(identity_root/'protocol.json'),code_sources_sha256=sha(identity_root/'sources.json'))
        assert not p['optimizer_created'] and p['training_updates']==0 and p['gain_targets']==TARGETS
        assert p['source_manifest_sha256']==sha(identity_root/'sources.json') and p['protocol_sha256']==sha(identity_root/'protocol.json')
        if identity_root==ROOT:
            assert p['record_file_sha256']=={name:sha(root/name/(key(v,qp)+'.csv')) for name in ['frames','transitions','codec_trace','masks']}
        stats=read_csv(root/f'masks/{key(v,qp)}.csv');assert len(stats)==63
        for row in stats:
            gainkeys=[k for k in row if k.endswith('_inference_gain')];assert len(gainkeys)==5
            assert all(float(row[k])==gain for k in gainkeys)
            assert all(row[k]=='torch.float32' for k in row if k.endswith(('_gain_multiply_dtype','_gain_add_dtype','_parameter_dtype')))
        assert all(int(stats[-1][n+'_gain_calls'])>=63 for n in TARGETS)
    return p,fr,tr,ct

def compare_streams(v,items):
    points=[x[0] for x in items];rs=[p['result'] for p in points]
    for field in ['domain','sequence','split','QP','real_bytes','bpp','bitstream_sha256','source_sha256','source_RGB_sha256']:
        assert len({r[field] for r in rs})==1,('PAIR_IDENTITY_MISMATCH',v['sequence'],field)
    assert all(p['actual_qps']==points[0]['actual_qps'] and p['source_frame_indices']==points[0]['source_frame_indices'] for p in points)
    assert all(x[3]==items[0][3] for x in items)
    assert all([r['GT_flow_sha256'] for r in x[2]]==[r['GT_flow_sha256'] for r in items[0][2]] for x in items)

def smoke_check():
    verify_sources();v=video_list()[0];records=[]
    for qp in [0,5]:
        m={g:validate_point(ROOT/gain_batch(g,qp),v,qp,g) for g in [0.,.5,1.]}
        historical={0.:validate_point(BASE,v,qp),1.:validate_point(FULL,v,qp)}
        compare_streams(v,[*historical.values(),*m.values()])
        for gain,original in historical.items():
            actual=m[gain]
            assert actual[0]['reconstruction_sha256']==original[0]['reconstruction_sha256'],('ENDPOINT_RECONSTRUCTION_MISMATCH',qp,gain)
            for field in ['reconstruction_tensor_sha256','decoded_RGB_tensor_sha256']:
                assert [r[field] for r in actual[1]]==[r[field] for r in original[1]],('ENDPOINT_FRAME_MISMATCH',qp,gain,field)
            errors={metric:abs(actual[0]['result'][metric]-original[0]['result'][metric]) for metric in METRICS}
            assert all(e<(1e-9 if metric in ['PSNR','kbps'] else 1e-10) for metric,e in errors.items())
            records.append(dict(external_QP=qp,inference_gain=gain,reconstruction_tensor_SHA_exact=True,RGB_SHA_exact=True,metric_absolute_errors=errors))
        assert len({x[0]['reconstruction_sha256'] for x in m.values()})==3
    dump(ROOT/'audits/smoke.json',dict(status='PASS',sequence=v['sequence'],selection='fixed lexicographic first development video',
        QPs=[0,5],full_decodes=6,frames_each=64,endpoint_verification_decodes=4,half_gain_smoke_points_counted_once=2,
        five_branches_FP32_gain_and_add=True,no_weight_changes=True,no_optimizer=True,inherited_tolerances_unchanged=True,records=records,time=time.time()))

def copy_records(base,dest,v,qp):
    records=[]
    for folder,suffix in [('points','.json'),('frames','.csv'),('transitions','.csv'),('codec_trace','.csv'),('masks','.csv')]:
        src=base/folder/(key(v,qp)+suffix);out=dest/folder/src.name;out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():assert sha(out)==sha(src),'REUSED_RECORD_CONFLICT'
        else:shutil.copyfile(src,out)
        records.append(dict(source=str(src),source_SHA256=sha(src),destination=str(out.relative_to(ROOT))))
    return records

def reuse_sources():
    verify_sources();assert load(ROOT/'audits/smoke.json')['status']=='PASS'
    assert load(QP9_SOURCE/'final_integrity.json')['status']=='PASS' and load(QP9_SOURCE/'audits/smoke.json')['status']=='PASS'
    endpoints=[];half9=[];manifest=[]
    expected_metrics=load(ORIGINAL_EVAL/'metric_runtime_rank0.json')['hashes']
    for gain,base in [(0.,BASE),(1.,FULL)]:
        rows=read_csv(base/'metrics.csv');assert len(rows)==96
        indexed={paired_key(r):r for r in rows};assert len(indexed)==96
        for v in video_list():
            for qp in [0,5,9]:
                p,fr,tr,ct=validate_point(base,v,qp);r=dict(p['result']);csvrow=indexed[paired_key(r)]
                assert all(float(csvrow[m])==r[m] for m in METRICS) and p['metric_hashes']==expected_metrics
                assert p['lora_checkpoint_sha256']==(None if gain==0 else CHECKPOINT_SHA)
                if gain==1:
                    assert p['protocol_sha256']==sha(SOURCE/'protocol.json') and p['source_manifest_sha256']==sha(SOURCE/'source_manifest.json')
                else:
                    inherited=load(SOURCE/'common_baseline/evaluation/development_baseline/reuse_manifest.json')
                    assert inherited['status']=='PASS' and p['protocol_sha256']==inherited['proof']['reference_protocol_sha256']
                    assert p['source_manifest_sha256']==inherited['proof']['reference_source_manifest_sha256']
                files=copy_records(base,ROOT/f'endpoints/s{gain:g}',v,qp)
                r.update(inference_gain=gain,measurement_reused=True,independent_decode_performed_this_run=False,
                    measurement_source=str(base/'metrics.csv'),measurement_source_SHA256=sha(base/'metrics.csv'),
                    model='B+1500 matched-FP32, no LoRA contribution' if gain==0 else 'A2000 full LoRA strength',
                    diagnostic_checkpoint_SHA256=CHECKPOINT_SHA)
                endpoints.append(r);manifest.append(dict(kind='endpoint',gain=gain,key=list(paired_key(r)),files=files,
                    actual_checkpoint_SHA256=p['lora_checkpoint_sha256'],TCG_SHA256=p['baseline_sha256'],metric_hashes=p['metric_hashes'],frame_indices=p['source_frame_indices']))
    qp9csv=read_csv(QP9_SOURCE/'metrics_s05.csv');assert len(qp9csv)==32
    indexed={paired_key(r):r for r in qp9csv};assert len(indexed)==32
    for v in video_list():
        item=validate_point(QP9_SOURCE/'evaluation/s05',v,9,.5,QP9_SOURCE);p=item[0];r=dict(p['result'])
        compare_streams(v,[validate_point(BASE,v,9),item,validate_point(FULL,v,9)])
        assert all(float(indexed[paired_key(r)][m])==r[m] for m in METRICS) and p['metric_hashes']==expected_metrics
        files=copy_records(QP9_SOURCE/'evaluation/s05',ROOT/'reused_qp9/s05',v,9)
        r.update(inference_gain=.5,measurement_reused=True,independent_decode_performed_this_run=False,measurement_source=str(QP9_SOURCE/'metrics_s05.csv'),
            measurement_source_SHA256=sha(QP9_SOURCE/'metrics_s05.csv'),checkpoint_SHA256=CHECKPOINT_SHA,model='A2000 half LoRA strength')
        half9.append(r);manifest.append(dict(kind='QP9_half',gain=.5,key=list(paired_key(r)),files=files,checkpoint_SHA256=p['lora_checkpoint_sha256'],
            protocol_SHA256=p['protocol_sha256'],code_sources_SHA256=p['source_manifest_sha256'],metric_hashes=p['metric_hashes'],frame_indices=p['source_frame_indices']))
    assert len(endpoints)==192 and len(half9)==32
    write_csv(ROOT/'endpoint_metrics.csv',endpoints);write_csv(ROOT/'metrics_s05_qp9_reused.csv',half9)
    dump(ROOT/'audits/source_reuse.json',dict(status='PASS',endpoints_reused=192,QP9_half_reused=32,measurements_redecoded_this_run=False,records=manifest,time=time.time()))

def freeze_sources():
    env();assert not (ROOT/'sources.json').exists(),'SOURCE_FREEZE_EXISTS_RESUME_RUNNER'
    import torch
    q=load(QP9_SOURCE/'sources.json');assert q['status']=='FROZEN'
    assert load(QP9_SOURCE/'final_integrity.json')['status']=='PASS' and load(QP9_SOURCE/'status.json')['status']=='COMPLETE'
    protected={}
    def protect(p,expected=None):
        p=Path(p);h=sha(p)
        if expected is not None:assert h==expected,('SOURCE_SHA_CHANGED',str(p))
        if str(p) in protected:assert protected[str(p)]==h
        protected[str(p)]=h
    for n,h in q['local_files'].items():protect(QP9_SOURCE/n,h)
    for n,h in q['protected_source_files'].items():protect(n,h)
    protect(QP9_SOURCE/'protocol.json',q['protocol_sha256']);protect(QP9_SOURCE/'sources.json')
    for item in load(QP9_SOURCE/'artifact_manifest.json')['files']:protect(QP9_SOURCE/item['path'],item['sha256'])
    protect(QP9_SOURCE/'artifact_manifest.json')
    for p in QP9_SOURCE.glob('*.py'):protect(p)
    for name in ['final_integrity.json','status.json','summary.json','metrics_s05.csv','metrics_combined.csv','endpoint_metrics.csv',
                 'stream_identity_checks.csv','audits/smoke.json','audits/endpoint_reuse.json','audits/final_recheck.json']:
        protect(QP9_SOURCE/name)
    # Both historical remote receipts are kept exactly as found, including existing user modifications.
    for base in [SOURCE,QP9_SOURCE]:protect(base/'publication_remote_receipt.json')
    assert CHECKPOINT_SHA=='8a69765befa6ed267d0a95d3932cd63d9513c1da9818fd2126d8bc1c165f5e4e'
    protect(CHECKPOINT,CHECKPOINT_SHA)
    cp=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    assert cp['group']=='A_control' and cp['committed_optimizer_updates']==2000 and cp['TCG_cumulative_step_frozen_at']==6500
    assert cp['baseline_sha256']==BASELINE_SHA and cp['pilot_sha256']==PILOT_SHA and not cp['engineering_only']
    assert len(cp['lora_state'])==10 and sum(t.numel() for t in cp['lora_state'].values())==59392
    assert sha(ROOT/'convlora.py')==sha(QP9_SOURCE/'convlora.py'),'GAIN_ARITHMETIC_CHANGED'
    assert sha(ROOT/'fp32_island.py')==sha(QP9_SOURCE/'fp32_island.py') and sha(ROOT/'data_reader.py')==sha(QP9_SOURCE/'data_reader.py')
    for v in video_list():
        for qp in [0,5,9]:
            compare_streams(v,[validate_point(BASE,v,qp),validate_point(FULL,v,qp)])
            for base in [BASE,FULL]:
                for folder,suffix in [('points','.json'),('frames','.csv'),('transitions','.csv'),('codec_trace','.csv'),('masks','.csv')]:protect(base/folder/(key(v,qp)+suffix))
        item=validate_point(QP9_SOURCE/'evaluation/s05',v,9,.5,QP9_SOURCE)
        compare_streams(v,[validate_point(BASE,v,9),item,validate_point(FULL,v,9)])
        for folder,suffix in [('points','.json'),('frames','.csv'),('transitions','.csv'),('codec_trace','.csv'),('masks','.csv')]:protect(QP9_SOURCE/f'evaluation/s05/{folder}'/(key(v,9)+suffix))
    local={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and (p.suffix=='.py' or str(p.relative_to(ROOT)) in q['local_files'])}
    dump(ROOT/'sources.json',dict(status='FROZEN',local_files=local,protected_source_files=protected,protocol_sha256=sha(ROOT/'protocol.json'),
        template_protocol_SHA256=sha(QP9_SOURCE/'protocol.json'),template_sources_SHA256=sha(QP9_SOURCE/'sources.json'),checkpoint_SHA256=CHECKPOINT_SHA,
        expected_component_SHAs_read_from_existing_protocol=True,new_formal_QPs=[0,5],new_formal_points=64,QP9_reuse_points=32,time=time.time()))
    dump(ROOT/'audits/source_preflight.json',dict(status='PASS',protected_files=len(protected),checkpoint_SHA256=CHECKPOINT_SHA,
        expected_SHAs={name:sha(p) for name,p in [('pilot',PILOT),('B1500',BASELINE),('I_codec',PROJECT/'checkpoints/GVC-RT_I.pt'),('P_codec',PROJECT/'checkpoints/GVC-RT_P.pt')]},
        endpoint_points_validated=192,QP9_half_points_validated=32,identical_LoRA_and_FP32_implementation=True,training_updates=0,optimizer_created=False,time=time.time()))
    verify_sources();print('QP05_FIXED_SOURCES_AND_QP9_POINT_AUDIT_PASS')

if __name__=='__main__':freeze_sources()
