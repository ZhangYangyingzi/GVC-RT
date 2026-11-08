"""Exact source/endpoint identities and inherited metric aggregation tolerances."""
from common_s2 import *
from evaluate_core import videos,read_csv,METRICS

BASE=SOURCE/'common_baseline/evaluation/development_baseline'
FULL=SOURCE/'A_control/evaluation/development_lora_2000'
TARGETS=sorted([f'stage3.blocks.{i}.dc.2' for i in range(4)]+['head'])
QUALITY=['PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']

def video_list():return sorted(videos('development'),key=lambda v:v['sequence'])
def key(v):return f'{v["domain"]}_{v["sequence"]}_qp9'
def paired_key(r):return (r['domain'],r['sequence'],r['split'],int(r['QP']))
def gain_batch(g):return {0.:'audits/smoke_s0',1.:'audits/smoke_s1',.5:'evaluation/s05'}[g]

def validate_point(root,v,gain=None):
    p=load(root/f'points/{key(v)}.json');r=p['result']
    assert p['status']=='PASS' and paired_key(r)==(v['domain'],v['sequence'],'development',9)
    assert p['baseline_sha256']==BASELINE_SHA and p['pilot_sha256']==PILOT_SHA
    assert p['source_frame_indices']==v['source_frame_indices'] and r['source_RGB_sha256']==v['rgb_sha256'] and r['source_sha256']==v['source_sha256']
    stream=Path(p['stream_path']);assert sha(stream)==r['bitstream_sha256'] and stream.stat().st_size==r['real_bytes']==p['bytes_consumed']
    assert p['independent_decode'] and p['real_RANS'] and p['model_hashes_unchanged'] and p['TCG_reset']
    assert not p['fresh_encode'] and p['input_stream_reused'] and p['historical_codec_feature_trace_checked']
    fr=read_csv(root/f'frames/{key(v)}.csv');tr=read_csv(root/f'transitions/{key(v)}.csv');ct=read_csv(root/f'codec_trace/{key(v)}.csv')
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
        assert p['cache_identity']==dict(checkpoint_sha256=CHECKPOINT_SHA,inference_gain=gain,external_QP=9,domain=v['domain'],sequence=v['sequence'],split='development',
            frame_indices=v['source_frame_indices'],source_sha256=v['source_sha256'],source_RGB_sha256=v['rgb_sha256'],
            protocol_sha256=sha(ROOT/'protocol.json'),code_sources_sha256=sha(ROOT/'sources.json'))
        assert not p['optimizer_created'] and p['training_updates']==0 and p['gain_targets']==TARGETS
        assert p['source_manifest_sha256']==sha(ROOT/'sources.json') and p['protocol_sha256']==sha(ROOT/'protocol.json')
        stats=read_csv(root/f'masks/{key(v)}.csv');assert len(stats)==63
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
    verify_sources();v=video_list()[0];measurements={g:validate_point(ROOT/gain_batch(g),v,g) for g in [0.,1.,.5]}
    baseline=validate_point(BASE,v);full=validate_point(FULL,v);compare_streams(v,[baseline,full,*measurements.values()]);records=[]
    for g,original in [(0.,baseline),(1.,full)]:
        actual=measurements[g]
        assert actual[0]['reconstruction_sha256']==original[0]['reconstruction_sha256'],('ENDPOINT_RECONSTRUCTION_MISMATCH',g)
        assert [r['reconstruction_tensor_sha256'] for r in actual[1]]==[r['reconstruction_tensor_sha256'] for r in original[1]]
        assert [r['decoded_RGB_tensor_sha256'] for r in actual[1]]==[r['decoded_RGB_tensor_sha256'] for r in original[1]]
        errors={m:abs(actual[0]['result'][m]-original[0]['result'][m]) for m in METRICS}
        assert all(e<(1e-9 if m in ['PSNR','kbps'] else 1e-10) for m,e in errors.items()),('ENDPOINT_METRIC_MISMATCH',g,errors)
        records.append(dict(inference_gain=g,source_point=str((BASE if g==0 else FULL)/f'points/{key(v)}.json'),reconstruction_exact=True,metric_absolute_errors=errors))
    assert len({p[0]['reconstruction_sha256'] for p in measurements.values()})==3,'DIFFERENT_GAIN_RECONSTRUCTIONS_REQUIRED'
    dump(ROOT/'audits/smoke.json',dict(status='PASS',sequence=v['sequence'],selection='first sequence in fixed lexicographic development order',external_QP=9,
        full_frames_each=64,all_five_branches_checked=True,FP32_gain_multiply_and_add=True,weights_unchanged=True,endpoints=records,
        new_half_point_reusable=True,original_tolerances_unchanged=True,time=time.time()))

def reuse_endpoints():
    verify_sources();assert load(ROOT/'audits/smoke.json')['status']=='PASS'
    metrics=[];manifest=[]
    for gain,base in [(0.,BASE),(1.,FULL)]:
        rows=read_csv(base/'metrics.csv');assert len(rows)==96
        bykey={paired_key(r):r for r in rows if int(r['QP'])==9};assert len(bykey)==32
        for v in video_list():
            p,fr,tr,ct=validate_point(base,v);r=dict(p['result']);match=bykey[paired_key(r)]
            assert all(float(match[m])==float(r[m]) for m in METRICS)
            assert p['lora_checkpoint_sha256']==(None if gain==0 else CHECKPOINT_SHA)
            expected=load(ORIGINAL_EVAL/'metric_runtime_rank0.json')['hashes'];assert p['metric_hashes']==expected
            if gain==1:
                assert p['protocol_sha256']==sha(SOURCE/'protocol.json') and p['source_manifest_sha256']==sha(SOURCE/'source_manifest.json')
            else:
                # Baseline was previously audited and reused from the reference experiment.
                inherited=load(SOURCE/'common_baseline/evaluation/development_baseline/reuse_manifest.json')
                assert inherited['status']=='PASS' and p['protocol_sha256']==inherited['proof']['reference_protocol_sha256']
                assert p['source_manifest_sha256']==inherited['proof']['reference_source_manifest_sha256']
            files=[];dest=ROOT/('endpoints/s0' if gain==0 else 'endpoints/s1')
            for folder,suffix in [('points','.json'),('frames','.csv'),('transitions','.csv'),('codec_trace','.csv'),('masks','.csv')]:
                src=base/folder/(key(v)+suffix);out=dest/folder/src.name;out.parent.mkdir(parents=True,exist_ok=True)
                if out.exists():assert sha(out)==sha(src)
                else:shutil.copyfile(src,out)
                files.append(dict(source_path=str(src),source_sha256=sha(src),destination=str(out.relative_to(ROOT))))
            r.update(inference_gain=gain,measurement_reused=True,independent_decode_performed_this_run=False,endpoint_source=str(base/'metrics.csv'),
                A2000_diagnostic_checkpoint_sha256=CHECKPOINT_SHA,model='B+1500 matched-FP32, no LoRA' if gain==0 else 'A2000 full LoRA strength')
            metrics.append(r);manifest.append(dict(inference_gain=gain,key=list(paired_key(r)),checkpoint_sha256=p['lora_checkpoint_sha256'],TCG_checkpoint_sha256=p['baseline_sha256'],
                frame_indices=p['source_frame_indices'],source_RGB_sha256=r['source_RGB_sha256'],metric_hashes=p['metric_hashes'],files=files))
    write_csv(ROOT/'endpoint_metrics.csv',metrics)
    dump(ROOT/'audits/endpoint_reuse.json',dict(status='PASS',reused_points=64,recomputed_points=0,records=manifest,time=time.time()))

def freeze_sources():
    env();assert not (ROOT/'sources.json').exists(),'ALREADY_FROZEN_USE_RUNNER'
    import torch
    old=load(SOURCE/'source_manifest.json');assert old['status']=='FROZEN'
    protected={}
    def protect(path,expected=None):
        path=Path(path);h=sha(path)
        if expected is not None:assert h==expected,('SOURCE_SHA_MISMATCH',str(path))
        protected[str(path)]=h
    protect(CHECKPOINT,CHECKPOINT_SHA);protect(PILOT,PILOT_SHA);protect(BASELINE,BASELINE_SHA)
    protect(PROJECT/'checkpoints/GVC-RT_I.pt','c8a840415cc028307131c66f68076b7e9f6fcc4db7fed423c62e8b202f83a22d')
    protect(PROJECT/'checkpoints/GVC-RT_P.pt','7a89bd342a1484b3da7dfc3f03834ff81e0e4180e02f8a286dec0d1025c99f4e')
    for n,h in old['code'].items():protect(SOURCE/n,h)
    for n,h in old['immutable_manifests'].items():protect(SOURCE/n,h)
    for n,h in old['inherited_external_sources'].items():protect(n,h)
    for n,h in old['engineering_audits'].items():protect(SOURCE/f'audits/{n}.json',h)
    protect(SOURCE/'protocol.json',old['protocol_sha256']);protect(SOURCE/'source_manifest.json')
    protect(SOURCE/'common_initialization/untrained_common.pt',old['common_initialization_sha256'])
    for n in ['A_control/selection.json','B_expanded/selection.json','selection_manifest.json','final_integrity.json','baseline_provenance.json',
              'A_control/baseline_provenance.json','common_baseline/evaluation/development_baseline/reuse_manifest.json']:
        protect(SOURCE/n)
    cp=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    assert cp['group']=='A_control' and cp['committed_optimizer_updates']==2000 and cp['TCG_cumulative_step_frozen_at']==6500
    assert cp['baseline_sha256']==BASELINE_SHA and cp['pilot_sha256']==PILOT_SHA and cp['protocol_sha256']==old['protocol_sha256']
    assert cp['source_manifest_sha256']==sha(SOURCE/'source_manifest.json') and not cp['engineering_only']
    assert len(cp['lora_state'])==10 and sum(t.numel() for t in cp['lora_state'].values())==59392
    records=[]
    for v in video_list():
        for base in [BASE,FULL]:
            validate_point(base,v)
            for folder,suffix in [('points','.json'),('frames','.csv'),('transitions','.csv'),('codec_trace','.csv'),('masks','.csv')]:protect(base/folder/(key(v)+suffix))
        compare_streams(v,[validate_point(BASE,v),validate_point(FULL,v)])
        records.append(dict(sequence=v['sequence'],domain=v['domain'],split='development',QP=9,source_sha256=v['source_sha256'],RGB_sha256=v['rgb_sha256'],frame_indices=v['source_frame_indices']))
    for base in [BASE,FULL]:protect(base/'metrics.csv')
    local={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and (p.suffix=='.py' or str(p.relative_to(ROOT)) in ['bitstream_manifest.json','vendor/sealed/manifests/evaluation_rows.json','vendor/sealed/repairs/native_geometry_v1/geometry_errata.json'])}
    dump(ROOT/'sources.json',dict(status='FROZEN',local_files=local,protected_source_files=protected,protocol_sha256=sha(ROOT/'protocol.json'),
        source_checkpoint_sha256=CHECKPOINT_SHA,checkpoint_group='A_control',checkpoint_LoRA_step=2000,TCG_step=6500,videos=records,time=time.time()))
    dump(ROOT/'audits/source_preflight.json',dict(status='PASS',checkpoint_sha256=CHECKPOINT_SHA,LoRA_parameters=59392,targets=TARGETS,
        protected_source_files=len(protected),endpoint_points_checked=64,training_updates=0,optimizer_created=False,time=time.time()))
    verify_sources();print('SOURCE_AND_FIXED_CHECKPOINT_PREFLIGHT_PASS')

if __name__=='__main__':freeze_sources()
