"""Freeze historical provenance, self-tests, and exact old RD queue."""
from audit_io import *
import itertools, subprocess
def run(a): return subprocess.check_output(a,cwd=REPO,text=True,stderr=subprocess.STDOUT).strip()
def main():
    import numpy as np
    for n in ('logs','audits','tasks','evaluation','manifests','features','bitstreams','cache','new_clip_metrics'): (ROOT/n).mkdir(exist_ok=True)
    assert sha(WEIGHT)==WEIGHT_SHA, 'FID weight mismatch'
    vendor=OLD/'vendor/pytorch-fid'
    assert run(['git','-C',str(vendor),'rev-parse','HEAD'])==OFFICIAL_COMMIT
    assert load(OLD/'final_integrity.json')['status']=='PASS'
    cfg=load(OLD/'evaluation/config.json');dep=load(OLD/'audits/dependencies.json')
    # Hash every frozen dependency, not only file timestamps.
    for p,h in dep.items(): assert sha(p)==h, ('historical dependency changed',p)
    for m in METHODS[1:]: assert sha(cfg['checkpoints'][m]['path'])==cfg['checkpoints'][m]['sha256'],('checkpoint mismatch',m)
    version=[]
    for name in ('adapter.py','io21.py','evaluate.py','fid_statistics.py','evaluation/config.json','protocol.json','manifests/hevc_b.json','manifests/mcljcv.json'):
        p=OLD/name; local=sha(p)
        data=subprocess.check_output(['git','show',REFERENCE+':'+str(p.relative_to(REPO))],cwd=REPO)
        published=hashlib.sha256(data).hexdigest();version.append(dict(path=str(p),local_sha256=local,published_sha256=published,same=local==published))
    assert all(x['same'] for x in version), 'selected historical source differs from published reference; explicit review required'
    old={d:load(OLD/'manifests'/f'{d}.json')['videos'] for d in DATASETS}
    assert [len(old[d]) for d in DATASETS]==[5,30]
    for d in DATASETS: old[d].sort(key=lambda v:v['video_index'])
    dump(ROOT/'old_sources.json',old)
    points=[]
    for d in DATASETS:
        for v in old[d]:
            assert len(v['source_frame_indices'])==64 and v['width']==1920 and v['height']==1080
            for m,q in itertools.product(METHODS,QPS):
                p=point(d,m,v['video_index'],q);r=load(p)
                assert r['source_rgb_sha256']==v['rgb_sha256'] and r['source_frame_indices']==v['source_frame_indices']
                assert r['checkpoint_sha256']==('' if m=='original' else cfg['checkpoints'][m]['sha256'])
                assert r['actual_qps']==load(OLD/'qp_semantics_audit.json')['actual_qps'][str(q)][:64]
                assert r['force_zero_thres']==.12 and r['compression_hash_before']==r['compression_hash_after']==cfg['compression_hash']
                assert r['loaded_receiver_module_hashes']==(cfg['original_receiver_hashes'] if m=='original' else cfg['deployment_receiver_hashes'][m])
                assert r['independent_decode_pass'] and r['bytes_consumed']==r['real_bytes'] and r['real_RANS']
                for k in ('bitstream','feature'): assert sha(r[k+'_path'])==r[k+'_sha256'],(p,k)
                assert r['runtime_audit']['metric_module_hashes']==cfg['metric_module_hashes']
                points.append(dict(dataset=d,video=v['video_index'],method=m,qp=q,path=str(p),sha256=sha(p),bitstream_path=r['bitstream_path'],bitstream_sha256=r['bitstream_sha256'],feature_path=r['feature_path'],feature_sha256=r['feature_sha256']))
    dump(ROOT/'old_points_manifest.json',points)
    cfg.update(no_training=True,expected_points=420,external_qps=list(QPS),datasets=list(DATASETS))
    dump(ROOT/'evaluation/config.json',cfg);dump(ROOT/'qp_semantics_audit.json',load(OLD/'qp_semantics_audit.json'))
    dump(ROOT/'config.json',dict(schema='v630',updates=0,methods=METHODS,qps=QPS,GPUs=[4,5,6,7],reference_commit=REFERENCE,expected_closure_points=420,seed=20261030))
    dump(ROOT/'protocol.json',dict(schema='v630',updates=0,IP=-1,frames=64,padding=[1920,1088],crop=[1920,1080],force_zero_thres=.12,precision='FP16 codec FP32 wrapper and official features; float64 FID ddof1',FID=dict(commit=OFFICIAL_COMMIT,weight_sha256=WEIGHT_SHA,resize='bilinear299 align_corners=False',normalize='2*x-1',dims=2048,relative_L2=1e-4,max_abs='1e-4+1e-4*maxabs(cached)',pooled_tolerance='1e-4+1e-4*max(1,abs(fresh),abs(cached))'),new_start='max(old_source_frame_indices)+65',new_frames=64,rate='all bitstream bytes*8/(64*W*H)',diagnostic=dict(qps=QPS,grid=100,spacing='ln(bpp)',interpolation='PCHIP',extrapolate=False,label='3-QP stability diagnostic, not primary 10-QP comparison'),resource=dict(GPUs=[4,5,6,7],reserve_multiplier=1.25,extra_MiB=2048,CPU_workers=4,threads=2,OOM_retries=1)))
    selected={str(p):sha(p) for p in [OLD/'evaluation/config.json',OLD/'protocol.json',OLD/'final_integrity.json',WEIGHT,*vendor.glob('src/pytorch_fid/*.py'),*(OLD/n for n in ('adapter.py','io21.py','evaluate.py','fid_statistics.py','manifests/hevc_b.json','manifests/mcljcv.json'))]}
    selected.update(dep);selected.update({x['path']:x['sha256'] for x in points})
    for x in points:
        selected[x['bitstream_path']]=x['bitstream_sha256'];selected[x['feature_path']]=x['feature_sha256']
    for d in DATASETS:
        for v in old[d]:
            assert sha(v['source_path'])==v['source_sha256'],('source YUV changed',v['name'])
            selected[v['source_path']]=v['source_sha256']
            for i,h in enumerate(v['frame_png_sha256' if d=='hevc_b' else 'frame_file_sha256']):
                name=f'im{i+1:05d}.png' if d=='hevc_b' else f'frame_{i:06d}.png';p=Path(v['input_dir'])/name
                assert sha(p)==h,('canonical RGB file changed',p);selected[str(p)]=h
        pool=ROOT.parent/('gvcrt_neural_wrapper_v6_27_illm_effectiveness_mcljcv_audit' if d=='mcljcv' else 'gvcrt_neural_wrapper_v6_20_native_i_reference_adaptation')/'features'/f'GT_{d}.npy'
        selected[str(pool)]=sha(pool)
    training={
        'B1_initial':dict(path=str(ROOT.parent/'gvcrt_neural_wrapper_v6_22_long_reference_reconstruction_loss_control/branches/B1_reconstruction/checkpoints/adaptation_step_1000.pt'),sha256='dc76f0e221dc7aaa5e5f45f648722584ce226e65828ce24151f9961985f3f6f7'),
        'A_continue_1000':load(OLD/'branches/A_continue/checkpoint_index.json')['1000'],
        'B_covstats_1000':load(OLD/'branches/B_covstats/checkpoint_index.json')['1000']}
    import torch
    from io21 import tensor_hash
    for m,record in training.items():
        assert sha(record['path'])==record['sha256'],('training checkpoint mismatch',m)
        selected[record['path']]=record['sha256']
        if 'module_hashes' in record:assert record['module_hashes']==cfg['checkpoints'][m]['module_hashes']
        for cp in (record,cfg['checkpoints'][m]):
            payload=torch.load(cp['path'],map_location='cpu',weights_only=True)
            actual={k:tensor_hash(payload[k]) for k in ('wrapper','bridge','generator')}
            assert actual==cfg['checkpoints'][m]['module_hashes'],('checkpoint P/B/G mismatch',m,cp['path'])
            cp['verified_actual_module_hashes']=actual
            del payload
    dump(ROOT/'audits/training_deployment_checkpoints.json',dict(training=training,deployment={m:cfg['checkpoints'][m] for m in METHODS[1:]},file_SHA_are_independently_verified=True))
    selected.update({cfg['checkpoints'][m]['path']:cfg['checkpoints'][m]['sha256'] for m in METHODS[1:]})
    dump(ROOT/'source_hashes.json',dict(hashes=selected,reference_commit=REFERENCE,version_comparison=version,training_checkpoint_records=load(OLD/'binary_artifacts_index.json'),deployment=cfg,training_and_deployment_SHA_are_separate=True))
    dump(ROOT/'audits/dependencies.json',selected)
    dump(ROOT/'audits/protocol_hashes.json',{n:sha(ROOT/n) for n in ('config.json','protocol.json','evaluation/config.json','qp_semantics_audit.json')})
    for d in DATASETS: dump(ROOT/'manifests'/f'{d}.json',dict(videos=old[d]))
    rng=np.random.Generator(np.random.PCG64(20261030));sub=[]
    for vs in itertools.combinations([v['video_index'] for v in old['hevc_b']],3):sub.append(dict(dataset='hevc_b',size=3,videos=list(vs)))
    for size in (10,20):
        seen=set()
        while len(seen)<8:
            ids=tuple(sorted(rng.choice([v['video_index'] for v in old['mcljcv']],size,replace=False).tolist()))
            if ids in seen:continue
            seen.add(ids);sub.append(dict(dataset='mcljcv',size=size,videos=list(ids)))
    dump(ROOT/'subset_manifest.json',dict(seed=20261030,RNG='PCG64',generation_order=[10,20],subsets=sub))
    z=rng.normal(size=(64,2048)).astype('float32')
    tests=dict(identical=all(x['PASS'] for x in compare_features(z,z)),shuffled_rejected=not all(x['PASS'] for x in compare_features(z,z[::-1])),replacement_rejected=False)
    b=z.copy();b[3]=z[2];tests['replacement_rejected']=not all(x['PASS'] for x in compare_features(z,b))
    dump(ROOT/'audits/comparator_selftest.json',dict(status='PASS' if all(tests.values()) else 'FAIL',tests=tests,pooled_FID_is_permutation_invariant=True));assert all(tests.values())
    textfile(ROOT/'RUNBOOK.md',f'''# V6.30 execution\n\nNo model updates. Historical inputs are read-only.\n\nStart/resume (one process lock):\n```bash\ncd {REPO}\nscreen -dmS gvcrt_v630 {PYTHON} -B -u {ROOT}/run_all.py\n```\nRead-only check (no torch):\n```bash\n{PYTHON} -B {ROOT}/check_once.py\n```\nLogs: `{ROOT}/logs/`. Normal automatic publication uses a separate latest-main worktree. `DONE.json` records completed computation independently from publication status.\n''')
    dump(ROOT/'audits/preflight.json',dict(status='PASS',points=len(points),no_training=True,source_version=version))
    return points
if __name__=='__main__':main()
