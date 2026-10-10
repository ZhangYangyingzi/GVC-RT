"""Freeze source weights, cleaned 64-frame pools, and one shared replay plan."""
import concurrent.futures,random,re
from io21 import *
def verify_video(v):
    import cv2
    cv2.setNumThreads(1)
    p=ROOT/'audits/video_inventory'/f"{hashlib.sha256(v['path'].encode()).hexdigest()}.json"
    if p.exists():
        r=load(p);assert r['source_sha256']==v['sha256']
        if r.get('expected_dimensions')==[v['width'],v['height']]:return r
    reasons=[]
    if sha(v['path'])!=v['sha256']:reasons.append('source SHA256 mismatch')
    cap=cv2.VideoCapture(v['path']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n<64:reasons.append('fewer than 64 canonical frames')
    if n!=v['frames']:reasons.append('frame count differs from inherited manifest')
    for start in ([0,n-64] if n>=64 else []):
        cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        for i in range(64):
            ok,a=cap.read()
            if not ok or a.shape[:2]!=(v['height'],v['width']):reasons.append(f'cannot decode native64 at {start+i} at inherited dimensions');break
    cap.release()
    r=dict(path=v['path'],source_sha256=v['sha256'],actual_frames=n,expected_dimensions=[v['width'],v['height']],eligible=not reasons,reasons=reasons);dump(p,r);return r
def main():
    import torch
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for d in ('audits','logs','branches','evaluation','plans'): (ROOT/d).mkdir(parents=True,exist_ok=True)
    command([PYTHON,'-B',str(Path(__file__).resolve())])
    assert load(V20/'final_integrity.json')['status']=='PASS'
    source=load(V20/'branches/native_i_adapt/checkpoint_index.json')['1000']
    assert source['sha256']==EXPECTED and sha(source['path'])==EXPECTED
    for cp in (source,source['inference']):
        assert sha(cp['path'])==cp['sha256']
        state=torch.load(cp['path'],map_location='cpu',weights_only=True)
        assert {k:tensor_hash(state[k]) for k in ('wrapper','bridge','generator')}==source['module_hashes']
        if 'compression_hash' in state:assert state['compression_hash']==source['compression_hash']
        del state
    cfg=load(V20/'config.json')
    cfg.update(schema='v621',source_checkpoint={k:source[k] for k in ('path','sha256','module_hashes','compression_hash')},source_experiment='v620_native_i_adapt_1000',branch=None,branches=BRANCHES,methods=METHODS,expected_points=2400,seed=20261011,domain_seed=20261011,updates=1000,clip_length=64,training_plan_reused=None,training_plan='plans/shared.json',target_positions=list(range(49,64)),lambda_adv=.01,IPs=IPS,i_frame_modes={m:'native_original' if m=='original' else 'native_RGB_bypass' for m in EVAL_METHODS},discriminator=dict(channels=[3,64,128,256,512,1],kernel=4,padding=1,strides=[2,2,2,1,1],spectral_norm=True,activation='LeakyReLU(0.2)',lr=2e-4,betas=[0.,.9],weight_decay=0,seed=20261012))
    dump(ROOT/'config.json',cfg)
    ec=load(V20/'evaluation/config.json');ec.update(schema='v621',methods=METHODS,expected_points=2400,IPs=IPS,source_checkpoint=cfg['source_checkpoint'],i_frame_modes=cfg['i_frame_modes'],checkpoints={'native_initial':source['inference']},deployment_receiver_hashes={'native_initial':ec['deployment_receiver_hashes']['native_i_adapt_1000']})
    assert ec['compression_hash']==source['compression_hash'];dump(ROOT/'evaluation/config.json',ec)
    for d in DATASETS:dump(ROOT/'manifests'/f'{d}.json',load(V20/'manifests'/f'{d}.json'))
    dump(ROOT/'qp_semantics_audit.json',load(V20/'qp_semantics_audit.json'))
    exclusions=load(V18/'audits/exclusions.json');dump(ROOT/'audits/inherited_exclusions.json',exclusions)
    for name in ('historical_exposure.json','cleaned_original_pool.json','forbidden_content_windows.json'):dump(ROOT/'audits'/name,load(V18/'audits'/name))
    banned=set(exclusions['identities']);testhashes={v['source_sha256'] for d in DATASETS for v in sources(d)};testnames={v['name'] for d in DATASETS for v in sources(d)}
    pools={};excluded=[]
    for d in ('ulong','uvg'):
        manifest=load(V20/f'train_manifest_{d}.json');keep=[]
        for v in manifest['videos']:
            name=v['filename' if d=='ulong' else 'name'];identities=set(re.findall(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',str(v)));h=v['sha256' if d=='ulong' else 'source_sha256']
            reasons=[]
            if identities&banned or name in testnames or h in testhashes or (d=='ulong' and h in exclusions['file_hashes']):reasons.append('historical validation/test source')
            if v['frames']<64:reasons.append('fewer than 64 canonical frames')
            if reasons:excluded.append(dict(domain=d,video=name,source_sha256=h,reasons=reasons))
            else:keep.append(v)
        if d=='ulong':
            valid=[]
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                for i,(v,r) in enumerate(zip(keep,executor.map(verify_video,keep)),1):
                    if r['eligible']:valid.append(v)
                    else:excluded.append(dict(domain=d,video=v['filename'],**r))
                    if i%100==0:print('POOL CHECK',i,len(keep),flush=True)
            keep=valid
        assert keep
        manifest.update(schema='v621_cleaned_64',videos=keep,source_manifest=str(V20/f'train_manifest_{d}.json'),source_manifest_sha256=sha(V20/f'train_manifest_{d}.json'),minimum_canonical_frames=64)
        dump(ROOT/f'train_manifest_{d}.json',manifest);pools[d]=keep
    assert {v['name'] for v in pools['uvg']}=={'Beauty','ReadySteadyGo','ShakeNDry','YachtRide'}
    dump(ROOT/'audits/pool_filter.json',dict(status='PASS',excluded=excluded,counts={d:len(v) for d,v in pools.items()},inherited_exclusion_sha256=sha(ROOT/'audits/inherited_exclusions.json'),historical_cleaning=load(ROOT/'audits/cleaned_original_pool.json')))
    dump(ROOT/'dataset_split.json',dict(schema='v621',training={d:[v['filename' if d=='ulong' else 'name'] for v in vs] for d,vs in pools.items()},testing={d:[v['name'] for v in sources(d)] for d in DATASETS},source_split=load(V20/'dataset_split.json'),disjoint=True))
    from sampling import make_plan
    make_plan()
    protocol=dict(schema='v621',source_checkpoint=cfg['source_checkpoint'],frames=64,IPs=IPS,methods=METHODS,expected_points=2400,I_positions={str(ip):[0] if ip<0 else list(range(0,64,ip)) for ip in IPS},I_input='native_RGB_no_wrapper',QP_phase='distance_from_most_recent_I',reference_reset='clear_DPB_and_set_POC_0_both_sides_initialize_from_native_I_independent_decode',full_bytes=True,rate_denominator='64*1920*1080',fps='inherited per-video rate_accounting_fps',metrics='unchanged v620; all63 transitions including I refresh; dataset pooled FID',training=dict(shared_plan_sha256=sha(ROOT/'plans/shared.json'),target_positions=list(range(49,64)),denominator=15,updates=1000,A=dict(I=48,warmup=0),B=dict(I=0,warmup=48,no_grad=True),C=dict(I=48,warmup=0,PatchGAN=cfg['discriminator'],lambda_adv=.01),reference_detach=True,optimizer=cfg['optimizer'],loss=cfg['objective']),source_protocol_sha256=sha(V20/'protocol.json'),equal_rate=dict(grid_points=100,space='ln(bpp)',interpolation='PCHIP',extrapolation=False),checkpoint_selection=False)
    dump(ROOT/'protocol.json',protocol)
    deps=load(V20/'audits/dependencies.json')
    deps.update({str(p):sha(p) for p in V20.glob('*.py')})
    deps.update({str(V20/n):sha(V20/n) for n in ('config.json','protocol.json','evaluation/config.json','final_integrity.json','train_manifest_ulong.json','train_manifest_uvg.json')})
    deps[source['path']]=EXPECTED
    for p,h in deps.items():assert sha(p)==h
    dump(ROOT/'audits/dependencies.json',deps)
    dump(ROOT/'audits/checkpoint_integrity.json',dict(status='PASS',source=source,expected_sha256=EXPECTED,modules_verified=True))
    paths=[ROOT/'config.json',ROOT/'protocol.json',ROOT/'plans/shared.json',ROOT/'sampling_order.json',ROOT/'dataset_split.json',ROOT/'train_manifest_ulong.json',ROOT/'train_manifest_uvg.json',ROOT/'qp_semantics_audit.json',*(ROOT/'manifests').glob('*.json')]
    dump(ROOT/'audits/protocol_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    dump(ROOT/'audits/historical_inventory.json',historical_inventory())
    dump(ROOT/'preflight_audit.json',dict(status='PASS',source_sha256=EXPECTED,counts={d:len(v) for d,v in pools.items()},updates=1000,supervised_P_frames=15000))
    print('PREPARE PASS',flush=True)
if __name__=='__main__':main()
