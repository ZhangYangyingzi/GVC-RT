"""Only GT/source audits select data. No candidate quality is read here."""
from common_s2 import *
import collections

def rng(name):return random.Random(int(digest([PLAN_SEED,name]),16))

def main():
    env()
    if (ROOT/'plans/index.json').exists():return
    assert sha(INIT)==INIT_SHA
    src=REFERENCE_EXPERIMENT
    isolation=load(src/'audits/source_isolation.json');assert isolation['status']=='PASS'
    original_pool=HIST/'manifests/B_expanded_ulong.json'
    assert sha(original_pool)=='0526eace3882402792105fae5cff7817b624f0e5b0ff49ddd6cce03829d1de63'
    manifest=load(src/'source_manifest.json')
    for path in ['manifests/effective_ULong_pool.json']:
        assert sha(src/path)==manifest['immutable_manifests'][path]
    pool=load(src/'manifests/effective_ULong_pool.json')
    dev=[r for r in load(src/'vendor/sealed/manifests/evaluation_rows.json') if r['split']=='development']
    test=load(src/'manifests/final_videos.json')
    excluded={r['source_group'] for r in dev}|{r.get('source_group') for r in test}
    banned='dce599d3-d196-4d87-b2b1-698acdbbb3d4'
    assert len(dev)==32 and all(r['source_group'] not in excluded and banned not in r['video'] for r in pool)
    assert not any(banned in r['video'] for r in dev)
    dump(ROOT/'manifests/effective_ULong_pool.json',pool)
    dump(ROOT/'audits/source_isolation.json',dict(status='PASS',inherited_audit=str(src/'audits/source_isolation.json'),inherited_SHA256=sha(src/'audits/source_isolation.json'),
        existing_quarantined_groups=236,training_identities=len(pool),training_source_groups=len({r['source_group'] for r in pool}),MCL_checks_inherited=True,final_test_excluded=True,forbidden_identity=banned,
        scope='Inherited source graph and MCL checks; no new exhaustive dataset scan',UVG_samples=0,Vimeo_samples=0))
    groups=collections.defaultdict(list)
    for r in pool:groups[r['source_group']].append(r)
    order=sorted(groups);rng('source-group-order').shuffle(order);order=order[:1200]
    assert len(order)==1200 and len(set(order))==1200
    for g in groups:rng('within-group:'+g).shuffle(groups[g])
    qpmap={}
    for sh in load(src/'plans/index.json')['shards']:
        for row in load(src/sh['file']):qpmap.setdefault(row['external_qp'],row['actual_frame_qps'])
        if len(qpmap)==10:break
    qps=[];qr=rng('external-QP')
    for _ in range(120):b=list(range(10));qr.shuffle(b);qps+=b
    cr=rng('crop');wr=rng('window');plan=[]
    for slot,g in enumerate(order):
        r=groups[g][0];n=r['frames']-63;nb=min(4,n);bins=list(range(nb));rng('bins:'+r['identity']).shuffle(bins);bi=bins[0]
        lo=bi*n//nb;hi=(bi+1)*n//nb;start=wr.randrange(lo,hi);x=cr.randrange(r['width']-255);y=cr.randrange(r['height']-255)
        row=dict(domain='ulong',logical_sample=True,plan_seed=PLAN_SEED,plan_slot=slot,global_update=slot//4+1,logical_rank=slot%4,
            video=r['video'],identity=r['identity'],source_group=g,source_path=r['source_path'],source_sha256=r['source_sha256'],source_size_bytes=r['source_size_bytes'],
            source_total_frames=r['frames'],source_fps=r['source_fps'],width=r['width'],height=r['height'],source_frame_indices=list(range(start,start+64)),canonical_indices=list(range(start,start+64)),
            crop_x=x,crop_y=y,crop_width=256,crop_height=256,external_qp=qps[slot],actual_frame_qps=qpmap[qps[slot]],temporal_bin=bi,temporal_bin_interval=[lo,hi])
        row['sample_key']=digest(row);plan.append(row)
    dump(ROOT/'plans/train.json.gz',plan)
    dump(ROOT/'plans/index.json',dict(status='FROZEN',rows=1200,global_updates_cap=300,effective_batch=4,seed=PLAN_SEED,shards=[dict(file='plans/train.json.gz',sha256=sha(ROOT/'plans/train.json.gz'),rows=1200)],QP_counts=dict(collections.Counter(qps))))
    # Deterministic GT-only characterization: 1/8 native sampling is for screening,
    # never a model input. Fixed 0,16,32,48,63 native frames are read and hashed.
    import cv2,numpy as np
    from inherited_reader import source_file
    cv2.setNumThreads(1);profiles=[]
    for r in sorted(dev,key=lambda r:r['video']):
        with source_file(r) as media:
            cap=cv2.VideoCapture(str(media));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));images=[];hashes=[]
            try:
                for index in [r['source_frame_indices'][i] for i in [0,16,32,48,63]]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read();assert ok
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert rgb.shape==(r['height'],r['width'],3)
                    hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest());images.append(cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),(240,135),interpolation=cv2.INTER_AREA))
            finally:cap.release()
        flows=[cv2.calcOpticalFlowFarneback(a,b,None,.5,3,15,3,5,1.2,0) for a,b in zip(images,images[1:])]
        mag=float(np.mean([np.linalg.norm(f,axis=2).mean() for f in flows]));camera=float(np.mean([np.linalg.norm(np.median(f,axis=(0,1))) for f in flows]))
        residual=float(np.mean([np.linalg.norm(f-np.median(f,axis=(0,1)),axis=2).mean() for f in flows]))
        texture=float(np.mean([np.abs(cv2.Laplacian(a,cv2.CV_32F)).mean() for a in images]))
        profiles.append(dict(sequence=Path(r['video']).stem,source_group=r['source_group'],source_frames=count,motion=mag,camera_translation=camera,nontranslation_motion=residual,texture=texture,anchor_RGB_SHA256=hashes,row=r))
    remaining=list(profiles);chosen=[]
    rankings=[('low_motion_high_texture',lambda p:(p['motion']/max(p['texture'],1e-6),p['sequence'])),
              ('camera_motion',lambda p:(-p['camera_translation'],p['sequence'])),('dynamic_occlusion_candidate',lambda p:(-p['nontranslation_motion'],p['sequence']))]
    for category,key in rankings:
        selected=sorted(remaining,key=key)[:4]
        for p in selected:p['GT_category_rule']=category;chosen.append(p);remaining.remove(p)
    assert len(chosen)==12 and len({p['source_group'] for p in chosen})==12
    dump(ROOT/'manifests/quick_development.json',chosen)
    long=[]
    for category,_ in rankings:
        eligible=sorted([p for p in chosen if p['GT_category_rule']==category and p['source_frames']>=128],key=lambda p:p['sequence'])
        assert eligible,('NO_128_FRAME_DEVELOPMENT_SOURCE',category)
        long.append(eligible[0])
    long+=sorted([p for p in chosen if p['source_frames']>=128 and p not in long],key=lambda p:p['sequence'])[:1]
    assert len(long)==4
    dump(ROOT/'manifests/long_videos.json',[dict(p,source_frame_indices=list(range(128)),external_QP=5) for p in long])
    dump(ROOT/'audits/GT_only_development_selection.json',dict(status='PASS',candidate_profiles=profiles,selection_rules=[n for n,k in rankings],
        selection_determined_without_model_quality=True,GT_native_anchor_positions=[0,16,32,48,63],characterization_size=[240,135],algorithm='OpenCV Farneback fixed .5/3/15/3/5/1.2/0; normalized motion/texture then camera-median flow then residual flow',selected=12,long_videos=4))
    dep=load(src/'protocol.json')
    protocol=dict(task='tcg_flicker_pilot_abcd_20261109',revision='flicker_short_BPTT_4_v1',base_commit='142f73049f336b5b963d5e0e7030818149bc4c99',
        branch='codex/tcg-flicker-pilot-abcd-20261109',parent_checkpoint=dict(path=str(INIT),sha256=INIT_SHA),dependency_protocol=dict(path=str(src/'protocol.json'),sha256=sha(src/'protocol.json')),
        original_codec_and_receiver=dep['original_codec_and_receiver'],groups={g:dict(physical_GPU=4+i,train_Bridge_mlp3=g in 'BD',align_projected_history=g in 'CD') for i,g in enumerate(GROUPS)},
        training=dict(seed=PLAN_SEED,max_additional_updates=300,effective_batch=4,ULong=4,UVG=0,Vimeo=0,s_train=.5,s_eval=.5,frames=64,crop=256,warmup=[1,48],supervision=[49,63],segments=[[49,52],[53,56],[57,60],[61,63]],
            cache='pre-residual current; preserve graph inside segment, detach at boundary, reset after video',optimizer='new AdamW; no inherited moments',learning_rates=dict(TCG=1e-5,LoRA=5e-5,Bridge_mlp3=1e-6),betas=[.9,.999],eps=1e-8,weight_decay=1e-4,clip_norm=1.,loss_normalization='15 supervised frames then 4 windows exactly once',disposable_smoke_updates=20),
        loss=dict(LPIPS=1.,DISTS=.5,RGB_L1=.1383589248954108,RAFT_motion=.05,temp='lambda*(RGB_MCResidual+.5*Sobel_MCResidual)',lambda_calibration=dict(training_windows=[r['sample_key'] for r in plan[:8]],rule='.25*median(||grad spatial||/||grad temporal||)',trainables='TCG+LoRA',unupdated_A2000=True),rate_proxy='monitoring only; not trainable objectives',VJP='segment sums; RGB legacy4096/unscale-once equivalence checked separately'),
        alignment=dict(flow='local torchvision RAFT C_T_SKHT_V2',receiver='independent original GVC-RT I/P only',source='base reconstruction current->previous, detached; no GT or future',position='after previous_adapter before concat',sampler='pixelcenter align_corners=False zero padding; per-scale displacement rescaling',FB_valid='.01*(backward^2+warped_forward^2)+.5'),
        validation=dict(points=36,videos=12,QP=[0,5,9],steps=[0,100,200,300],source='ULong development subset',GT_selection_only=True,main_temporal='MCResidual_Sobel_RAFT_pixelcenter_v1_PtoP',screen=dict(median_video_QP_ratio_max=.95,videos_ratio_lt1_min=8,LPIPS_DISTS_QP_mean_max_ratio=1.01,consecutive_validations=2,near_zero_denominator=1e-12),
            stop_at_200='At BOTH100 and200 mean PtoP Sobel temporal error not below A2000 and any QP mean LPIPS or DISTS exceeds 1.01*A2000; otherwise stop300',final_test_evaluated=False),
        long_videos=dict(count=4,frames=128,QP=5,lossless=True),precision=dep['precision'],source_plan_sha256=sha(ROOT/'plans/index.json'),
        cache=dict(media_cap_GiB=80,derived_cap_GiB=200,min_free_GiB=100,CPU_workers=4,codec_features_shared_readonly=True),
        publication=dict(branch='codex/tcg-flicker-pilot-abcd-20261109',no_weights_media_bitstreams=True),time=time.time())
    dump(ROOT/'protocol.json',protocol)
    dump(ROOT/'status.json',dict(status='RUNNING',current_stage='implementation_and_engineering_checks',complete=False,time=time.time(),committed_additional_updates={g:0 for g in GROUPS}))
    dump(ROOT/'final_integrity.json',dict(status='NOT_RUN',complete=False))
    dump(ROOT/'publication_status.json',dict(status='NOT_PUBLISHED',remote_verified=False))
    print('DATA_PLAN_AND_GT_ONLY_SELECTION_FROZEN',len(plan))

if __name__=='__main__':main()
