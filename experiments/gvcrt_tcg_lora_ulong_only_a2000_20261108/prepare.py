"""Independent seeded ULong group/bin plan, native-MCL source checks, no model quality."""
from common_s2 import *
import collections,concurrent.futures
POOL_SHA='0526eace3882402792105fae5cff7817b624f0e5b0ff49ddd6cce03829d1de63'

def rng(namespace):return random.Random(int(digest([PLAN_SEED,namespace]),16))

def mcl_source(row):
    import numpy as np,cv2
    from PIL import Image
    cv2.setNumThreads(1);path=Path(row['source_path']);assert sha(path)==row['source_sha256']
    assert row['pixel_format']=='yuv420p' and row['color_matrix']=='BT709' and row['source_range_convention']=='limited'
    assert row['source_fps_metadata'] in [24,25,30] and row['source_frame_indices']==list(range(64))
    frame_hashes=[];dh=[];rgb=hashlib.sha256()
    for i in range(64):
        p=Path(row['input_dir'])/f'frame_{i:06d}.png';assert sha(p)==row['frame_file_sha256'][i]
        with Image.open(p) as im:assert im.mode=='RGB';a=np.asarray(im).copy()
        assert a.shape==(1080,1920,3);h=hashlib.sha256(a.tobytes()).hexdigest();assert h==row['frame_rgb_sha256'][i]
        frame_hashes.append(h);rgb.update(a.tobytes())
        if i in [0,31,63]:
            tiny=cv2.resize(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),(9,8),interpolation=cv2.INTER_AREA)
            bits=(tiny[:,1:]>tiny[:,:-1]).reshape(-1);value=0
            for b in bits:value=(value<<1)|int(b)
            dh.append(f'{value:016x}')
    assert rgb.hexdigest()==row['rgb_sha256']
    video=dict(domain='mcljcv',sequence=row['name'],split='historical_MCL_reference',source_path=row['source_path'],source_sha256=row['source_sha256'],
        input_dir=row['input_dir'],source_frame_indices=list(range(64)),width=1920,height=1080,fps=row['source_fps_metadata'],
        frame_file_sha256=row['frame_file_sha256'],frame_rgb_sha256=frame_hashes,rgb_sha256=rgb.hexdigest(),
        pixel_format='yuv420p',color_matrix='BT709',source_range='limited',RGB_range='full',conversion_command=row['conversion_command'],
        old_metric_reuse_allowed=False,I_frame='real_I_codec_bitstream',historically_used_test_data=True)
    profile=dict(identity=row['name'].split('_')[0].lower(),source_SHA256=row['source_sha256'],RGB_anchors=[frame_hashes[i] for i in [0,31,63]],dHash=dh)
    return video,profile

def main():
    env();assert not (ROOT/'plans/index.json').exists(),'PLAN_EXISTS_DO_NOT_REBUILD'
    poolpath=HIST/'manifests/B_expanded_ulong.json';assert sha(poolpath)==POOL_SHA
    pool=load(poolpath);assert len(pool)==8470 and len({r['source_group'] for r in pool})==7969
    review=load(REVIEW/'audit.json');assert review['MCL']['count']==30
    old=load(PROJECT/'experiments/gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/eval_manifest.json')['mcljcv'];assert len(old)==30
    expected={r['sequence']:r['sha256'] for r in review['MCL']['rows']}
    assert all(expected[r['name']]==r['source_sha256'] for r in old)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:result=list(executor.map(mcl_source,old))
    mcl=[x[0] for x in result];profiles=[x[1] for x in result]
    dump(ROOT/'manifests/MCL_JCV_30.json',mcl);dump(ROOT/'audits/MCL_native_sources.json',dict(status='PASS',videos=mcl,profiles=profiles,
        source_YUV_and_64_RGB_cache_frames_verified=True,quality_metrics_read=False,native_I_bypass_forbidden=True,CPU_workers=4))
    nodes=load(HIST/'graph/nodes.json');components=load(HIST/'graph/final_components.json');mapping=load(HIST/'graph/final_identity_to_group.json')
    dev=load(HIST/'manifests/development.json');test=load(HIST/'manifests/new_holdout_test.json');q=load(HIST/'graph/quarantine.json')
    quarantined={r['group_id'] for r in q['groups']};assert len(quarantined)==236
    reserved={r['group_id'] for r in components if r['reserved_members'] or r['quarantined']}
    protected_groups=quarantined|reserved|{r['source_group'] for r in dev+test}
    inventory=load(HIST/'inventory/qualified.json');mcl_ids={r['identity'] for r in profiles}
    node_ids={r['identity']:r for r in nodes};assert len(mcl_ids)==30 and mcl_ids<=set(node_ids)
    edges=[];new_quarantine=set()
    for profile in profiles:
        assert node_ids[profile['identity']]['reserved'] and profile['source_SHA256'] in node_ids[profile['identity']]['hashes']
        if profile['identity'] in mapping:protected_groups.add(mapping[profile['identity']])
        for item in inventory:
            ds=item.get('fingerprint_first64_dhash',[]);rs=item.get('fingerprint_first64_RGB',[])
            exact=item.get('sha256')==profile['source_SHA256']
            rgb=len(rs)==3 and rs==profile['RGB_anchors']
            distances=[(int(a,16)^int(b,16)).bit_count() for a,b in zip(ds,profile['dHash'])]
            close=len(distances)==3 and max(distances)<=4
            if exact or rgb or close:
                group=mapping[item['identity']];new_quarantine.add(group)
                edges.append(dict(MCL=profile['identity'],ULong=item['identity'],source_group=group,file_SHA_equal=exact,
                    three_RGB_anchors_equal=rgb,three_dHash_distances=distances,conservative_group_isolation=True,full_video_duplicate_proven=False))
    protected_groups|=new_quarantine
    excluded=[dict(identity=r['identity'],source_group=r['source_group'],reason='protected/reserved/dev/test/quarantine/MCL_link') for r in pool if r['source_group'] in protected_groups]
    effective=[dict(r) for r in pool if r['source_group'] not in protected_groups]
    assert effective and not ({r['source_group'] for r in effective}&protected_groups)
    errata=load(ROOT/'vendor/sealed/repairs/native_geometry_v1/geometry_errata.json')['entries']
    native={r['sha256']:r for r in inventory if r.get('status')=='ELIGIBLE'}
    for row in effective:
        assert row['domain']=='ulong' and row['source_sha256'] in native
        if row['source_sha256'] in errata:row['height'],row['width'],_=errata[row['source_sha256']]['audited_native_shape']
        else:row['height']=native[row['source_sha256']]['height'];row['width']=native[row['source_sha256']]['width']
        row['source_fps']=native[row['source_sha256']]['probe']['avg_frame_rate']
        assert row['frames']>=64 and min(row['width'],row['height'])>=256
    dump(ROOT/'manifests/effective_ULong_pool.json',effective)
    dump(ROOT/'audits/source_isolation.json',dict(status='PASS',original_pool_SHA256=POOL_SHA,original_videos=8470,original_groups=7969,
        effective_videos=len(effective),effective_groups=len({r['source_group'] for r in effective}),existing_quarantined_groups=236,
        MCL_sources_checked=30,MCL_candidate_edges=edges,newly_quarantined_groups=sorted(new_quarantine),excluded_training_identities=excluded,
        UVG_samples=0,Vimeo_samples=0,source_group_overlap=[],historical_UVG_exposure_not_erased=True,
        screening_scope='Existing full source graph/reserved/holdouts/quarantine plus MCL native0/31/63 RGB and three dHash<=4 edges; no exhaustive-offset/crop/reencode detection claim'))
    # Only inherit the audited internal QP mapping, never old mixed-plan content or cursor.
    qpmap={};oldindex=load(SOURCE/'plans/index.json')
    for shard in oldindex['shards']:
        rows=load(SOURCE/shard['file'])
        for row in rows:qpmap.setdefault(row['external_qp'],row['actual_frame_qps'])
        if len(qpmap)==10:break
    assert set(qpmap)==set(range(10))
    groups=collections.defaultdict(list)
    for row in effective:groups[row['source_group']].append(row)
    for g in groups:rng('clip-order:'+g).shuffle(groups[g])
    group_order=[];passes=0;group_rng=rng('group-order')
    while len(group_order)<12000:
        part=sorted(groups);group_rng.shuffle(part)
        for i in range(len(part)):
            used=set(group_order[-(len(group_order)%4):]) if len(group_order)%4 else set()
            if part[i] in used:
                j=next(j for j in range(i+1,len(part)) if part[j] not in used);part[i],part[j]=part[j],part[i]
            group_order.append(part[i])
            if len(group_order)==12000:break
        passes+=1
    group_visits=collections.Counter();video_visits=collections.Counter();starts=collections.defaultdict(set);bin_orders={};plan=[]
    qp_rng=rng('external-QP-independent');qps=[]
    for i in range(1200):block=list(range(10));qp_rng.shuffle(block);qps+=block
    crop_rng=rng('spatial-crop-independent');window_rng=rng('temporal-start-independent')
    for slot,group in enumerate(group_order):
        row=groups[group][group_visits[group]%len(groups[group])];group_visits[group]+=1
        identity=row['identity'];visit=video_visits[identity];video_visits[identity]+=1;n=row['frames']-63;nb=min(4,n)
        if identity not in bin_orders:order=list(range(nb));rng('bin-order:'+identity).shuffle(order);bin_orders[identity]=order
        selected=bin_orders[identity][visit%nb];lo=selected*n//nb;hi=(selected+1)*n//nb
        choices=[x for x in range(lo,hi) if x not in starts[identity]]
        if not choices:choices=[x for x in range(n) if x not in starts[identity]]
        assert choices,'NO_UNUSED_VALID_WINDOW_FOR_PLANNED_VIDEO';start=window_rng.choice(choices);starts[identity].add(start)
        x=crop_rng.randrange(row['width']-255);y=crop_rng.randrange(row['height']-255);qp=qps[slot]
        record=dict(domain='ulong',logical_sample=True,plan_seed=PLAN_SEED,plan_slot=slot,global_update=slot//4+1,logical_rank=slot%4,
            video=row['video'],identity=identity,source_group=group,source_path=row['source_path'],source_sha256=row['source_sha256'],source_size_bytes=row['source_size_bytes'],
            source_total_frames=row['frames'],source_fps=row['source_fps'],width=row['width'],height=row['height'],
            source_frame_indices=list(range(start,start+64)),canonical_indices=list(range(start,start+64)),crop_x=x,crop_y=y,crop_width=256,crop_height=256,
            external_qp=qp,actual_frame_qps=qpmap[qp],video_visit=visit,temporal_bin=selected,temporal_bin_interval=[lo,hi],temporal_bin_order=bin_orders[identity],
            source_group_visit=group_visits[group],window_RGB_hash_status='to_be_verified_on_native_read_in_separate_audit')
        record['sample_key']=digest(record);plan.append(record)
    assert all(len({r['source_group'] for r in plan[i:i+4]})==4 for i in range(0,12000,4))
    shards=[]
    for i in range(0,12000,400):
        filename=f'plans/samples_{i:05d}_{i+399:05d}.json.gz';dump(ROOT/filename,plan[i:i+400]);shards.append(dict(file=filename,sha256=sha(ROOT/filename),rows=400))
    dump(ROOT/'plans/index.json',dict(status='FROZEN',rows=12000,global_updates_cap=3000,effective_batch=4,seed=PLAN_SEED,shards=shards,
        source_pool_SHA256=POOL_SHA,effective_pool_SHA256=sha(ROOT/'manifests/effective_ULong_pool.json'),old_mixed_cursor_used=False,
        group_shuffle_seed_namespace='group-order',QP_seed_namespace='external-QP-independent',QP_counts=dict(collections.Counter(qps)),
        independent_video_windows=len({(r['identity'],tuple(r['source_frame_indices'])) for r in plan}),exact_window_repeats=0,coverage_passes_not_scene_independence=True))
    dump(ROOT/'audits/logical_plan.json',dict(status='PASS',seed=PLAN_SEED,plan_index_SHA256=sha(ROOT/'plans/index.json'),samples=12000,
        four_ULong_per_batch=True,four_distinct_groups_per_batch=True,UVG_samples=0,Vimeo_samples=0,old_RGB_hashes_not_reused=True,
        all_new_windows_will_have_separate_native_read_receipts=True,training_rng_consumed=False))
    print('ULONG_ONLY_GROUP_BIN_PLAN_FROZEN',len(effective),len(groups),sha(ROOT/'plans/index.json'))

if __name__=='__main__':main()
