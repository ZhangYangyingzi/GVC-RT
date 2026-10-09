"""Read-only historical inputs, local data audit, exact RGB expansion and plans."""
import concurrent.futures,random,re,shutil,traceback,zipfile
from io18 import *
UUID=re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',re.I)
def identities(value):return set(UUID.findall(json.dumps(value)))
def forbidden_content():
    import cv2
    dest=ROOT/'audits/forbidden_content_windows.json'
    if dest.exists():return set(load(dest)['consecutive16_frame_digest_sha256'])
    rows=[];digests=set()
    for item in load(ROOT/'audits/original_pool_historical_test_conflicts.json')['conflicts']:
        v=item['v617_training_record'];assert sha(v['path'])==v['sha256'];cap=cv2.VideoCapture(v['path']);frames=[]
        try:
            while True:
                ok,bgr=cap.read()
                if not ok:break
                frames.append(hashlib.sha256(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).tobytes()).hexdigest())
        finally:cap.release()
        for start in range(len(frames)-15):digests.add(hashlib.sha256(''.join(frames[start:start+16]).encode()).hexdigest())
        rows.append(dict(identity=item['identity'],source_path=v['path'],source_sha256=v['sha256'],decoded_frames=len(frames),full_frame_RGB_sha256=frames))
    dump(dest,dict(status='PASS',sources=rows,consecutive16_frame_digest_sha256=sorted(digests),comparison='native cv2 RGB24; full removed sources, every possible consecutive16 window; candidate beginning/end and every actual sampled training window checked'))
    return digests
def decode_probe(v):
    import cv2
    p=Path(v['path']); cache=ROOT/'audits/video_inventory'/(hashlib.sha256(str(p).encode()).hexdigest()+'.json')
    if cache.exists():
        r=load(cache);assert r['size']==p.stat().st_size and r['mtime_ns']==p.stat().st_mtime_ns;return r
    r=dict(v,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=sha(p),eligible=False)
    cap=cv2.VideoCapture(str(p))
    try:
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));fps=cap.get(cv2.CAP_PROP_FPS)
        assert n>=16 and w>=256 and h>=256 and fps>0,(n,w,h,fps)
        digest=hashlib.sha256();window_digests=[]
        for start in (0,n-16):
            framehash=[]
            cap.set(cv2.CAP_PROP_POS_FRAMES,start)
            for i in range(16):
                ok,bgr=cap.read();assert ok and bgr.shape[:2]==(h,w),(start,i)
                raw=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).tobytes();framehash.append(hashlib.sha256(raw).hexdigest())
                if start==0:digest.update(raw)
            window_digests.append(hashlib.sha256(''.join(framehash).encode()).hexdigest())
        r.update(frames=n,width=w,height=h,fps=fps,duration_seconds=n/fps,first16_RGB_sha256=digest.hexdigest(),native_window_content_digests=window_digests,eligible=True,decode_checks='first and last 16 native consecutive frames; sampled training windows separately fully verified')
    except Exception:r['error']=traceback.format_exc()
    finally:cap.release()
    dump(cache,r);return r
def exclusions():
    split=load(V17/'dataset_split.json');ids=set(x for k,vs in split['excluded'].items() if k.startswith('ulong') for x in vs);records=[];hashes=set()
    # All historical split files, validation manifests and fixed ULong tests.
    paths=set()
    for folder in (REPO/'experiments',REPO/'expericent_generation_input'):
        for p in folder.rglob('*.json'):
            if p.is_relative_to(ROOT):continue
            if any(x in p.parts for x in ('parts','results','cache','outputs','logs','analysis','audits','dependency_sources','bitstreams','features','visualizations','data_cache')):continue
            if 'split' in p.name or (any(s in p.name for s in ('validation','test','holdout')) and 'manifest' in p.name) or (p.name=='ulong.json' and 'manifests' in p.parts):paths.add(p)
    for p in sorted(paths):
        try:value=load(p)
        except (ValueError,OSError):continue
        def walk(x,held=False):
            if isinstance(x,dict):
                for k,v in x.items():
                    flag=held or any(t in k.lower() for t in ('test','validation','valid','holdout','held','excluded','val_','confirmation'))
                    if flag:ids.update(identities(v))
                    walk(v,flag)
                if held:
                    for k in ('sha256','source_sha256','file_sha256'):
                        if isinstance(x.get(k),str):hashes.add(x[k])
            elif isinstance(x,list):
                for v in x:walk(v,held)
        initial=any(s in p.name for s in ('validation','test','holdout')) or p.name=='ulong.json';walk(value,initial)
        records.append(dict(path=str(p),sha256=sha(p)))
    dump(ROOT/'audits/exclusions.json',dict(identities=sorted(ids),file_hashes=sorted(hashes),sources=records));return ids,hashes
def ulong_pool(original):
    from cleaning import clean_original
    original=clean_original();assert len(original)==1020
    ids,excluded_hashes=exclusions();origids=set().union(*(identities(v['filename']) for v in original))
    assert not origids & ids,('historical training/test overlap',sorted(origids & ids))
    candidates={}
    # Reuse already extracted local videos without changing their bytes.
    roots={Path(v['path']).parent for v in original}
    for parent in roots:
        for p in parent.glob('*.mp4'):
            uid=next(iter(identities(p.name)),None)
            if uid and uid not in ids and uid not in origids:candidates.setdefault(uid,dict(filename='clips_long_1920/'+uid+'.mp4',path=str(p),origin='existing_local_extraction'))
    inventory=[]
    for z in sorted(Path('/Huang_group/zyyz/datasets/UltraVideo-Long/clips_long_1920').glob('*.zip')):
        with zipfile.ZipFile(z) as archive:
            for member in archive.infolist():
                if not member.filename.lower().endswith('.mp4'):continue
                uid=next(iter(identities(member.filename)),None)
                inventory.append(dict(archive=str(z),member=member.filename,size=member.file_size,compressed_size=member.compress_size,CRC=member.CRC,identity=uid,excluded=uid in ids,original=uid in origids))
                if not uid or uid in ids or uid in origids or uid in candidates:continue
                target=ROOT/'training_videos'/('ulong_'+uid+'.mp4');target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    temp=target.with_suffix('.tmp')
                    with archive.open(member) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,1<<20)
                    assert temp.stat().st_size==member.file_size;temp.replace(target)
                assert target.stat().st_size==member.file_size
                candidates[uid]=dict(filename='clips_long_1920/'+uid+'.mp4',path=str(target),origin='local_archive',archive_path=str(z),archive_member=member.filename,archive_CRC=member.CRC)
        print('ARCHIVE INVENTORY',z.name,len(candidates),flush=True)
        dump(ROOT/'preparation_status.json',dict(phase='local_ULong_extraction',candidate_videos=len(candidates),updated_unix=time.time()))
    write(ROOT/'audits/ulong_archive_inventory.csv',inventory)
    banned=forbidden_content();verified=[];reject=[];seen_hash={};seen_rgb={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(decode_probe,[*original,*[candidates[k] for k in sorted(candidates)]])):
            if i<len(original):assert r['eligible'] and r['sha256']==original[i]['sha256'],('original pool failed audit',r)
            reason='decode/format' if not r['eligible'] else 'excluded content' if r['sha256'] in excluded_hashes or any(h in banned for h in r['native_window_content_digests']) else 'duplicate file hash' if r['sha256'] in seen_hash else 'duplicate decoded first16 content' if r['first16_RGB_sha256'] in seen_rgb else None
            if reason:
                assert i>=len(original),('duplicate original pool',r['filename'],reason)
                reject.append(dict(r,rejection_reason=reason));continue
            seen_hash[r['sha256']]=r['filename'];seen_rgb[r['first16_RGB_sha256']]=r['filename'];verified.append(r)
            if i%50==0:
                print('ULONG DECODE',i,len(verified),flush=True);dump(ROOT/'preparation_status.json',dict(phase='ULong_decode_audit',processed=i+1,eligible=len(verified),updated_unix=time.time()))
    assert len(verified)>=1020
    dump(ROOT/'train_manifest_ulong.json',dict(videos=verified[:1020],train_video_count=1020,original_manifest=str(V17/'train_manifest_ulong.json'),original_manifest_sha256=sha(V17/'train_manifest_ulong.json'),cleaning_audit=str(ROOT/'audits/cleaned_original_pool.json')))
    dump(ROOT/'train_manifest_ulong_expanded.json',dict(videos=verified,train_video_count=len(verified),retained_original=1020,added_count=len(verified)-1020,rejected=reject))
    write(ROOT/'audits/ulong_dataset_inventory.csv',[*verified,*reject])
def uvg_pool():
    import numpy as np
    from PIL import Image
    old=load(V17/'train_manifest_uvg.json');dump(ROOT/'train_manifest_uvg.json',old);expanded=[]
    audit=V17.parent/'gvcrt_neural_wrapper_v5a_2_fair_30fps_rgb/ffmpeg_conversion_audit.json';conversion=load(audit)
    for v in old['videos']:
        path=Path(v['source_path']);framebytes=1920*1080*3//2;n,rem=divmod(path.stat().st_size,framebytes)
        assert not rem and n>=253 and path.name.endswith('_1920x1080_120fps_420_8bit_YUV.yuv') and sha(path)==v['source_sha256']
        indices=list(range(0,n,4));directory=ROOT/'rgb_cache'/v['name'];directory.mkdir(parents=True,exist_ok=True)
        manifest=ROOT/'audits'/('expanded_'+v['name']+'.json')
        if manifest.exists():
            item=load(manifest);assert item['source_sha256']==v['source_sha256'];expanded.append(item);continue
        cmd=list(conversion['inherited_command']);cmd[cmd.index('-i')+1]=str(path);cmd[cmd.index('-frames:v')+1]=str(n);command(cmd)
        files=[];rgb=[];whole=hashlib.sha256()
        with (ROOT/'logs'/('convert_'+v['name']+'.log')).open('a') as err:
            proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=err)
            try:
                for source in range(n):
                    raw=proc.stdout.read(1920*1080*3);assert len(raw)==1920*1080*3,(v['name'],source)
                    if source%4:continue
                    i=source//4;hashval=hashlib.sha256(raw).hexdigest()
                    if i<64:assert hashval==v['frame_rgb_sha256'][i],('old/new UVG RGB mismatch',v['name'],i)
                    p=directory/f'frame_{i:06d}.png'
                    if not p.exists():
                        temp=p.with_suffix('.tmp');Image.fromarray(np.frombuffer(raw,np.uint8).reshape(1080,1920,3)).save(temp,format='PNG',compress_level=1);temp.replace(p)
                    with Image.open(p) as image:assert image.size==(1920,1080) and image.tobytes()==raw
                    files.append(sha(p));rgb.append(hashval);whole.update(raw)
            except Exception:proc.terminate();proc.wait();raise
            finally:proc.stdout.close()
            assert proc.wait()==0
        item=dict(v,input_dir=str(directory),frames=len(indices),source_total_frames=n,source_frame_indices=indices,frame_file_sha256=files,frame_rgb_sha256=rgb,rgb_sha256=whole.hexdigest(),duration_seconds=len(indices)/30,conversion=dict(pixel_format='yuv420p',bit_depth=8,color_matrix='bt709',source_range='limited',RGB_range='full',command=cmd,audit_source=str(audit),audit_sha256=sha(audit)),first64_match=True)
        dump(manifest,item);expanded.append(item);print('UVG EXPANSION PASS',v['name'],len(indices),flush=True)
    dump(ROOT/'train_manifest_uvg_expanded.json',dict(videos=expanded))
def make_plans():
    import cv2,numpy as np
    from PIL import Image
    rng=random.Random(20261010);order=[(d,q) for d in ('ulong','uvg') for q in range(10) for _ in range(50)];rng.shuffle(order)
    dump(ROOT/'sampling_order.json',dict(seed=20261010,order=order,domain_counts={'ulong':500,'uvg':500},per_domain_QP_counts={str(q):50 for q in range(10)}))
    qp=load(ROOT/'qp_semantics_audit.json')
    # Actual QPs use the original periodic INDEX_MAP schedule, not local reset positions.
    sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src'));from gvc_hooks import INDEX_MAP
    from src.models.video_model_gvcrt import qp_shift
    for poolname,suffix,seed in [('AB','',20261011),('C','_expanded',20261012)]:
        dest=ROOT/'plans'/f'{poolname}.json'
        if dest.exists():assert load(dest)['sampling_order_sha256']==sha(ROOT/'sampling_order.json');continue
        vr=random.Random(seed);records={d:load(ROOT/f'train_manifest_{d}{suffix}.json')['videos'] for d in ('ulong','uvg')};rows=[]
        for step,(d,q) in enumerate(order,1):
            v=vr.choice(records[d]);n=v['frames'];start=vr.randrange(n-15);x=vr.randrange(v['width']-256+1);y=vr.randrange(v['height']-256+1);arr=[];fullhash=[]
            if d=='ulong':
                cap=cv2.VideoCapture(v['path']);cap.set(cv2.CAP_PROP_POS_FRAMES,start)
                try:
                    for i in range(16):
                        ok,bgr=cap.read();assert ok,(v['path'],start,i);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);fullhash.append(hashlib.sha256(rgb.tobytes()).hexdigest());arr.append(rgb[y:y+256,x:x+256].copy())
                finally:cap.release()
                assert hashlib.sha256(''.join(fullhash).encode()).hexdigest() not in forbidden_content(),('sample matches removed source content',v['filename'],start)
                source=list(range(start,start+16));name=v['filename'];path=v['path']
            else:
                for i in range(start,start+16):
                    p=Path(v['input_dir'])/f'frame_{i:06d}.png';assert sha(p)==v['frame_file_sha256'][i]
                    with Image.open(p) as im:a=np.asarray(im,dtype=np.uint8)
                    h=hashlib.sha256(a.tobytes()).hexdigest();assert h==v['frame_rgb_sha256'][i];fullhash.append(h);arr.append(a[y:y+256,x:x+256].copy())
                source=v['source_frame_indices'][start:start+16];name=v['name'];path=v['source_path']
            assert all(a.shape==(256,256,3) for a in arr)
            # Native shift_qp is q + qp_shift[INDEX_MAP[pos%8]]; get audited offsets from the frozen model constants.
            actual=[q if pos==0 else q+qp_shift[INDEX_MAP[pos%len(INDEX_MAP)]] for pos in range(16)]
            rows.append(dict(adaptation_step=step,domain=d,video=name,source_path=path,source_sha256=v['sha256' if d=='ulong' else 'source_sha256'],canonical_indices=list(range(start,start+16)),source_frame_indices=source,crop_x=x,crop_y=y,crop_size=256,external_qp=q,actual_qps=actual,full_RGB_sha256=fullhash,cropped_RGB_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in arr],supervised_window_positions=list(range(1,16)),normalization_denominator=15))
            if step%50==0:print('PLAN',poolname,step,flush=True);dump(ROOT/'preparation_status.json',dict(phase='sampling_plans',plan=poolname,updates=step,updated_unix=time.time()))
        dump(dest,dict(updates=rows,sampling_order_sha256=sha(ROOT/'sampling_order.json'),video_selection='uniform independent video within domain',seed=seed))
    for name in ('AB','C'):
        rows=load(ROOT/'plans'/f'{name}.json')['updates'];assert len(rows)==1000
        for d in ('ulong','uvg'):
            assert sum(r['domain']==d for r in rows)==500
            assert all(sum(r['domain']==d and r['external_qp']==q for r in rows)==50 for q in range(10))
    dump(ROOT/'audits/plan_integrity.json',dict(status='PASS',AB_sha256=sha(ROOT/'plans/AB.json'),C_sha256=sha(ROOT/'plans/C.json'),same_domain_QP_order=True,AB_exact_shared=True,updates=1000,P_targets_per_update=15,P_targets_per_branch=15000))
def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from cleaning import archive_previous,exposure
    archive_previous()
    torch.set_num_threads(2)
    for d in ('logs','audits','manifests','plans','evaluation','features','branches'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    assert load(V17/'final_integrity.json')['status']=='PASS'
    index=load(V17/'checkpoint_index.json')['1000'];assert sha(index['path'])==EXPECTED
    state=torch.load(index['path'],map_location='cpu',weights_only=True)
    assert {k:tensor_hash(state[k]) for k in ('wrapper','bridge','generator')}==index['module_hashes'];assert state['compression_hash']==index['compression_hash'];del state
    if not (ROOT/'config.json').exists():
        cfg=load(V17/'config.json');cfg.update(schema='v618',seed=20261010,clip_length=16,branches=['A','B','C'],P_targets_per_update=15,updates=1000,source_step=1000,source_experiment='v6.17',cleaned_original_ULong_count=1020,exclude_historical_test_identities=True,source_checkpoint={k:index[k] for k in ('path','sha256','module_hashes','compression_hash')});dump(ROOT/'config.json',cfg)
        for d in DATASETS:dump(ROOT/'manifests'/f'{d}.json',load(V17/'manifests'/f'{d}.json'))
        dump(ROOT/'qp_semantics_audit.json',load(V17/'qp_semantics_audit.json'));dump(ROOT/'dataset_split.json',load(V17/'dataset_split.json'))
        ec=load(V17/'evaluation/config.json');ec.update(schema='v618',methods=METHODS,expected_points=800);ec['checkpoints']={'v617_initial':ec['checkpoints']['dists_w05_1000']};ec['deployment_receiver_hashes']={'v617_initial':ec['deployment_receiver_hashes']['dists_w05_1000']};dump(ROOT/'evaluation/config.json',ec)
        protocol=load(V17/'protocol.json');protocol.update(methods=METHODS,new_branches=['A_1000','B_1000','C_1000'],source_model='v6.17 final step1000',comparison='five-model common actual bpp interval; 100 ln(bpp) PCHIP; no extrapolation');dump(ROOT/'protocol.json',protocol)
        deps=load(V17/'audits/dependencies.json')
        for p in [*V17.glob('*.py'),V17/'config.json',V17/'evaluation/config.json',V17/'protocol.json',V17/'checkpoint_index.json',V17/'final_integrity.json',*V17.glob('train_manifest*.json'),*(V17/'manifests').glob('*.json')]:deps[str(p)]=sha(p)
        deps[index['path']]=EXPECTED;dump(ROOT/'audits/dependencies.json',deps);dump(ROOT/'audits/historical_inventory.json',historical_inventory())
    exposure()
    if not (ROOT/'train_manifest_ulong_expanded.json').exists():ulong_pool(load(V17/'train_manifest_ulong.json')['videos'])
    from sampling import make_plans as parallel_plans
    uvg_pool();parallel_plans()
    pools={}
    for d in ('ulong','uvg'):
        pools[d]={}
        for label,suffix in [('AB',''),('C','_expanded')]:
            vs=load(ROOT/f'train_manifest_{d}{suffix}.json')['videos'];pools[d][label]=dict(independent_videos=len(vs),total_available_frames=sum(v['frames'] for v in vs),duration_seconds=sum(v.get('duration_seconds',v['frames']/v.get('fps',30)) for v in vs))
    coverage={}
    for b,p in [('A','AB'),('B','AB'),('C','C')]:
        rows=load(ROOT/'plans'/f'{p}.json')['updates'];coverage[b]={d:dict(sampled_unique_videos=len({r['video'] for r in rows if r['domain']==d}),sampled_unique_video_frames=len({(r['video'],i) for r in rows if r['domain']==d for i in r['source_frame_indices']}),updates=500) for d in ('ulong','uvg')}
    dump(ROOT/'audits/data_coverage.json',dict(status='PASS',pools=pools,sampled_coverage=coverage,actual_ULong_added=pools['ulong']['C']['independent_videos']-1020))
    base=load(V17/'dataset_split.json');clean=load(ROOT/'audits/cleaned_original_pool.json');excl=load(ROOT/'audits/exclusions.json')
    dump(ROOT/'dataset_split.json',dict(status='PASS',schema='v618_cleaned_split',source_manifest=str(V17/'dataset_split.json'),source_manifest_sha256=sha(V17/'dataset_split.json'),ulong_original_count=1024,ulong_train_count=1020,ulong_expanded_train_count=pools['ulong']['C']['independent_videos'],uvg_train=base['uvg_train'],excluded=base['excluded'],additional_historical_excluded_identities=excl['identities'],removed_four=clean['excluded_identities'],training_source_disjoint=True,cleaning_audit_sha256=sha(ROOT/'audits/cleaned_original_pool.json'),training_sources_verified=[dict(path=v['path'],sha256=v['sha256']) for v in load(ROOT/'train_manifest_ulong.json')['videos']],historical_exposure_audit_sha256=sha(ROOT/'audits/historical_exposure.json')))
    for p in load(ROOT/'audits/dependencies.json'):
        source=Path(p)
        if source.suffix=='.py':
            target=ROOT/'dependency_sources'/(hashlib.sha256(p.encode()).hexdigest()[:12]+'_'+source.name);target.parent.mkdir(exist_ok=True)
            if not target.exists():shutil.copyfile(source,target)
            assert sha(source)==sha(target)
    ps=[ROOT/'config.json',ROOT/'protocol.json',ROOT/'dataset_split.json',ROOT/'qp_semantics_audit.json',ROOT/'sampling_order.json',ROOT/'audits/cleaned_original_pool.json',ROOT/'audits/historical_exposure.json',ROOT/'audits/exclusions.json',ROOT/'audits/forbidden_content_windows.json',*ROOT.glob('train_manifest*.json'),*(ROOT/'plans').glob('*.json'),*(ROOT/'manifests').glob('*.json')];dump(ROOT/'audits/protocol_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in ps})
    dump(ROOT/'preflight_audit.json',dict(status='PASS',source_checkpoint=index,plan_integrity=load(ROOT/'audits/plan_integrity.json'),data_coverage=load(ROOT/'audits/data_coverage.json')));print('PREPARE PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'blocking_error.json',dict(status='FAIL',phase='prepare',traceback=traceback.format_exc()));raise
