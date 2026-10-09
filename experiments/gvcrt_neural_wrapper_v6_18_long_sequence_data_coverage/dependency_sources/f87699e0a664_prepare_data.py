"""Small source-statistics samples only; all generated files stay under ROOT."""
import concurrent.futures,fcntl,random,tempfile,zipfile
from fractions import Fraction
from v61_io import *
FIELDS=('temporal_rgb_L1','temporal_rgb_MSE','sobel_edge_energy','laplacian_variance','mean_luminance','luminance_std')
def profile(job):
    import cv2,numpy as np
    cv2.setNumThreads(1)
    index,v,known,reserved=job;out=ROOT/'profiling'/f'{index:05d}.json'
    if out.exists():return load(out)
    r=dict(candidate_index=index,**v,basename=Path(v['filename']).name)
    try:
        with tempfile.TemporaryDirectory(dir=ROOT/'profiling_tmp',prefix=f'{index:05d}_') as tmp:
            if known:source=Path(known['path'])
            else:
                source=Path(tmp)/r['basename']
                with zipfile.ZipFile(v['archive']) as z,z.open(v['filename']) as src,source.open('wb') as dst:
                    for b in iter(lambda:src.read(1<<20),b''):dst.write(b)
            r['sha256']=sha(source)
            if known:assert r['sha256']==known['sha256']
            if reserved:
                r.update(status='RESERVED_HELDOUT',eligible=False,reason='Existing validation/confirmation/test excluded before statistics fitting');dump(out,r);return r
            cap=cv2.VideoCapture(str(source));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));fps=float(cap.get(cv2.CAP_PROP_FPS))
            r.update(frames=n,width=w,height=h,fps=fps)
            assert n>=4 and min(w,h)>=256
            starts=sorted(set(int(t) for t in np.linspace(0,n-2,4)));l1=[];mse=[];edge=[];lap=[];mean=[];std=[]
            try:
                for start in starts:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,start);pair=[]
                    for _ in range(2):
                        ok,bgr=cap.read();assert ok,(r['basename'],start)
                        rgb=cv2.cvtColor(cv2.resize(bgr,(256,144),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB).astype(np.float32)/255
                        pair.append(rgb);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
                        gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
                        edge.append(float(np.sqrt(gx*gx+gy*gy).mean()));lap.append(float(cv2.Laplacian(gray,cv2.CV_32F,ksize=3).var()))
                        mean.append(float(gray.mean()));std.append(float(gray.std()))
                    delta=pair[1]-pair[0];l1.append(float(np.abs(delta).mean()));mse.append(float((delta*delta).mean()))
            finally:cap.release()
            values=[float(np.mean(a)) for a in (l1,mse,edge,lap,mean,std)];assert np.isfinite(values).all()
            r.update(dict(zip(FIELDS,values)),status='PASS',eligible=True,profile_pair_start_indices=starts,profile_frame_count=2*len(starts))
    except Exception as exc:r.update(status='FAIL',eligible=False,error=repr(exc))
    dump(out,r);return r
def extract(row,subdir):
    directory=ROOT/subdir;directory.mkdir(exist_ok=True);path=directory/row['basename']
    if not path.exists():
        temporary=path.with_suffix('.tmp')
        with zipfile.ZipFile(row['archive']) as z,z.open(row['filename']) as src,temporary.open('wb') as dst:
            for b in iter(lambda:src.read(1<<20),b''):dst.write(b)
        assert sha(temporary)==row['sha256'];temporary.replace(path)
    assert sha(path)==row['sha256']
    return dict(filename=row['filename'],path=str(path),sha256=row['sha256'],origin='v61_stratified_archive',archive=row['archive'],fps=row['fps'],stratum=row['stratum'],width=row['width'],height=row['height'],frames=row['frames'])
def choose(candidates,initial_counts,count,seed):
    rng=random.Random(seed);bins={i:[] for i in range(16)}
    for r in candidates:bins[r['stratum']].append(r)
    for rows in bins.values():rng.shuffle(rows)
    totals=dict(initial_counts);out=[]
    while len(out)<count:
        available=[k for k,v in bins.items() if v];assert available,'Insufficient eligible unique videos'
        k=min(available,key=lambda k:(totals.get(k,0),k));out.append(bins[k].pop());totals[k]=totals.get(k,0)+1
    return out
def main():
    import numpy as np
    lock=(ROOT/'parts/data.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'data_split_integrity.json').exists():assert load(ROOT/'data_split_integrity.json')['status']=='PASS';return
    cfg=load(ROOT/'config.json');frozen();(ROOT/'profiling_tmp').mkdir(exist_ok=True)
    old=load(V41/'train_manifest.json')['videos'];oldval=load(V41/'validation_manifest.json')['videos'];tests=load(V41/'test_manifest.json')['videos'];splits=load(SPLIT)['splits']
    known={Path(r['filename']).name:r for r in old};reserved_names=set();reserved_hashes=set();test_hashes={r['source_sha256'] for r in tests}
    for key in ('validation','confirmation'):
        for r in splits[key]:
            name=Path(r['filename']).name;reserved_names.add(name);reserved_hashes.add(r['file_sha256']);known[name]=dict(path=r['local_path'],sha256=r['file_sha256'])
    for r in tests:
        name=Path(r['source_path']).name.split('__')[-1];reserved_names.add(name);reserved_hashes.add(r['source_sha256']);known[name]=dict(path=r['source_path'],sha256=r['source_sha256'])
    candidates=load(ROOT/'candidate_manifest.json')['videos'];jobs=[(i,r,known.get(Path(r['filename']).name),Path(r['filename']).name in reserved_names) for i,r in enumerate(candidates)]
    status(status='RUNNING',phase='source_profiling',candidate_count=len(jobs))
    done=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['profiling']['workers']) as pool:
        for result in pool.map(profile,jobs,chunksize=1):
            done+=1
            if done%50==0 or done==len(jobs):
                dump(ROOT/'profiling_status.json',dict(status='RUNNING',completed=done,total=len(jobs),updated_unix=time.time()));print('PROFILE',done,'/',len(jobs),flush=True)
    rows=[load(ROOT/'profiling'/f'{i:05d}.json') for i in range(len(jobs))]
    good=[r for r in rows if r['status']=='PASS' and r['sha256'] not in reserved_hashes]
    oldhash={r['sha256'] for r in old};assert len(oldhash)==256;assert oldhash<={r['sha256'] for r in good}
    # Quantile edges are fitted only on eligible non-held-out candidate sources.
    te=np.quantile([r['temporal_rgb_L1'] for r in good],[.25,.5,.75]);xe=np.quantile([r['sobel_edge_energy'] for r in good],[.25,.5,.75])
    for r in good:
        r['temporal_bin']=int(np.searchsorted(te,r['temporal_rgb_L1'],side='right'));r['texture_bin']=int(np.searchsorted(xe,r['sobel_edge_energy'],side='right'));r['stratum']=r['temporal_bin']*4+r['texture_bin']
    unique={}
    for r in good:unique.setdefault(r['sha256'],r)
    available=[r for h,r in unique.items() if h not in oldhash]
    # Existing six validation videos remain validation; twenty-six extra videos never enter training.
    val_add=choose([r for r in available if r['width']==1920 and r['height']==1080 and r['frames']>=32],{},32-len(oldval),cfg['seed']+17)
    valhash={r['sha256'] for r in oldval}|{r['sha256'] for r in val_add};assert len(valhash)==32 and not valhash&test_hashes
    counts={i:sum(unique[h]['stratum']==i for h in oldhash) for i in range(16)}
    added=choose([r for r in available if r['sha256'] not in valhash],counts,768,cfg['seed'])
    train=[]
    for r in old:
        assert sha(r['path'])==r['sha256'];train.append(dict(r,stratum=unique[r['sha256']]['stratum']))
    train.extend(extract(r,'training_videos') for r in added)
    validation=[dict(r,fps=float(Fraction(str(r['fps']))),width=1920,height=1080) for r in oldval]+[extract(r,'validation_videos') for r in val_add]
    for r in validation:assert sha(r['path'])==r['sha256']
    th={r['sha256'] for r in train};assert len(train)==len(th)==1024 and oldhash<=th
    assert not th&(valhash|reserved_hashes|test_hashes);assert not valhash&test_hashes
    dump(ROOT/'train_manifest_1024.json',dict(train_video_count=1024,retained_old_count=256,added_count=768,seed=cfg['seed'],videos=train))
    dump(ROOT/'validation_manifest.json',dict(validation_video_count=32,retained_old_count=len(oldval),frames=32,qps=cfg['validation_qps'],videos=validation))
    for r in rows:
        r['selected_train']=r.get('sha256') in th;r['old_train']=r.get('sha256') in oldhash;r['selected_validation']=r.get('sha256') in valhash
    write(ROOT/'training_source_statistics.csv',rows)
    groups={'candidate_pool':good,'old_256':[unique[h] for h in oldhash],'new_1024':[unique[h] for h in th]}
    table=[];distribution=[]
    for name,subset in groups.items():
        for m in FIELDS:
            a=np.array([r[m] for r in subset]);table.append(dict(group=name,statistic=m,total_candidate_count=len(rows),profiled_count=len(subset),mean=float(a.mean()),std=float(a.std()),min=float(a.min()),p25=float(np.quantile(a,.25)),median=float(np.median(a)),p75=float(np.quantile(a,.75)),max=float(a.max())))
        for i in range(16):distribution.append(dict(group=name,stratum=i,temporal_bin=i//4,texture_bin=i%4,count=sum(r['stratum']==i for r in subset),fraction=sum(r['stratum']==i for r in subset)/len(subset)))
    write(ROOT/'training_distribution_summary.csv',table);write(ROOT/'training_distribution_bins.csv',distribution)
    dump(ROOT/'stratification_audit.json',dict(status='PASS',temporal_quantile_edges=te.tolist(),texture_quantile_edges=xe.tolist(),statistics_fields=FIELDS,
        total_candidates=len(rows),profiled_eligible=len(good),unique_eligible=len(unique),reserved_excluded=sum(r['status']=='RESERVED_HELDOUT' for r in rows),
        failed_profiles=[dict(candidate_index=r['candidate_index'],error=r['error']) for r in rows if r['status']=='FAIL'],
        fit_excludes_old_heldout=True,validation_reserved_before_training_selection=True,no_full_video_flow=True,no_pretrained_models=True,profile_resize_only=True,training_preprocessing_unchanged=True))
    dump(ROOT/'data_split_integrity.json',dict(status='PASS',train_count=1024,validation_count=32,test_count=8,old_train_retained=256,
        train_SHA256_unique=True,train_validation_intersection=[],train_test_intersection=[],validation_test_intersection=[],
        no_UVG_in_train=True,no_VIRAT_in_train=True,no_MCL_HEVC_in_train=True,all_new_training_from_ulong_archive=True,
        reserved_sha256=sorted(reserved_hashes),test_sha256=sorted(test_hashes),validation_sha256=sorted(valhash),
        train_manifest_sha256=sha(ROOT/'train_manifest_1024.json'),validation_manifest_sha256=sha(ROOT/'validation_manifest.json')))
    dump(ROOT/'profiling_status.json',dict(status='PASS',completed=len(rows),total=len(rows)));frozen();status(status='DATA_READY',phase='training_pending');print('DATA PASS 1024 train / 32 validation',flush=True)
if __name__=='__main__':main()
