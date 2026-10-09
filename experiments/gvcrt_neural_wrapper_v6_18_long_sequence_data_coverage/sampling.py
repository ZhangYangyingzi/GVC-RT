"""Deterministic plans; independent CPU decode tasks preserve RNG order."""
import concurrent.futures,random
from io18 import *
def verify_sample(job):
    import cv2,numpy as np
    from PIL import Image
    cv2.setNumThreads(1)
    record,v,poolname=job;dest=ROOT/'audits/sample_records'/poolname/f'{record["adaptation_step"]:04d}.json'
    if dest.exists():
        old=load(dest);assert all(old[k]==v for k,v in record.items());return old
    d=record['domain'];indices=record['canonical_indices'];x,y=record['crop_x'],record['crop_y'];arr=[];full=[]
    if d=='ulong':
        cap=cv2.VideoCapture(v['path']);cap.set(cv2.CAP_PROP_POS_FRAMES,indices[0])
        try:
            for i in indices:
                ok,bgr=cap.read();assert ok,(v['path'],i);a=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);full.append(hashlib.sha256(a.tobytes()).hexdigest());arr.append(a[y:y+256,x:x+256].copy())
        finally:cap.release()
        banned=set(load(ROOT/'audits/forbidden_content_windows.json')['consecutive16_frame_digest_sha256']);assert hashlib.sha256(''.join(full).encode()).hexdigest() not in banned,('removed source content',v['filename'],indices)
    else:
        for i in indices:
            p=Path(v['input_dir'])/f'frame_{i:06d}.png';assert sha(p)==v['frame_file_sha256'][i]
            with Image.open(p) as im:a=np.asarray(im,dtype=np.uint8)
            h=hashlib.sha256(a.tobytes()).hexdigest();assert h==v['frame_rgb_sha256'][i];full.append(h);arr.append(a[y:y+256,x:x+256].copy())
    assert len(arr)==16 and all(a.shape==(256,256,3) for a in arr)
    result=dict(record,full_RGB_sha256=full,cropped_RGB_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in arr]);dump(dest,result);return result
def make_plans():
    rng=random.Random(20261010);order=[(d,q) for d in ('ulong','uvg') for q in range(10) for _ in range(50)];rng.shuffle(order)
    dump(ROOT/'sampling_order.json',dict(seed=20261010,order=order,domain_counts={'ulong':500,'uvg':500},per_domain_QP_counts={str(q):50 for q in range(10)}))
    sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src'))
    from gvc_hooks import INDEX_MAP
    from src.models.video_model_gvcrt import qp_shift
    for poolname,suffix,seed in [('AB','',20261011),('C','_expanded',20261012)]:
        dest=ROOT/'plans'/f'{poolname}.json'
        if dest.exists():assert load(dest)['sampling_order_sha256']==sha(ROOT/'sampling_order.json');continue
        vr=random.Random(seed);records={d:load(ROOT/f'train_manifest_{d}{suffix}.json')['videos'] for d in ('ulong','uvg')};jobs=[]
        for step,(d,q) in enumerate(order,1):
            v=vr.choice(records[d]);start=vr.randrange(v['frames']-15);x=vr.randrange(v['width']-256+1);y=vr.randrange(v['height']-256+1);indices=list(range(start,start+16))
            r=dict(adaptation_step=step,domain=d,video=v['filename' if d=='ulong' else 'name'],source_path=v['path' if d=='ulong' else 'source_path'],source_sha256=v['sha256' if d=='ulong' else 'source_sha256'],canonical_indices=indices,source_frame_indices=indices if d=='ulong' else v['source_frame_indices'][start:start+16],crop_x=x,crop_y=y,crop_size=256,external_qp=q,actual_qps=[q if p==0 else q+qp_shift[INDEX_MAP[p%len(INDEX_MAP)]] for p in range(16)],supervised_window_positions=list(range(1,16)),normalization_denominator=15);jobs.append((r,v,poolname))
        rows=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for r in pool.map(verify_sample,jobs):
                rows.append(r)
                if len(rows)%50==0:print('PLAN',poolname,len(rows),flush=True);dump(ROOT/'preparation_status.json',dict(phase='sampling_plans',plan=poolname,updates=len(rows),updated_unix=time.time(),CPU_workers=8))
        dump(dest,dict(updates=rows,sampling_order_sha256=sha(ROOT/'sampling_order.json'),video_selection='uniform independent video within domain',seed=seed))
    for name in ('AB','C'):
        rows=load(ROOT/'plans'/f'{name}.json')['updates'];assert len(rows)==1000
        for d in ('ulong','uvg'):
            assert sum(r['domain']==d for r in rows)==500
            assert all(sum(r['domain']==d and r['external_qp']==q for r in rows)==50 for q in range(10))
    dump(ROOT/'audits/plan_integrity.json',dict(status='PASS',AB_sha256=sha(ROOT/'plans/AB.json'),C_sha256=sha(ROOT/'plans/C.json'),same_domain_QP_order=True,AB_exact_shared=True,updates=1000,P_targets_per_update=15,P_targets_per_branch=15000))
