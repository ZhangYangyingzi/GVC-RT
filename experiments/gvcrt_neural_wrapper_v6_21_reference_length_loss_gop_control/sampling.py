"""Deterministic plans; independent CPU decode tasks preserve RNG order."""
import concurrent.futures,random
from io21 import *
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
        banned=set(load(ROOT/'audits/forbidden_content_windows.json')['consecutive16_frame_digest_sha256']);assert all(hashlib.sha256(''.join(full[i:i+16]).encode()).hexdigest() not in banned for i in range(49)),('removed source content',v['filename'],indices)
    else:
        for i in indices:
            p=Path(v['input_dir'])/f'frame_{i:06d}.png';assert sha(p)==v['frame_file_sha256'][i]
            with Image.open(p) as im:a=np.asarray(im,dtype=np.uint8)
            h=hashlib.sha256(a.tobytes()).hexdigest();assert h==v['frame_rgb_sha256'][i];full.append(h);arr.append(a[y:y+256,x:x+256].copy())
    assert len(arr)==64 and all(a.shape==(256,256,3) for a in arr)
    result=dict(record,full_RGB_sha256=full,cropped_RGB_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in arr]);dump(dest,result);return result

def make_plan():
    rng=random.Random(20261011);order=[(d,q) for d in ('ulong','uvg') for q in range(10) for _ in range(50)];rng.shuffle(order)
    dump(ROOT/'sampling_order.json',dict(seed=20261011,order=order,QP_sampling_rule='inherited v620 balanced50/domain/QP; shuffled',domain_counts={'ulong':500,'uvg':500}))
    dest=ROOT/'plans/shared.json'
    if dest.exists():assert load(dest)['sampling_order_sha256']==sha(ROOT/'sampling_order.json');return
    sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src'))
    from gvc_hooks import INDEX_MAP
    from src.models.video_model_gvcrt import qp_shift
    records={d:load(ROOT/f'train_manifest_{d}.json')['videos'] for d in ('ulong','uvg')};jobs=[]
    for step,(d,q) in enumerate(order,1):
        v=rng.choice(records[d]);start=rng.randrange(v['frames']-63);x=rng.randrange(v['width']-255);y=rng.randrange(v['height']-255);indices=list(range(start,start+64))
        target=[q+qp_shift[INDEX_MAP[p%len(INDEX_MAP)]] for p in range(49,64)]
        assert target==[q+qp_shift[INDEX_MAP[p%len(INDEX_MAP)]] for p in range(1,16)]
        r=dict(adaptation_step=step,domain=d,video=v['filename' if d=='ulong' else 'name'],source_path=v['path' if d=='ulong' else 'source_path'],source_sha256=v['sha256' if d=='ulong' else 'source_sha256'],canonical_indices=indices,source_frame_indices=indices if d=='ulong' else v['source_frame_indices'][start:start+64],crop_x=x,crop_y=y,crop_size=256,external_qp=q,actual_qps=target,supervised_window_positions=list(range(49,64)),normalization_denominator=15)
        jobs.append((r,v,'shared'))
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for r in pool.map(verify_sample,jobs):
            rows.append(r)
            if len(rows)%50==0:print('PLAN',len(rows),flush=True);dump(ROOT/'preparation_status.json',dict(phase='sampling',completed=len(rows),expected=1000))
    dump(dest,dict(updates=rows,seed=20261011,sampling_order_sha256=sha(ROOT/'sampling_order.json'),video_selection='uniform within domain',shared_ABC=True))
    dump(ROOT/'audits/plan_integrity.json',dict(status='PASS',updates=1000,domain_updates={d:sum(r['domain']==d for r in rows) for d in records},per_domain_QP_counts={d:{str(q):sum(r['domain']==d and r['external_qp']==q for r in rows) for q in range(10)} for d in records},plan_sha256=sha(dest),target_QP_phase_offset48_equivalent=True))
